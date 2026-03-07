"""Integration tests for conversation-loop recovery behavior."""

from unittest.mock import MagicMock, patch

from whai.core.executor import run_conversation_loop
from whai.llm import LLMProvider


def _stream(chunks):
    for chunk in chunks:
        yield chunk


def test_no_tool_call_recovery_retries_with_required_tool_choice():
    """If assistant implies continuation but emits no tool call, loop retries once."""
    mock_provider = MagicMock(spec=LLMProvider)

    responses = [
        _stream([
            {"type": "text", "content": "Let me check the largest directories now."},
        ]),
        _stream([
            {"type": "text", "content": "Running command."},
            {
                "type": "tool_call",
                "id": "call_1",
                "name": "execute_shell",
                "arguments": {"command": "echo ok"},
            },
        ]),
        _stream([
            {"type": "text", "content": "Done."},
        ]),
    ]

    call_kwargs = []

    def _mock_send_message(messages, tools=None, stream=True, tool_choice=None):
        call_kwargs.append({"tool_choice": tool_choice, "stream": stream})
        return responses.pop(0)

    mock_provider.send_message.side_effect = _mock_send_message

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Clean up disk space."},
    ]

    with (
        patch("whai.core.executor.approval_loop", return_value="echo ok"),
        patch("whai.core.executor.execute_command", return_value=("ok\n", "", 0)) as mock_exec,
    ):
        run_conversation_loop(mock_provider, messages, timeout=30)

    assert len(call_kwargs) == 3
    assert call_kwargs[0]["tool_choice"] is None
    assert call_kwargs[1]["tool_choice"] == "required"
    assert call_kwargs[2]["tool_choice"] is None
    mock_exec.assert_called_once_with("echo ok", timeout=30)


def test_missing_command_tool_call_does_not_end_conversation():
    """A tool call without command should produce tool error and continue loop."""
    mock_provider = MagicMock(spec=LLMProvider)

    responses = [
        _stream([
            {"type": "text", "content": "Running diagnostics."},
            {
                "type": "tool_call",
                "id": "call_missing",
                "name": "execute_shell",
                "arguments": {},
            },
        ]),
        _stream([
            {"type": "text", "content": "Retrying with a valid command."},
            {
                "type": "tool_call",
                "id": "call_valid",
                "name": "execute_shell",
                "arguments": {"command": "echo hi"},
            },
        ]),
        _stream([
            {"type": "text", "content": "Done."},
        ]),
    ]

    def _mock_send_message(messages, tools=None, stream=True, tool_choice=None):
        return responses.pop(0)

    mock_provider.send_message.side_effect = _mock_send_message

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Clean up disk space."},
    ]

    with (
        patch("whai.core.executor.approval_loop", return_value="echo hi"),
        patch("whai.core.executor.execute_command", return_value=("hi\n", "", 0)) as mock_exec,
    ):
        run_conversation_loop(mock_provider, messages, timeout=30)

    assert mock_provider.send_message.call_count == 3
    mock_exec.assert_called_once_with("echo hi", timeout=30)


def test_no_tool_call_recovery_stops_after_max_retry():
    """Recovery should not loop forever when model keeps returning text-only turns."""
    mock_provider = MagicMock(spec=LLMProvider)

    responses = [
        _stream([
            {"type": "text", "content": "Let me check that now."},
        ]),
        _stream([
            {"type": "text", "content": "I will run another check next."},
        ]),
    ]

    call_kwargs = []

    def _mock_send_message(messages, tools=None, stream=True, tool_choice=None):
        call_kwargs.append({"tool_choice": tool_choice})
        return responses.pop(0)

    mock_provider.send_message.side_effect = _mock_send_message

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Clean up disk space."},
    ]

    run_conversation_loop(mock_provider, messages, timeout=30)

    assert len(call_kwargs) == 2
    assert call_kwargs[0]["tool_choice"] is None
    assert call_kwargs[1]["tool_choice"] == "required"
