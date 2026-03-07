"""Conversation loop execution for whai."""

import json
import re
from typing import Any, Dict, List, Optional

from whai import ui
from whai.constants import TOOL_OUTPUT_MAX_TOKENS
from whai.core.session_logger import SessionLogger
from whai.interaction import approval_loop, execute_command
from whai.llm import LLMProvider
from whai.llm.token_utils import truncate_text_with_tokens
from whai.logging_setup import get_logger
from whai.utils import PerformanceLogger

logger = get_logger(__name__)


NO_TOOL_CALL_RECOVERY_MAX_RETRIES = 1
NO_TOOL_CALL_RECOVERY_HINT = (
    "Your previous response suggested running a shell command, but you did not emit a tool call. "
    "If another command is needed, call the execute_shell tool with a valid JSON argument containing a non-empty 'command' field. "
    "If no command is needed and the task is complete, reply with a final answer only."
)


def _looks_like_continuation_without_tool_call(text: str) -> bool:
    """Heuristic: detect when assistant text implies another command should be run."""
    if not text:
        return False

    normalized = " ".join(text.lower().split())
    continuation_patterns = (
        r"\bi(?:\s*'ll|\s+will)\s+(?:run|check|inspect|look|try|execute|analyze|dig)\b",
        r"\blet\s+me\s+(?:run|check|inspect|look|try|execute|analyze|dig)\b",
        r"\bnext\s+i(?:\s*'ll|\s+will)\b",
        r"\bi\s+need\s+to\s+(?:run|check|inspect|look)\b",
    )
    return any(re.search(pattern, normalized) for pattern in continuation_patterns)


def _looks_like_final_answer(text: str) -> bool:
    """Heuristic: detect whether the assistant likely intended to finish."""
    if not text:
        return False

    normalized = " ".join(text.lower().split())
    final_markers = (
        "task is complete",
        "you're all set",
        "that should do it",
        "done.",
        "done!",
        "no further action",
        "no more commands",
    )
    return any(marker in normalized for marker in final_markers)


def run_conversation_loop(
    llm_provider: LLMProvider,
    messages: List[Dict[str, Any]],
    timeout: int,
    command_string: Optional[str] = None,
) -> None:
    """
    Run the main conversation loop with the LLM.

    Args:
        llm_provider: Configured LLM provider instance.
        messages: Initial conversation messages.
        timeout: Command timeout in seconds.
        command_string: Optional full command string for logging (e.g., "whai -vv 'query'").
    """
    # Initialize session logger for context capture in whai shell
    session_logger = SessionLogger(console=ui.console)
    
    # Log the whai command itself if provided
    if command_string and session_logger.enabled:
        session_logger.log_command(command_string)
    
    loop_iteration = 0
    no_tool_call_retries = 0
    next_tool_choice = None
    while True:
        loop_iteration += 1
        loop_perf = PerformanceLogger(f"Conversation Loop (iteration {loop_iteration})")
        loop_perf.start()
        
        try:
            # Send to LLM with streaming; show spinner until first chunk arrives
            with ui.spinner("Thinking"):
                response = llm_provider.send_message(
                    messages,
                    stream=True,
                    tool_choice=next_tool_choice,
                )
                next_tool_choice = None
                if isinstance(response, dict):
                    raise RuntimeError("Expected streaming response but received non-streaming payload")
                response_stream = response
                response_chunks: List[Dict[str, Any]] = []
                first_chunk: Optional[Dict[str, Any]] = None
                for chunk in response_stream:
                    first_chunk = chunk
                    break
            loop_perf.log_section("LLM API call (streaming)")

            # Print first chunk and continue streaming
            if first_chunk is not None:
                response_chunks.append(first_chunk)
                if first_chunk["type"] == "text":
                    session_logger.print(first_chunk["content"], end="", soft_wrap=True)
            for chunk in response_stream:
                response_chunks.append(chunk)
                if chunk["type"] == "text":
                    session_logger.print(chunk["content"], end="", soft_wrap=True)
            if any(c["type"] == "text" for c in response_chunks):
                session_logger.print()

            # Extract tool calls from chunks
            tool_calls = [c for c in response_chunks if c["type"] == "tool_call"]
            assistant_content = "".join(
                c["content"] for c in response_chunks if c["type"] == "text"
            )
            logger.debug(
                "Received %d tool calls from stream",
                len(tool_calls),
                extra={"category": "api"},
            )
            loop_perf.log_section("Response parsing", extra_info={"tool_calls": len(tool_calls)})

            if not tool_calls:
                # If model text implies another command but no tool call was emitted,
                # retry once with explicit tool-use guidance.
                if (
                    no_tool_call_retries < NO_TOOL_CALL_RECOVERY_MAX_RETRIES
                    and assistant_content
                    and _looks_like_continuation_without_tool_call(assistant_content)
                    and not _looks_like_final_answer(assistant_content)
                ):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_content,
                        }
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": NO_TOOL_CALL_RECOVERY_HINT,
                        }
                    )
                    no_tool_call_retries += 1
                    next_tool_choice = "required"
                    ui.warn(
                        "Model response indicated more actions but contained no tool call. "
                        "Requesting a corrected tool call."
                    )
                    loop_perf.log_complete(extra_info={"ended": "tool_recovery_retry"})
                    continue

                if (
                    assistant_content
                    and _looks_like_continuation_without_tool_call(assistant_content)
                    and no_tool_call_retries >= NO_TOOL_CALL_RECOVERY_MAX_RETRIES
                ):
                    ui.info(
                        "Model did not produce a runnable tool call after retry. "
                        "Ending conversation."
                    )

                # No tool calls, conversation is done
                loop_perf.log_complete(extra_info={"ended": "no_tool_calls"})
                break

            # Reset retry budget after a normal tool-call turn.
            no_tool_call_retries = 0

            # Process each tool call
            tool_results = []
            for tool_call in tool_calls:
                if tool_call["name"] == "execute_shell":
                    command = tool_call["arguments"].get("command", "")

                    if not command:
                        tool_results.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "output": "Invalid execute_shell tool call: missing or empty 'command' argument.",
                            }
                        )
                        logger.warning(
                            "Skipping execute_shell tool call with empty command (id=%s)",
                            tool_call["id"],
                        )
                        continue

                    # Get user approval
                    approved_command = approval_loop(command)
                    loop_perf.log_section("Command approval")

                    if approved_command is None:
                        # User rejected
                        tool_results.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "output": "Command rejected by user.",
                            }
                        )
                        continue

                    # Execute the command
                    try:
                        logger.debug(
                            "Executing approved command: %s",
                            approved_command,
                            extra={"category": "cmd"},
                        )
                        
                        # Log command to session for context
                        session_logger.log_command(approved_command)
                        
                        with ui.spinner("Executing command..."):
                            stdout, stderr, returncode = execute_command(
                                approved_command, timeout=timeout
                            )
                        loop_perf.log_section(
                            "Command execution",
                            extra_info={"command": approved_command, "exit_code": returncode},
                        )
                        
                        # Log command output to session for context
                        session_logger.log_command_output(stdout, stderr, returncode)

                        # Format the result for LLM (plain text)
                        result = f"Command: {approved_command}\n"
                        result += f"Exit code: {returncode}\n"
                        if stdout:
                            result += f"\nOutput:\n{stdout}"
                        if stderr:
                            result += f"\nErrors:\n{stderr}"
                        if not stdout and not stderr:
                            result += "\nOutput: (empty - command produced no output)"

                        # Truncate tool output if needed to respect token limits
                        truncated_result, was_truncated = truncate_text_with_tokens(
                            result, TOOL_OUTPUT_MAX_TOKENS
                        )
                        loop_perf.log_section(
                            "Tool output truncation",
                            extra_info={"truncated": was_truncated},
                        )
                        if was_truncated:
                            ui.warn(
                                f"Command output for '{approved_command}' was truncated to fit token limits. "
                                "Recent output has been preserved."
                            )

                        tool_results.append(
                            {"tool_call_id": tool_call["id"], "output": truncated_result}
                        )

                        # Display the output (pretty formatted)
                        ui.console.print()
                        ui.print_output(stdout, stderr, returncode)
                        ui.console.print()

                    except Exception as e:
                        error_text = str(e)
                        logger.exception("Command execution failed: %s", e)
                        # Surface error to user
                        ui.error(f"Failed to execute command: {error_text}")

                        # Log failure to session for deep context capture
                        # This ensures subsequent commands will have context about the failure
                        if "timed out" in error_text.lower():
                            session_logger.log_command_failure(error_text, timeout=timeout)
                            timeout_note = (
                                f"Command: {approved_command}\n\n"
                                f"OUTPUT: NO OUTPUT, {timeout}s TIMEOUT EXCEEDED"
                            )
                            tool_results.append(
                                {
                                    "tool_call_id": tool_call["id"],
                                    "output": timeout_note,
                                }
                            )
                        else:
                            session_logger.log_command_failure(error_text)
                            tool_results.append(
                                {
                                    "tool_call_id": tool_call["id"],
                                    "output": f"Failed to execute command: {error_text}",
                                }
                            )

            # Decide whether to end the conversation
            all_rejected = tool_results and all(
                "rejected" in r["output"].lower() for r in tool_results
            )

            if not tool_results and tool_calls:
                # Tool calls existed but none were runnable (e.g., empty/missing command)
                ui.info("No runnable tool calls were produced (missing command).")
                loop_perf.log_complete(extra_info={"ended": "no_runnable_tool_calls"})
                break

            if not tool_results or all_rejected:
                ui.console.print("\nConversation ended.")
                loop_perf.log_complete(extra_info={"ended": "all_rejected"})
                break

            # Build assistant message for history
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }

            # Add tool_calls to assistant message if present
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ]

            messages.append(assistant_message)

            # Add tool results to messages
            for result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["output"],
                    }
                )

            loop_perf.log_section("Message history update", extra_info={"tool_results": len(tool_results)})
            loop_perf.log_complete(extra_info={"tool_calls": len(tool_calls), "tool_results": len(tool_results)})

            # Continue loop to get LLM's next response

        except KeyboardInterrupt:
            ui.console.print("\n\nInterrupted by user.")
            loop_perf.log_complete(extra_info={"ended": "keyboard_interrupt"})
            break
        except Exception as e:
            import traceback

            text = str(e)
            # Check for LLM-related errors (API errors, model errors, auth errors, etc.)
            if (
                "LLM API error" in text
                or "Model" in text and "provider" in text
                or "Authentication failed" in text
                or "Permission denied" in text
                or "Rate limit" in text
                or "Network or service error" in text
            ):
                # Show concise, helpful message for provider/model/auth errors
                ui.error(text)
                ui.info(
                    "Run 'whai --interactive-config' to review your keys and model."
                )
                # Keep full details in logs only
                logger.exception("LLM error in conversation loop: %s", e)
                loop_perf.log_complete(extra_info={"ended": "llm_error"})
                break
            else:
                ui.error(f"Unexpected error: {e}")
                ui.error(f"Details: {traceback.format_exc()}")
                logger.exception("Unexpected error in conversation loop: %s", e)
                loop_perf.log_complete(extra_info={"ended": "unexpected_error"})
                break
