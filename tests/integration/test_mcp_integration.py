"""Integration tests for MCP support using real MCP servers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import patch

import nest_asyncio2
import pytest

from tests.conftest import create_test_config, create_test_perf_logger
from whai.llm.provider import LLMProvider


@pytest.mark.anyio
class TestMCPIntegration:
    """Integration tests for MCP with LLM provider and executor."""

    @pytest.fixture(autouse=True, scope="class")
    def enable_nest_asyncio(self):
        """Enable nest_asyncio2 for tests in this class that need it."""
        # Only apply once for the class, and only when needed
        # This allows asyncio.run() to work within async test contexts.
        # nest_asyncio2 is used (not nest-asyncio) for Python 3.14 compatibility.
        try:
            nest_asyncio2.apply()
        except RuntimeError:
            # Already applied, ignore
            pass
        yield
        # Note: apply() is persistent, no cleanup needed

    async def test_mcp_tools_in_llm_provider(self, mcp_server_time):
        """Test that MCP tools are discovered and included in LLM provider."""
        config = create_test_config()
        perf_logger = create_test_perf_logger()
        provider = LLMProvider(config, perf_logger=perf_logger)

        # Verify MCP tools are discovered
        # We need to call _get_mcp_tools in a way that works with async context
        # Since we're in an async test, we can't use asyncio.run()
        # Instead, verify the manager is initialized and can get tools
        if provider._mcp_manager is None:
            from whai.mcp.manager import MCPManager

            provider._mcp_manager = MCPManager()

        try:
            if provider._mcp_manager.is_enabled():
                tools = await provider._mcp_manager.get_all_tools()
                assert len(tools) > 0
                assert all(
                    tool["function"]["name"].startswith("mcp_") for tool in tools
                )
        finally:
            if provider._mcp_manager:
                await provider._mcp_manager.close_all()

    async def test_mcp_manager_initialization(self, mcp_server_time):
        """Test MCP manager initialization with real server."""
        from whai.mcp.manager import MCPManager

        manager = MCPManager()
        assert manager.is_enabled()
        await manager.initialize()
        assert len(manager.clients) > 0

        try:
            tools = await manager.get_all_tools()
            assert len(tools) > 0
            assert all(tool["function"]["name"].startswith("mcp_") for tool in tools)
        finally:
            await manager.close_all()

    async def test_mcp_tool_call_execution(self, mcp_server_time):
        """Test executing MCP tool call through manager."""
        from whai.mcp.manager import MCPManager

        manager = MCPManager()
        await manager.initialize()
        try:
            tools = await manager.get_all_tools()
            assert len(tools) > 0

            tool_name = tools[0]["function"]["name"]
            result = await manager.call_tool(tool_name, {})
            assert isinstance(result, dict)
        finally:
            await manager.close_all()

    async def test_mcp_executor_integration(self, mcp_server_time):
        """Test MCP executor integration function (sync handler with non-running loop)."""
        from whai.mcp.executor import handle_mcp_tool_call_sync
        from whai.mcp.manager import MCPManager

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                manager = MCPManager()
                loop.run_until_complete(manager.initialize())
                try:
                    tools = loop.run_until_complete(manager.get_all_tools())
                    assert len(tools) > 0
                    tool_call = {
                        "id": "call_123",
                        "name": tools[0]["function"]["name"],
                        "arguments": {"timezone": "UTC"},
                    }
                    return handle_mcp_tool_call_sync(tool_call, manager, loop=loop)
                finally:
                    loop.run_until_complete(manager.close_all())
            finally:
                loop.close()

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(pool, run_in_thread)

        assert "tool_call_id" in result
        assert "output" in result
        assert result["tool_call_id"] == "call_123"
        assert "error" not in result["output"].lower()

    async def test_mcp_error_handling(self, mcp_server_time):
        """Test error handling for invalid MCP tool calls (sync handler with non-running loop)."""
        from whai.mcp.executor import handle_mcp_tool_call_sync
        from whai.mcp.manager import MCPManager

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                manager = MCPManager()
                loop.run_until_complete(manager.initialize())
                try:
                    tool_call = {
                        "id": "call_123",
                        "name": "mcp_invalid-server_invalid-tool",
                        "arguments": {},
                    }
                    return handle_mcp_tool_call_sync(tool_call, manager, loop=loop)
                finally:
                    loop.run_until_complete(manager.close_all())
            finally:
                loop.close()

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(pool, run_in_thread)

        assert "tool_call_id" in result
        assert "output" in result
        assert (
            "error" in result["output"].lower() or "failed" in result["output"].lower()
        )

    @pytest.mark.integration
    @pytest.mark.api
    def test_llm_calls_mcp_tool_real_api(self, mcp_server_time):
        """
        Test that a real LLM actually calls an MCP tool when given a task that requires it.

        This test uses a real LLM API to verify observable behavior: when asked "What time is it?",
        the LLM should receive MCP tools, choose to call the time tool, and the tool should be
        executed successfully.

        Requires a valid API key in the config file or environment.
        """
        import os

        from whai.core.executor import run_conversation_loop
        from whai.mcp.executor import handle_mcp_tool_call_sync

        # Read API key and model strictly from environment for tests
        api_key = os.environ.get("OPENAI_API_KEY")

        # Skip if no API key in environment or if it's a dummy/test key
        if not api_key or api_key in ("test-key-123", "test-key", "your-api-key-here"):
            pytest.skip("No valid OPENAI_API_KEY in environment")

        model = os.environ.get("WHAI_TEST_OPENAI_MODEL", "gpt-5-mini")

        config = create_test_config(
            default_provider="openai",
            default_model=model,
            api_key=api_key,
        )

        provider = LLMProvider(config, perf_logger=create_test_perf_logger())

        # Verify MCP is enabled via config check (no async initialization here — pre-initializing
        # with asyncio.run() would close the event loop and kill the subprocess connections,
        # causing a hang when run_conversation_loop creates a new loop and tries to reuse them).
        from whai.mcp.manager import MCPManager

        probe = MCPManager()
        assert probe.is_enabled(), "MCP should be enabled for this test"

        # Track MCP tool calls by wrapping the executor's MCP handler
        mcp_tool_calls_made = []
        tools_sent_to_llm = []

        def tracked_handle_mcp_tool_call(tool_call, mcp_manager, loop=None):
            """Track MCP tool calls while still executing them."""
            mcp_tool_calls_made.append(
                {
                    "name": tool_call.get("name"),
                    "arguments": tool_call.get("arguments", {}),
                }
            )
            return handle_mcp_tool_call_sync(tool_call, mcp_manager, loop)

        # Track tools sent to LLM by patching litellm.completion to inspect kwargs
        original_completion = None
        try:
            import litellm

            original_completion = litellm.completion
        except ImportError:
            pass

        def tracked_completion(**kwargs):
            if "tools" in kwargs and kwargs["tools"]:
                tools_sent_to_llm.extend(kwargs["tools"])
            return original_completion(**kwargs)

        # Create messages that should trigger MCP tool usage
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. When the user asks about time, use the available MCP tools to get the current time.",
            },
            {"role": "user", "content": "What time is it right now?"},
        ]

        # Run the conversation loop with all necessary patches
        approval_patches = [
            # Auto-approve MCP tool calls (approve_tool is bound in executor's namespace)
            patch("whai.core.executor.approve_tool", return_value=True),
            # Auto-reject shell command approval loop
            patch("whai.core.executor.approval_loop", return_value=None),
            # Track MCP tool call execution
            patch(
                "whai.mcp.executor.handle_mcp_tool_call_sync",
                side_effect=tracked_handle_mcp_tool_call,
            ),
            # Suppress UI output
            patch("whai.ui.console.print"),
            patch("whai.ui.print_output"),
            patch("whai.ui.info"),
            patch("whai.ui.warn"),
            patch("whai.ui.error"),
            patch("whai.ui.spinner"),
        ]

        if original_completion:
            approval_patches.append(
                patch("litellm.completion", side_effect=tracked_completion)
            )

        with ExitStack() as stack:
            for p in approval_patches:
                stack.enter_context(p)
            run_conversation_loop(provider, messages, timeout=30)

        # Verify MCP tools were sent to LLM
        mcp_tools_sent = [
            t
            for t in tools_sent_to_llm
            if t.get("function", {}).get("name", "").startswith("mcp_")
        ]
        assert len(mcp_tools_sent) > 0, (
            f"MCP tools should have been sent to LLM. Tools sent: {[t.get('function', {}).get('name') for t in tools_sent_to_llm]}"
        )

        # Verify that an MCP tool was actually called
        assert len(mcp_tool_calls_made) > 0, (
            f"LLM should have called an MCP tool. MCP tools available: {[t.get('function', {}).get('name') for t in mcp_tools_sent]}"
        )

        # Verify it was a time-related tool
        time_tool_called = any(
            "time" in call["name"].lower() for call in mcp_tool_calls_made
        )
        assert time_tool_called, (
            f"LLM should have called a time tool. Calls made: {mcp_tool_calls_made}"
        )

        # Clean up MCP connections
        if provider._mcp_manager:

            async def cleanup():
                await provider._mcp_manager.close_all()

            asyncio.run(cleanup())
