"""Tests for MCP client using real MCP servers."""

import asyncio

import pytest

from whai.mcp.client import MCPClient


@pytest.mark.anyio
class TestMCPClient:
    """Tests for MCP client with real MCP server."""

    async def test_client_initialization(self, mcp_server_time):
        """Test MCP client can be initialized."""
        client = MCPClient(
            server_name=mcp_server_time["server_name"],
            command=mcp_server_time["command"],
            args=mcp_server_time["args"],
            env=mcp_server_time["env"],
        )
        assert client.server_name == mcp_server_time["server_name"]
        assert client.command == mcp_server_time["command"]

    async def test_list_tools(self, mcp_server_time):
        """Test discovering tools from real MCP server."""
        client = MCPClient(
            server_name=mcp_server_time["server_name"],
            command=mcp_server_time["command"],
            args=mcp_server_time["args"],
            env=mcp_server_time["env"],
        )
        await client.connect()
        try:
            tools = await client.list_tools()
            assert len(tools) > 0
            assert all(tool["type"] == "function" for tool in tools)
            assert all("function" in tool for tool in tools)
            assert all(tool["function"]["name"].startswith("mcp_") for tool in tools)
        finally:
            await client.close()

    async def test_call_tool(self, mcp_server_time):
        """Test calling a tool on real MCP server."""
        client = MCPClient(
            server_name=mcp_server_time["server_name"],
            command=mcp_server_time["command"],
            args=mcp_server_time["args"],
            env=mcp_server_time["env"],
        )
        await client.connect()
        try:
            tools = await client.list_tools()
            assert len(tools) > 0

            # Find a tool that doesn't require arguments (or use appropriate args)
            tool_name = tools[0]["function"]["name"]
            # Some tools require arguments, so we test with empty dict and handle validation errors
            result = await client.call_tool(tool_name, {})
            assert isinstance(result, dict)
            assert "content" in result or "isError" in result
        finally:
            await client.close()

    async def test_call_tool_with_prefix(self, mcp_server_time):
        """Test calling tool with mcp_ prefix."""
        client = MCPClient(
            server_name=mcp_server_time["server_name"],
            command=mcp_server_time["command"],
            args=mcp_server_time["args"],
            env=mcp_server_time["env"],
        )
        await client.connect()
        try:
            tools = await client.list_tools()
            assert len(tools) > 0

            tool_name = tools[0]["function"]["name"]
            result = await client.call_tool(tool_name, {})
            assert isinstance(result, dict)
            assert "content" in result or "isError" in result
        finally:
            await client.close()

    async def test_invalid_tool_name(self, mcp_server_time):
        """Test calling invalid tool name raises error."""
        client = MCPClient(
            server_name=mcp_server_time["server_name"],
            command=mcp_server_time["command"],
            args=mcp_server_time["args"],
            env=mcp_server_time["env"],
        )
        await client.connect()
        try:
            # MCP server may return error result instead of raising, so check for error in result
            result = await client.call_tool("invalid_tool_name", {})
            assert isinstance(result, dict)
            # Result should indicate an error (either isError flag or error in content)
            assert result.get("isError", False) or any(
                "error" in str(item).lower() for item in result.get("content", [])
            )
        finally:
            await client.close()

    async def test_close_and_reconnect(self, mcp_server_time):
        """Test that closing and reconnecting allows tool discovery again."""
        client = MCPClient(
            server_name=mcp_server_time["server_name"],
            command=mcp_server_time["command"],
            args=mcp_server_time["args"],
            env=mcp_server_time["env"],
        )
        await client.connect()
        tools_before = await client.list_tools()
        await client.close()

        # After reconnect, should be able to list tools again
        await client.connect()
        try:
            tools_after = await client.list_tools()
            assert len(tools_after) == len(tools_before)
        finally:
            await client.close()

