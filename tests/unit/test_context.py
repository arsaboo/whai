"""Tests for context module."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whai import context
from whai.context.history import (
    CMDHandler,
    get_additional_context,
)


def test_get_context_prefers_tmux():
    """Test that get_context prefers tmux over history."""
    with (
        patch("whai.context.capture._get_tmux_context", return_value="tmux output"),
        patch("whai.context.capture._get_history_context", return_value="history output"),
        patch("whai.context.capture.get_additional_context", return_value=None),
    ):
        context_str, is_deep = context.get_context()

        assert context_str == "tmux output"
        assert is_deep is True


def test_get_context_falls_back_to_history():
    """Test that get_context falls back to history when tmux is unavailable."""
    with (
        patch("whai.context.capture._get_tmux_context", return_value=None),
        patch("whai.context.capture._get_history_context", return_value="history output"),
        patch("whai.context.capture.get_additional_context", return_value=None),
        patch.dict(os.environ, {}, clear=True),
    ):
        context_str, is_deep = context.get_context()

        assert context_str == "history output"
        assert is_deep is False


def test_get_context_combines_history_and_additional_context():
    """Test that get_context combines history and additional context."""
    with (
        patch("whai.context.capture._get_tmux_context", return_value=None),
        patch("whai.context.capture.read_session_context", return_value=None),
        patch("whai.context.capture._get_history_context", return_value="history output"),
        patch("whai.context.capture.get_additional_context", return_value="error output"),
    ):
        context_str, is_deep = context.get_context()

        assert "history output" in context_str
        assert "error output" in context_str
        assert is_deep is False


def test_get_context_no_context_available():
    """Test get_context when no context is available."""
    with (
        patch("whai.context.capture._get_tmux_context", return_value=None),
        patch("whai.context.capture._get_history_context", return_value=None),
        patch("whai.context.capture.get_additional_context", return_value=None),
        patch.dict(os.environ, {}, clear=True),
    ):
        context_str, is_deep = context.get_context()

        assert context_str == ""
        assert is_deep is False


def test_get_context_tmux_active_but_empty():
    """Test get_context when tmux is active but capture is empty (new session)."""
    with (
        patch("whai.context.capture._get_tmux_context", return_value=""),
        patch("whai.context.capture._get_history_context", return_value="history output"),
        patch("whai.context.capture.get_additional_context", return_value=None),
        patch.dict(os.environ, {"TMUX": "/tmp/tmux-1000/default,123,456"}),
    ):
        context_str, is_deep = context.get_context()

        # Should return empty string with is_deep=True to indicate tmux is active
        assert context_str == ""
        assert is_deep is True


def test_cmd_handler_get_history_context(monkeypatch):
    """Test CMDHandler.get_history_context() via doskey."""
    handler = CMDHandler(shell_name="cmd")
    
    mock_output = "dir\ncd projects\necho hello\n"
    
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output
        mock_run.return_value = mock_result
        
        monkeypatch.setattr("os.name", "nt")
        
        result = handler.get_history_context(max_commands=3)
        
        assert result is not None
        assert "Recent command history:" in result
        assert "dir" in result
        assert "cd projects" in result
        assert "echo hello" in result
        mock_run.assert_called_once_with(
            ["doskey", "/history"],
            capture_output=True,
            text=True,
            timeout=5,
        )


def test_cmd_handler_non_windows(monkeypatch):
    """Test CMDHandler returns None on non-Windows."""
    handler = CMDHandler(shell_name="cmd")
    
    monkeypatch.setattr("os.name", "posix")
    
    result = handler.get_history_context()
    
    assert result is None


def test_get_additional_context_unknown_shell():
    """Test get_additional_context() returns None for an unknown shell."""
    result = get_additional_context(shell="unknown_shell_xyz")
    assert result is None
