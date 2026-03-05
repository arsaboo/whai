"""Pytest configuration for whai tests."""

import os

import mcp.client.stdio  # noqa: F401 — force early import so default errlog=sys.stderr captures the real stderr, not Click's test wrapper
import pytest

from whai.configuration.user_config import (
    AnthropicConfig,
    GeminiConfig,
    LLMConfig,
    MistralConfig,
    OpenAIConfig,
    RolesConfig,
    WhaiConfig,
)
from whai.utils import PerformanceLogger


@pytest.fixture(scope="session", autouse=True)
def plain_mode_for_tests():
    """Force plain mode (no Rich styling) for all tests."""
    original = os.environ.get("WHAI_PLAIN")
    os.environ["WHAI_PLAIN"] = "1"
    yield
    if original is None:
        os.environ.pop("WHAI_PLAIN", None)
    else:
        os.environ["WHAI_PLAIN"] = original


@pytest.fixture(scope="session", autouse=True)
def test_mode_for_config():
    """Enable test mode for config loading."""
    from whai.constants import ENV_WHAI_TEST_MODE
    
    original = os.environ.get(ENV_WHAI_TEST_MODE)
    os.environ[ENV_WHAI_TEST_MODE] = "1"
    yield
    if original is None:
        os.environ.pop(ENV_WHAI_TEST_MODE, None)
    else:
        os.environ[ENV_WHAI_TEST_MODE] = original


def pytest_configure(config):
    """Configure pytest-anyio to only use asyncio backend."""
    os.environ.setdefault("ANYIO_BACKEND", "asyncio")
    
    # Try to configure pytest-anyio plugin directly
    try:
        import anyio
        # Force asyncio backend
        anyio._backend = "asyncio"
    except (ImportError, AttributeError):
        pass

def pytest_collection_modifyitems(config, items):
    """Skip trio backend tests."""
    for item in items:
        # Check if this is a parametrized test with trio backend
        if hasattr(item, "callspec") and item.callspec:
            params = item.callspec.params
            if "asynclib_name" in params and params["asynclib_name"] == "trio":
                # Skip this test variant
                skip_marker = pytest.mark.skip(reason="trio backend not available")
                item.add_marker(skip_marker)
        
        # Also check for anyio parametrization in the test name
        if "[trio]" in item.name:
            skip_marker = pytest.mark.skip(reason="trio backend not available")
            item.add_marker(skip_marker)


def create_test_config(
    default_provider: str = "openai",
    default_model: str = "gpt-5-mini",
    api_key: str = "test-key-123",
    providers: dict = None,
) -> WhaiConfig:
    """
    Helper function to create a test WhaiConfig object.
    
    Args:
        default_provider: Default provider name.
        default_model: Default model name.
        api_key: API key for the default provider.
        providers: Optional dict of provider configs to add.
    
    Returns:
        WhaiConfig instance for testing.
    """
    from whai.constants import DEFAULT_PROVIDER, DEFAULT_ROLE_NAME
    
    provider_configs = {}
    
    # Add default provider
    if default_provider == "openai":
        provider_configs["openai"] = OpenAIConfig(
            api_key=api_key,
            default_model=default_model,
        )
    elif default_provider == "anthropic":
        provider_configs["anthropic"] = AnthropicConfig(
            api_key=api_key,
            default_model=default_model,
        )
    elif default_provider == "gemini":
        provider_configs["gemini"] = GeminiConfig(
            api_key=api_key,
            default_model=default_model,
        )
    elif default_provider == "mistral":
        provider_configs["mistral"] = MistralConfig(
            api_key=api_key,
            default_model=default_model,
        )
    
    # Add any additional providers
    if providers:
        provider_configs.update(providers)
    
    return WhaiConfig(
        llm=LLMConfig(
            default_provider=default_provider or DEFAULT_PROVIDER,
            providers=provider_configs,
        ),
        roles=RolesConfig(default_role=DEFAULT_ROLE_NAME),
    )


def create_test_perf_logger() -> PerformanceLogger:
    """
    Helper function to create a test PerformanceLogger instance.
    
    Returns:
        PerformanceLogger instance for testing.
    """
    perf_logger = PerformanceLogger("Test")
    perf_logger.start()
    return perf_logger




@pytest.fixture(scope="session")
def _mcp_uvx_path():
    """Session-scoped validation that uvx and mcp-server-time are available.

    Runs the subprocess check once per session instead of per test.
    """
    import shutil
    import subprocess

    uvx_path = shutil.which("uvx")
    if not uvx_path:
        pytest.skip("uvx not available, cannot run MCP server tests")

    try:
        result = subprocess.run(
            [uvx_path, "mcp-server-time", "--help"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            pytest.skip("mcp-server-time not available via uvx")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Cannot run mcp-server-time")

    return uvx_path


@pytest.fixture
def mcp_server_time(_mcp_uvx_path, tmp_path, monkeypatch):
    """Fixture that configures a real MCP time server for testing.

    The heavy uvx availability check is done once by the session-scoped
    _mcp_uvx_path fixture; this fixture only creates the per-test config.
    """
    import json

    config_dir = tmp_path / "whai"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "mcp.json"

    config_data = {
        "mcpServers": {
            "time-server": {
                "command": _mcp_uvx_path,
                "args": ["mcp-server-time"],
                "env": {},
            }
        }
    }
    config_file.write_text(json.dumps(config_data))

    monkeypatch.setattr(
        "whai.configuration.user_config.get_config_dir", lambda: config_dir
    )

    yield {
        "server_name": "time-server",
        "command": _mcp_uvx_path,
        "args": ["mcp-server-time"],
        "env": {},
    }
