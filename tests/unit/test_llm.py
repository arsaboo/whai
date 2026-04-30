"""Tests for LLM module."""

import os

import pytest

from tests.conftest import create_test_config, create_test_perf_logger
from whai import llm
from whai.configuration.user_config import (
    AnthropicConfig,
    AzureOpenAIConfig,
    LMStudioConfig,
    OllamaConfig,
    OpenAIAPIConfig,
    OpenAIConfig,
)


def test_get_base_system_prompt_deep_context():
    """Test base system prompt with deep context."""
    prompt = llm.get_base_system_prompt(is_deep_context=True)
    assert "terminal scrollback" in prompt
    assert "commands and their output" in prompt
    assert "whai" in prompt
    assert "execute_shell" in prompt
    assert "Emit at most one execute_shell tool call per response" in prompt
    # Should include system information
    assert "System:" in prompt
    assert "OS:" in prompt
    assert "DateTime:" in prompt


def test_get_base_system_prompt_shallow_context():
    """Test base system prompt with shallow context."""
    prompt = llm.get_base_system_prompt(is_deep_context=False)
    assert "command history" in prompt
    assert "commands only, no command outputs" in prompt
    # Should include system information
    assert "System:" in prompt
    assert "OS:" in prompt
    assert "DateTime:" in prompt


def test_get_base_system_prompt_with_timeout():
    """Test base system prompt includes timeout information when provided."""
    prompt = llm.get_base_system_prompt(is_deep_context=True, timeout=60)
    assert "60 seconds timeout" in prompt
    assert "doesn't finish executing in that time it will be interrupted" in prompt


def test_get_base_system_prompt_without_timeout():
    """Test base system prompt doesn't include timeout information when not provided."""
    prompt = llm.get_base_system_prompt(is_deep_context=True, timeout=None)
    assert "seconds timeout" not in prompt


def test_command_only_system_prompt_is_different_and_contains_execute_shell_focus():
    """Command-only system prompt should be tailored for command-only behavior."""
    base_prompt = llm.get_base_system_prompt(is_deep_context=True)
    # New command-only prompt function (to be implemented) must load a different template.
    command_only_prompt = llm.get_command_only_system_prompt(is_deep_context=True)

    # Both prompts should include the context note and system info.
    assert "System:" in command_only_prompt
    assert "OS:" in command_only_prompt
    assert "DateTime:" in command_only_prompt

    # Command-only prompt should differ from the base prompt.
    assert command_only_prompt != base_prompt

    # It should emphasize execute_shell and command-only behavior.
    assert "execute_shell" in command_only_prompt
    assert "command-only" in command_only_prompt.lower() or "command only" in command_only_prompt.lower()
    # It should not mention MCP tools, which are irrelevant in this mode.
    assert "mcp" not in command_only_prompt.lower()


def test_execute_shell_tool_schema():
    """Test that the execute_shell tool schema is valid."""
    tool = llm.EXECUTE_SHELL_TOOL

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "execute_shell"
    assert (
        "Emit at most one execute_shell call per response"
        in tool["function"]["description"]
    )
    assert "command" in tool["function"]["parameters"]["properties"]
    assert "command" in tool["function"]["parameters"]["required"]


def test_llm_provider_init():
    """Test LLMProvider initialization."""
    config = create_test_config(
        default_provider="openai",
        default_model="gpt-5-mini",
        api_key="test-key-123",
    )

    provider = llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert provider.configured_provider == "openai"
    assert provider.model == "gpt-5-mini"
    # Default: temperature should not be set for gpt-5 models
    assert provider.temperature is None


def test_llm_provider_init_with_overrides():
    """Test LLMProvider initialization with overrides."""
    config = create_test_config(
        default_provider="openai",
        default_model="gpt-5-mini",
    )

    provider = llm.LLMProvider(
        config,
        model="gpt-5-mini",
        temperature=0.5,
        perf_logger=create_test_perf_logger(),
    )

    assert provider.model == "gpt-5-mini"
    assert provider.temperature == 0.5


@pytest.mark.integration
@pytest.mark.api
def test_send_message_real_api():
    """
    Integration test with real API.

    Requires a valid API key in the config file or environment.
    Checks config file first to avoid environment pollution from other tests.
    """
    import os

    api_key = os.environ.get("OPENAI_API_KEY")

    # Skip if no API key from env, or if it's a dummy/test key
    # Note: "test-key" is returned by test mode when config doesn't exist
    if not api_key or api_key in ("test-key-123", "test-key", "your-api-key-here"):
        pytest.skip("No valid OpenAI API key in environment")

    config = create_test_config(
        default_provider="openai",
        default_model="gpt-5-mini",
        api_key=api_key,
    )

    provider = llm.LLMProvider(config, perf_logger=create_test_perf_logger())
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Say "test successful" and nothing else.'},
    ]

    result = provider.send_message(messages, stream=False, tools=[])

    assert "test successful" in result["content"].lower()


@pytest.mark.integration
@pytest.mark.api
def test_send_message_mistral_real_api():
    """
    Integration test with real Mistral API.

    Requires a valid Mistral API key in the config file or environment.
    Checks config file first to avoid environment pollution from other tests.
    """
    import os

    api_key = os.environ.get("MISTRAL_API_KEY")

    # Skip if no API key from env, or if it's a dummy/test key
    # Note: "test-key" is returned by test mode when config doesn't exist
    if not api_key or api_key in (
        "test-key-123",
        "test-key",
        "test-mistral-key",
        "your-api-key-here",
    ):
        pytest.skip("No valid Mistral API key in environment")

    config = create_test_config(
        default_provider="mistral",
        default_model="mistral-small-latest",
        api_key=api_key,
    )

    provider = llm.LLMProvider(config, perf_logger=create_test_perf_logger())
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Say "Mistral test successful" and nothing else.'},
    ]

    result = provider.send_message(messages, stream=False, tools=[])

    assert "mistral test successful" in result["content"].lower()


# ============================================================================
# Environment Variable Configuration Tests
# ============================================================================


def _clear_provider_env_vars():
    """
    Clear all provider-related environment variables.

    This ensures tests start with a clean environment and can verify
    that only the expected variables are set by the provider.
    """
    env_vars_to_clear = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "OLLAMA_API_BASE",
        "LM_STUDIO_API_BASE",
        "LM_STUDIO_API_KEY",
    ]
    for var in env_vars_to_clear:
        os.environ.pop(var, None)


def test_configure_api_keys_openai_sets_openai_key():
    """Test that OpenAI provider sets OPENAI_API_KEY environment variable."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="openai",
        default_model="gpt-4",
        api_key="sk-test-openai-key",
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("OPENAI_API_KEY") == "sk-test-openai-key"
    # Other provider keys should not be set
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "LM_STUDIO_API_BASE" not in os.environ


def test_configure_api_keys_anthropic_sets_anthropic_key():
    """Test that Anthropic provider sets ANTHROPIC_API_KEY environment variable."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="anthropic",
        default_model="claude-3-opus",
        api_key="sk-ant-test-anthropic-key",
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-anthropic-key"
    # Other provider keys should not be set
    assert "OPENAI_API_KEY" not in os.environ
    assert "LM_STUDIO_API_BASE" not in os.environ


def test_configure_api_keys_gemini_sets_gemini_key():
    """Test that Gemini provider sets GEMINI_API_KEY environment variable."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="gemini",
        default_model="gemini-2.5-flash",
        api_key="AIza-test-gemini-key",
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("GEMINI_API_KEY") == "AIza-test-gemini-key"
    # Other provider keys should not be set
    assert "OPENAI_API_KEY" not in os.environ
    assert "LM_STUDIO_API_BASE" not in os.environ


def test_configure_api_keys_mistral_sets_mistral_key():
    """Test that Mistral provider sets MISTRAL_API_KEY environment variable."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="mistral",
        default_model="mistral-small-latest",
        api_key="test-mistral-key",
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("MISTRAL_API_KEY") == "test-mistral-key"
    # Other provider keys should not be set
    assert "OPENAI_API_KEY" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_configure_api_keys_azure_sets_azure_vars():
    """Test that Azure OpenAI provider sets all Azure environment variables."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="azure_openai",
        default_model="gpt-4",
        providers={
            "azure_openai": AzureOpenAIConfig(
                api_key="test-azure-key",
                api_base="https://test.openai.azure.com",
                api_version="2023-05-15",
                default_model="gpt-4",
            )
        },
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("AZURE_API_KEY") == "test-azure-key"
    assert os.environ.get("AZURE_API_BASE") == "https://test.openai.azure.com"
    assert os.environ.get("AZURE_API_VERSION") == "2023-05-15"
    # Other provider keys should not be set
    assert "OPENAI_API_KEY" not in os.environ


def test_configure_api_keys_ollama_sets_ollama_base():
    """Test that Ollama provider sets OLLAMA_API_BASE environment variable."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="ollama",
        default_model="mistral",
        providers={
            "ollama": OllamaConfig(
                api_base="http://localhost:11434",
                default_model="mistral",
            )
        },
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("OLLAMA_API_BASE") == "http://localhost:11434"
    # Other provider keys should not be set
    assert "OPENAI_API_KEY" not in os.environ
    assert "LM_STUDIO_API_BASE" not in os.environ


def test_configure_api_keys_lm_studio_sets_lm_studio_vars():
    """Test that LM Studio provider sets LM_STUDIO_API_BASE and LM_STUDIO_API_KEY."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="lm_studio",
        default_model="qwen3-30b",
        providers={
            "lm_studio": LMStudioConfig(
                api_base="http://localhost:1234/v1",
                default_model="qwen3-30b",
                api_key=None,  # No API key configured
            )
        },
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("LM_STUDIO_API_BASE") == "http://localhost:1234/v1"
    assert os.environ.get("LM_STUDIO_API_KEY") == ""  # Should default to empty string
    # Other provider keys should not be set
    assert "OPENAI_API_KEY" not in os.environ


def test_configure_api_keys_lm_studio_with_custom_key():
    """Test that LM Studio provider uses custom API key when provided."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="lm_studio",
        default_model="qwen3-30b",
        providers={
            "lm_studio": LMStudioConfig(
                api_base="http://localhost:1234/v1",
                default_model="qwen3-30b",
                api_key="custom-lm-studio-key",
            )
        },
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    assert os.environ.get("LM_STUDIO_API_BASE") == "http://localhost:1234/v1"
    assert os.environ.get("LM_STUDIO_API_KEY") == "custom-lm-studio-key"
    assert "OPENAI_API_KEY" not in os.environ


def test_configure_api_keys_openai_api_no_env_vars():
    """openai_api provider sets no global env vars (api_base/key passed per-call)."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="openai_api",
        default_model="llama3",
        providers={
            "openai_api": OpenAIAPIConfig(
                api_base="http://localhost:8080/v1",
                default_model="llama3",
                api_key=None,
            )
        },
    )

    provider = llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    # No global env vars should be touched
    assert "OPENAI_API_KEY" not in os.environ
    assert "LM_STUDIO_API_BASE" not in os.environ
    # But the provider must have stored api_base and model
    assert provider.api_base == "http://localhost:8080/v1"
    assert provider.model == "openai/llama3"


def test_configure_api_keys_openai_api_with_key():
    """openai_api provider stores api_key for direct injection into completion()."""
    _clear_provider_env_vars()

    config = create_test_config(
        default_provider="openai_api",
        default_model="llama3",
        providers={
            "openai_api": OpenAIAPIConfig(
                api_base="http://localhost:8080/v1",
                default_model="llama3",
                api_key="my-local-token",
            )
        },
    )

    provider = llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    # Real OpenAI key must not be overwritten
    assert "OPENAI_API_KEY" not in os.environ
    assert provider.api_key == "my-local-token"


def test_configure_api_keys_only_active_provider():
    """Test that only the active provider's environment variables are set."""
    _clear_provider_env_vars()

    # Create config with multiple providers
    config = create_test_config(
        default_provider="lm_studio",
        default_model="qwen3-30b",
        providers={
            "openai": OpenAIConfig(
                api_key="sk-openai-key-should-not-be-set",
                default_model="gpt-4",
            ),
            "anthropic": AnthropicConfig(
                api_key="sk-ant-anthropic-key-should-not-be-set",
                default_model="claude-3-opus",
            ),
            "lm_studio": LMStudioConfig(
                api_base="http://localhost:1234/v1",
                default_model="qwen3-30b",
            ),
        },
    )

    # Initialize with LM Studio as active provider
    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    # LM Studio keys should be set
    assert os.environ.get("LM_STUDIO_API_BASE") == "http://localhost:1234/v1"
    assert os.environ.get("LM_STUDIO_API_KEY") == ""

    # Other providers' keys should NOT be set (this is the key fix!)
    assert "OPENAI_API_KEY" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_configure_api_keys_switching_providers():
    """Test that switching providers correctly updates environment variables."""
    _clear_provider_env_vars()

    # Create config with multiple providers
    config = create_test_config(
        default_provider="openai",
        default_model="gpt-4",
        providers={
            "openai": OpenAIConfig(
                api_key="sk-openai-key",
                default_model="gpt-4",
            ),
            "lm_studio": LMStudioConfig(
                api_base="http://localhost:1234/v1",
                default_model="qwen3-30b",
            ),
        },
    )

    # First, use OpenAI
    provider = llm.LLMProvider(
        config, provider="openai", perf_logger=create_test_perf_logger()
    )
    assert os.environ.get("OPENAI_API_KEY") == "sk-openai-key"
    assert "LM_STUDIO_API_BASE" not in os.environ

    # Clear and switch to LM Studio
    for var in ["OPENAI_API_KEY", "LM_STUDIO_API_BASE", "LM_STUDIO_API_KEY"]:
        os.environ.pop(var, None)

    llm.LLMProvider(
        config, provider="lm_studio", perf_logger=create_test_perf_logger()
    )
    assert os.environ.get("LM_STUDIO_API_BASE") == "http://localhost:1234/v1"
    assert "OPENAI_API_KEY" not in os.environ


def test_configure_api_keys_no_keys_when_not_configured():
    """Test that environment variables are not set when provider has no keys."""
    _clear_provider_env_vars()

    # Ollama doesn't require API key, only api_base
    config = create_test_config(
        default_provider="ollama",
        default_model="mistral",
        providers={
            "ollama": OllamaConfig(
                api_base="http://localhost:11434",
                default_model="mistral",
            )
        },
    )

    llm.LLMProvider(config, perf_logger=create_test_perf_logger())

    # Should set api_base
    assert os.environ.get("OLLAMA_API_BASE") == "http://localhost:11434"
    # Should not set any API keys
    assert "OPENAI_API_KEY" not in os.environ


# ============================================================================
# End-to-End Integration Tests (Require Running Services)
# ============================================================================

# (file continues unchanged)
