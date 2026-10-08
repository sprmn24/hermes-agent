"""Test that fallback-chain identity dedup normalizes provider aliases.

A provider can appear under two ids that are the same backend: the canonical
models.dev id (`opencode`) and a rename alias (`opencode-zen`, the legacy
pre-rename name). `_entry_identity` must collapse these to one identity so the
fallback chain does not keep two routes to the same backend (which shifts the
chain by one slot and lets an alias slip past the same-backend skip).
"""

from hermes_cli.fallback_config import get_fallback_chain


def _entry(provider, model, base_url=""):
    entry = {"provider": provider, "model": model}
    if base_url:
        entry["base_url"] = base_url
    return entry


def test_fallback_chain_dedups_canonical_id_and_alias():
    config = {
        "fallback_providers": [
            _entry("opencode", "deepseek-v4-flash-free"),
            # ``opencode-zen`` is the legacy pre-rename name of ``opencode``
            # (same backend). It must not add a second slot in the chain.
            _entry("opencode-zen", "deepseek-v4-flash-free"),
            _entry("anthropic", "claude-sonnet-4"),
        ]
    }

    chain = get_fallback_chain(config)

    # The two opencode ids describe the same backend; expect a single slot.
    providers = [e["provider"] for e in chain]
    assert providers == ["opencode", "anthropic"], providers


def test_fallback_chain_keeps_configured_provider_with_endpoint():
    config = {
        "providers": {"openai": {"base_url": "https://api.openai.com/v1"}},
        "fallback_providers": [
            _entry("openrouter", "gpt-5"),
            # A configured ``providers.openai`` endpoint wins over the legacy
            # ``openai`` -> ``openrouter`` alias at runtime: a distinct route.
            _entry("openai", "gpt-5"),
        ],
    }

    chain = get_fallback_chain(config)

    providers = [e["provider"] for e in chain]
    assert providers == ["openrouter", "openai"], providers


def test_fallback_chain_collapses_endpointless_provider_block():
    config = {
        # No endpoint: tuning for the built-in provider, runtime still aliases.
        "providers": {"openai": {"stale_timeout_seconds": 600}},
        "fallback_providers": [
            _entry("openrouter", "gpt-5"),
            _entry("openai", "gpt-5"),
        ],
    }

    chain = get_fallback_chain(config)

    providers = [e["provider"] for e in chain]
    assert providers == ["openrouter"], providers


def test_fallback_chain_collapses_mixed_case_provider_key():
    config = {
        # Runtime looks up the lowercased id verbatim, so ``OpenAI`` never matches.
        "providers": {"OpenAI": {"base_url": "https://api.openai.com/v1"}},
        "fallback_providers": [
            _entry("openrouter", "gpt-5"),
            _entry("openai", "gpt-5"),
        ],
    }

    chain = get_fallback_chain(config)

    providers = [e["provider"] for e in chain]
    assert providers == ["openrouter"], providers


def test_fallback_chain_keeps_configured_local_alias():
    config = {
        "providers": {"ollama": {"base_url": "http://localhost:11434/v1"}},
        "fallback_providers": [
            _entry("vllm", "qwen3:8b"),
            # ``ollama`` and ``vllm`` both alias to ``custom``, but a configured
            # ``providers.ollama`` endpoint is its own route.
            _entry("ollama", "qwen3:8b"),
        ],
    }

    chain = get_fallback_chain(config)

    providers = [e["provider"] for e in chain]
    assert providers == ["vllm", "ollama"], providers


def test_fallback_chain_dedups_configured_provider_duplicates():
    config = {
        "providers": {"openai": {"base_url": "https://api.openai.com/v1"}},
        "fallback_providers": [
            _entry("openrouter", "gpt-5"),
            _entry("openai", "gpt-5"),
            # Same configured route after strip/lowercase; no second slot.
            _entry(" OPENAI ", "gpt-5"),
        ],
    }

    chain = get_fallback_chain(config)

    providers = [e["provider"] for e in chain]
    assert providers == ["openrouter", "openai"], providers


def test_fallback_chain_keeps_configured_provider_from_legacy_fallback_model():
    config = {
        "providers": {"openai": {"base_url": "https://api.openai.com/v1"}},
        "fallback_providers": [_entry("openrouter", "gpt-5")],
        # The legacy key merges after ``fallback_providers`` and is a distinct route.
        "fallback_model": _entry("openai", "gpt-5"),
    }

    chain = get_fallback_chain(config)

    providers = [e["provider"] for e in chain]
    assert providers == ["openrouter", "openai"], providers
