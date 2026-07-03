from __future__ import annotations

from structured_output_creator import ProviderType


def test_provider_type_values() -> None:
    assert ProviderType.openai.value == "openai"
    assert ProviderType.claude.value == "claude"


def test_provider_type_is_str() -> None:
    assert isinstance(ProviderType.openai, str)
    assert isinstance(ProviderType.claude, str)
