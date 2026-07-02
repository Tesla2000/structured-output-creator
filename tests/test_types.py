from __future__ import annotations

from structured_output_creator._types import _ProviderType


def test_provider_type_values() -> None:
    assert _ProviderType.openai.value == "openai"
    assert _ProviderType.claude.value == "claude"


def test_provider_type_is_str() -> None:
    assert isinstance(_ProviderType.openai, str)
    assert isinstance(_ProviderType.claude, str)
