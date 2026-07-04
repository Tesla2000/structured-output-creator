from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

import structured_output_creator


def test_all_contains_base_exports() -> None:
    assert "Message" in structured_output_creator.__all__
    assert "Role" in structured_output_creator.__all__
    assert "ProviderType" in structured_output_creator.__all__
    assert "ErrorObject" in structured_output_creator.__all__
    assert "RefusalError" in structured_output_creator.__all__
    assert "NoContentError" in structured_output_creator.__all__
    assert "LLMError" in structured_output_creator.__all__
    assert "LLMRefusalError" in structured_output_creator.__all__
    assert "LLMNoContentError" in structured_output_creator.__all__
    assert "RaisingService" in structured_output_creator.__all__


def test_all_contains_openai_service() -> None:
    assert "OpenAIService" in structured_output_creator.__all__


def test_all_contains_claude_service() -> None:
    assert "ClaudeService" in structured_output_creator.__all__


def test_all_contains_any_provider_service() -> None:
    assert "AnyProviderService" in structured_output_creator.__all__


def test_openai_import_error_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(
        sys.modules, "structured_output_creator", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "structured_output_creator._openai", raising=False
    )

    with patch.dict(sys.modules, {"openai": None}):
        soc = importlib.import_module("structured_output_creator")
        assert "OpenAIService" not in soc.__all__

    monkeypatch.delitem(
        sys.modules, "structured_output_creator", raising=False
    )


def test_claude_import_error_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(
        sys.modules, "structured_output_creator", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "structured_output_creator._claude", raising=False
    )

    with patch.dict(sys.modules, {"anthropic": None}):
        soc = importlib.import_module("structured_output_creator")
        assert "ClaudeService" not in soc.__all__

    monkeypatch.delitem(
        sys.modules, "structured_output_creator", raising=False
    )
