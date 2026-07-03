from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from structured_output_creator import (
    ClaudeService,
    LLMError,
    LLMNoContentError,
    LLMRefusalError,
    RaisingService,
)

_MAX_TOKENS = 4096


class _Output(BaseModel):
    name: str


def _parsed_response(obj: _Output) -> MagicMock:
    return MagicMock(parsed_output=obj)


def _refusal_response(explanation: str) -> MagicMock:
    return MagicMock(
        parsed_output=None,
        stop_reason="refusal",
        stop_details=MagicMock(explanation=explanation),
    )


def _no_content_response() -> MagicMock:
    return MagicMock(
        parsed_output=None,
        stop_reason="tool_use",
        stop_details=None,
    )


def _make_service(return_value: MagicMock) -> ClaudeService:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = return_value
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=return_value
    )
    return ClaudeService.model_construct(
        client=mock_client,
        async_client=mock_async_client,
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )


def test_happy_path_returns_result() -> None:
    service = RaisingService(
        service=_make_service(_parsed_response(_Output(name="ok")))
    )
    result = service.create_structured_output("hello", _Output)
    assert isinstance(result, _Output)
    assert result.name == "ok"


def test_happy_path_async_returns_result() -> None:
    service = RaisingService(
        service=_make_service(_parsed_response(_Output(name="ok")))
    )
    result = asyncio.run(
        service.create_structured_output_async("hello", _Output)
    )
    assert isinstance(result, _Output)
    assert result.name == "ok"


def test_refusal_raises_llm_refusal_error() -> None:
    service = RaisingService(
        service=_make_service(_refusal_response("policy violation"))
    )
    with pytest.raises(LLMRefusalError) as exc_info:
        service.create_structured_output("hello", _Output)
    assert exc_info.value.message == "policy violation"


def test_refusal_raises_llm_refusal_error_async() -> None:
    service = RaisingService(
        service=_make_service(_refusal_response("async refusal"))
    )
    with pytest.raises(LLMRefusalError) as exc_info:
        asyncio.run(service.create_structured_output_async("hello", _Output))
    assert exc_info.value.message == "async refusal"


def test_no_content_raises_llm_no_content_error() -> None:
    service = RaisingService(service=_make_service(_no_content_response()))
    with pytest.raises(LLMNoContentError):
        service.create_structured_output("hello", _Output)


def test_no_content_raises_llm_no_content_error_async() -> None:
    service = RaisingService(service=_make_service(_no_content_response()))
    with pytest.raises(LLMNoContentError):
        asyncio.run(service.create_structured_output_async("hello", _Output))


def test_llm_refusal_error_is_subclass_of_llm_error() -> None:
    assert issubclass(LLMRefusalError, LLMError)


def test_llm_no_content_error_is_subclass_of_llm_error() -> None:
    assert issubclass(LLMNoContentError, LLMError)
