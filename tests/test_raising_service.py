from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from structured_output_creator._base_service import PydanticType, _BaseService
from structured_output_creator._models import (
    _ErrorObject,
    _Message,
    _NoContentError,
    _RefusalError,
    LLMError,
    LLMNoContentError,
    LLMRefusalError,
)
from structured_output_creator._raising_service import _RaisingService


class _Output(BaseModel):
    name: str


class _EchoService(_BaseService):
    model: str = "test"
    return_error: _ErrorObject | None = None

    def _generate(
        self,
        _messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType | _ErrorObject:
        if self.return_error is not None:
            return self.return_error
        return output_type.model_validate({"name": "ok"})

    async def _generate_async(
        self,
        _messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType | _ErrorObject:
        if self.return_error is not None:
            return self.return_error
        return output_type.model_validate({"name": "ok"})


def test_happy_path_returns_result() -> None:
    service = _RaisingService(service=_EchoService())
    result = service.create_structured_output("hello", _Output)
    assert isinstance(result, _Output)
    assert result.name == "ok"


def test_happy_path_async_returns_result() -> None:
    service = _RaisingService(service=_EchoService())
    result = asyncio.run(
        service.create_structured_output_async("hello", _Output)
    )
    assert isinstance(result, _Output)
    assert result.name == "ok"


def test_refusal_raises_llm_refusal_error() -> None:
    service = _RaisingService(
        service=_EchoService(
            return_error=_RefusalError(message="policy violation")
        )
    )
    with pytest.raises(LLMRefusalError) as exc_info:
        service.create_structured_output("hello", _Output)
    assert exc_info.value.message == "policy violation"


def test_refusal_raises_llm_refusal_error_async() -> None:
    service = _RaisingService(
        service=_EchoService(
            return_error=_RefusalError(message="async refusal")
        )
    )
    with pytest.raises(LLMRefusalError) as exc_info:
        asyncio.run(service.create_structured_output_async("hello", _Output))
    assert exc_info.value.message == "async refusal"


def test_no_content_raises_llm_no_content_error() -> None:
    service = _RaisingService(
        service=_EchoService(return_error=_NoContentError(message="empty"))
    )
    with pytest.raises(LLMNoContentError):
        service.create_structured_output("hello", _Output)


def test_no_content_raises_llm_no_content_error_async() -> None:
    service = _RaisingService(
        service=_EchoService(return_error=_NoContentError())
    )
    with pytest.raises(LLMNoContentError):
        asyncio.run(service.create_structured_output_async("hello", _Output))


def test_llm_refusal_error_is_subclass_of_llm_error() -> None:
    assert issubclass(LLMRefusalError, LLMError)


def test_llm_no_content_error_is_subclass_of_llm_error() -> None:
    assert issubclass(LLMNoContentError, LLMError)
