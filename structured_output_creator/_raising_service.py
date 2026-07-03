from __future__ import annotations

from typing import ClassVar, NoReturn, TypeVar

from pydantic import BaseModel, ConfigDict, InstanceOf

from structured_output_creator._base_service import _BaseService
from structured_output_creator._models import (
    _ErrorObject,
    _Message,
    _RefusalError,
    LLMNoContentError,
    LLMRefusalError,
)

T = TypeVar("T")


def _raise_for_error(error: _ErrorObject) -> NoReturn:
    if isinstance(error, _RefusalError):
        raise LLMRefusalError(error.message)
    raise LLMNoContentError(error.message)


class _RaisingService(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service: InstanceOf[_BaseService]

    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[T],
        *,
        use_cache: bool = False,
    ) -> T:
        result = self.service.create_structured_output(
            prompt_or_messages, output_type, use_cache=use_cache
        )
        if isinstance(result, _ErrorObject):
            _raise_for_error(result)
        return result

    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[T],
        *,
        use_cache: bool = False,
    ) -> T:
        result = await self.service.create_structured_output_async(
            prompt_or_messages, output_type, use_cache=use_cache
        )
        if isinstance(result, _ErrorObject):
            _raise_for_error(result)
        return result
