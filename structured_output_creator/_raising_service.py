from __future__ import annotations

from typing import ClassVar, Generic, NoReturn, overload

from pydantic import BaseModel, ConfigDict

from structured_output_creator._base_service import (
    ScalarT,
    SchemaT,
    _BaseService,
)
from structured_output_creator._models import (
    LLMNoContentError,
    LLMRefusalError,
    _ErrorObject,
    _Message,
    _RefusalError,
)


def _raise_for_error(error: _ErrorObject) -> NoReturn:
    if isinstance(error, _RefusalError):
        raise LLMRefusalError(error.message)
    raise LLMNoContentError(error.message)


class _RaisingService(BaseModel, Generic[SchemaT]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service: _BaseService[SchemaT]

    @overload
    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT],
        *,
        use_cache: bool = False,
    ) -> SchemaT: ...

    @overload
    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[ScalarT],
        *,
        use_cache: bool = False,
    ) -> ScalarT: ...

    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT | ScalarT],
        *,
        use_cache: bool = False,
    ) -> SchemaT | ScalarT:
        result = self.service.create_structured_output(
            prompt_or_messages, output_type, use_cache=use_cache
        )
        if isinstance(result, _ErrorObject):
            _raise_for_error(result)
        return result

    @overload
    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT],
        *,
        use_cache: bool = False,
    ) -> SchemaT: ...

    @overload
    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[ScalarT],
        *,
        use_cache: bool = False,
    ) -> ScalarT: ...

    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT | ScalarT],
        *,
        use_cache: bool = False,
    ) -> SchemaT | ScalarT:
        result = await self.service.create_structured_output_async(
            prompt_or_messages, output_type, use_cache=use_cache
        )
        if isinstance(result, _ErrorObject):
            _raise_for_error(result)
        return result
