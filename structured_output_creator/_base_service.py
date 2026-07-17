from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Protocol, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, Field, create_model

from structured_output_creator._cache import (
    _make_default_cache,
    _ResponseCache,
)
from structured_output_creator._models import _ErrorObject, _Message, _Role

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ScalarT = TypeVar("ScalarT", bound=str | int | float | bool)
_ValueT = TypeVar("_ValueT")


class _ValueHolder(Protocol[_ValueT]):
    value: _ValueT


class _BaseService(BaseModel, ABC, Generic[SchemaT]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    model: str
    cache: _ResponseCache = Field(
        default_factory=_make_default_cache,
    )

    @overload
    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT],
        *,
        use_cache: bool = False,
    ) -> SchemaT | _ErrorObject: ...

    @overload
    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[ScalarT],
        *,
        use_cache: bool = False,
    ) -> ScalarT | _ErrorObject: ...

    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT | ScalarT],
        *,
        use_cache: bool = False,
    ) -> SchemaT | ScalarT | _ErrorObject:
        messages = (
            [_Message(role=_Role.user, content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        if not issubclass(output_type, BaseModel):
            wrapper_type = create_model("Model", value=(output_type, ...))
            wrapped = self.create_structured_output(
                messages,
                cast("type[SchemaT]", wrapper_type),
                use_cache=use_cache,
            )
            if isinstance(wrapped, _ErrorObject):
                return wrapped
            return cast("_ValueHolder[ScalarT]", wrapped).value
        key = _ResponseCache.make_key(messages, output_type, self)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return output_type.model_validate(cached)
        result = self._generate(messages, output_type)
        if use_cache and not isinstance(result, _ErrorObject):
            self.cache.set(key, result.model_dump())
        return result

    @overload
    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT],
        *,
        use_cache: bool = False,
    ) -> SchemaT | _ErrorObject: ...

    @overload
    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[ScalarT],
        *,
        use_cache: bool = False,
    ) -> ScalarT | _ErrorObject: ...

    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[SchemaT | ScalarT],
        *,
        use_cache: bool = False,
    ) -> SchemaT | ScalarT | _ErrorObject:
        messages = (
            [_Message(role=_Role.user, content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        if not issubclass(output_type, BaseModel):
            wrapper_type = create_model("Model", value=(output_type, ...))
            wrapped = await self.create_structured_output_async(
                messages,
                cast("type[SchemaT]", wrapper_type),
                use_cache=use_cache,
            )
            if isinstance(wrapped, _ErrorObject):
                return wrapped
            return cast("_ValueHolder[ScalarT]", wrapped).value
        key = _ResponseCache.make_key(messages, output_type, self)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return output_type.model_validate(cached)
        result = await self._generate_async(messages, output_type)
        if use_cache and not isinstance(result, _ErrorObject):
            self.cache.set(key, result.model_dump())
        return result

    @abstractmethod
    def _generate(
        self,
        messages: list[_Message],
        output_type: type[SchemaT],
    ) -> SchemaT | _ErrorObject: ...

    @abstractmethod
    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[SchemaT],
    ) -> SchemaT | _ErrorObject: ...
