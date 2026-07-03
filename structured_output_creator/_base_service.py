from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, InstanceOf, create_model

from structured_output_creator._cache import (
    _ResponseCache,
    _make_default_cache,
)
from structured_output_creator._models import _ErrorObject, _Message, _Role

T = TypeVar("T")
PydanticType = TypeVar("PydanticType", bound=BaseModel)
_ValueT = TypeVar("_ValueT")


class _ValueHolder(Protocol[_ValueT]):
    value: _ValueT


class _BaseService(BaseModel, ABC):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    model: str
    cache: InstanceOf[_ResponseCache] = Field(
        default_factory=_make_default_cache, exclude=True
    )

    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[T],
        *,
        use_cache: bool = False,
    ) -> T | _ErrorObject:
        messages = (
            [_Message(role=_Role.user, content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        if not issubclass(output_type, BaseModel):
            wrapper_type = create_model("Model", value=(output_type, ...))
            wrapped = self.create_structured_output(
                messages, wrapper_type, use_cache=use_cache
            )
            if isinstance(wrapped, _ErrorObject):
                return wrapped
            return cast("_ValueHolder[T]", wrapped).value
        key = _ResponseCache.make_key(messages, output_type, self)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return output_type.model_validate(cached)
        result = self._generate(messages, output_type)
        if use_cache and not isinstance(result, _ErrorObject):
            self.cache.set(key, result.model_dump())
        return result

    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[T],
        *,
        use_cache: bool = False,
    ) -> T | _ErrorObject:
        messages = (
            [_Message(role=_Role.user, content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        if not issubclass(output_type, BaseModel):
            wrapper_type = create_model("Model", value=(output_type, ...))
            wrapped = await self.create_structured_output_async(
                messages, wrapper_type, use_cache=use_cache
            )
            if isinstance(wrapped, _ErrorObject):
                return wrapped
            return cast("_ValueHolder[T]", wrapped).value
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
        output_type: type[PydanticType],
    ) -> PydanticType | _ErrorObject: ...

    @abstractmethod
    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType | _ErrorObject: ...
