from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict

from structured_output_creator._cache import _ResponseCache, _default_cache
from structured_output_creator._models import _Message, _Role

T = TypeVar("T", bound=BaseModel)
PydanticType = TypeVar("PydanticType", bound=BaseModel)


class _BaseService(BaseModel, ABC):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    model: str

    def create_structured_output(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[T],
        *,
        use_cache: bool = False,
    ) -> T:
        messages = (
            [_Message(role=_Role.user, content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        if not issubclass(output_type, BaseModel):

            class Model(BaseModel):
                value: output_type  # type: ignore[valid-type]

            result = self.create_structured_output(
                messages, Model, use_cache=use_cache
            )
            return result.value
        key = _ResponseCache.make_key(
            messages, output_type, type(self).__name__, self.model
        )
        if use_cache:
            cached = _default_cache.get(key)
            if cached is not None:
                return output_type.model_validate(cached)
        result = self._generate(messages, output_type)
        if use_cache:
            _default_cache.set(key, result.model_dump())
        return result

    async def create_structured_output_async(
        self,
        prompt_or_messages: str | list[_Message],
        output_type: type[T],
        *,
        use_cache: bool = False,
    ) -> T:
        messages = (
            [_Message(role=_Role.user, content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        if not issubclass(output_type, BaseModel):

            class Model(BaseModel):
                value: output_type  # type: ignore[valid-type]

            result = await self.create_structured_output_async(
                messages, Model, use_cache=use_cache
            )
            return result.value
        key = _ResponseCache.make_key(
            messages, output_type, type(self).__name__, self.model
        )
        if use_cache:
            cached = _default_cache.get(key)
            if cached is not None:
                return output_type.model_validate(cached)
        result = await self._generate_async(messages, output_type)
        if use_cache:
            _default_cache.set(key, result.model_dump())
        return result

    @abstractmethod
    def _generate(
        self,
        messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType: ...

    @abstractmethod
    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType: ...
