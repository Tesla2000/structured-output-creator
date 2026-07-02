from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from structured_output_creator._base_service import _BaseService, PydanticType
from structured_output_creator._cache import _ResponseCache, _default_cache
from structured_output_creator._models import _Message, _Role


class _ConcreteService(_BaseService):
    model: str = "test-model"

    def _generate(
        self,
        _messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType:
        return output_type.model_validate({"name": "generated"})

    async def _generate_async(
        self,
        _messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType:
        return output_type.model_validate({"name": "async-generated"})


class _Output(BaseModel):
    name: str


def test_string_prompt_converted_to_message() -> None:
    service = _ConcreteService()
    result = service.create_structured_output(
        "hello", _Output, use_cache=False
    )
    assert result.name == "generated"


def test_message_list_passed_through() -> None:
    service = _ConcreteService()
    messages = [_Message(role=_Role.user, content="hi")]
    result = service.create_structured_output(
        messages, _Output, use_cache=False
    )
    assert result.name == "generated"


def test_use_cache_false_skips_cache() -> None:
    service = _ConcreteService()
    with patch.object(_ResponseCache, _ResponseCache.get.__name__) as mock_get:
        service.create_structured_output("hello", _Output, use_cache=False)
        mock_get.assert_not_called()


def test_cache_hit_returns_without_generate() -> None:
    service = _ConcreteService()
    key = _ResponseCache.make_key(
        [_Message(role=_Role.user, content="cached")],
        _Output,
        "_ConcreteService",
        "test-model",
    )
    _default_cache.set(key, {"name": "from-cache"})

    with patch.object(
        _ConcreteService, _ConcreteService._generate.__name__
    ) as mock_gen:
        result = service.create_structured_output(
            "cached", _Output, use_cache=True
        )
        mock_gen.assert_not_called()

    assert result.name == "from-cache"
    _default_cache.data.pop(key, None)


def test_cache_miss_calls_generate_and_stores() -> None:
    service = _ConcreteService()
    with (
        patch.object(
            _ResponseCache, _ResponseCache.get.__name__, return_value=None
        ),
        patch.object(_ResponseCache, _ResponseCache.set.__name__) as mock_set,
    ):
        result = service.create_structured_output(
            "miss", _Output, use_cache=True
        )
        mock_set.assert_called_once()
    assert result.name == "generated"


def test_non_pydantic_type_wrapped_in_model() -> None:
    class _StrService(_ConcreteService):
        def _generate(
            self,
            _messages: list[_Message],
            output_type: type[PydanticType],
        ) -> PydanticType:
            return output_type.model_validate({"value": "hello"})

        async def _generate_async(
            self,
            _messages: list[_Message],
            output_type: type[PydanticType],
        ) -> PydanticType:
            return output_type.model_validate({"value": "hello"})

    service = _StrService()
    result = service.create_structured_output("prompt", str, use_cache=False)  # type: ignore[arg-type]
    assert result == "hello"


def test_base_service_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        _BaseService(model="x")  # type: ignore[abstract]


def test_async_string_prompt_converted() -> None:
    service = _ConcreteService()
    result = asyncio.run(
        service.create_structured_output_async(
            "hello", _Output, use_cache=False
        )
    )
    assert result.name == "async-generated"


def test_async_message_list_passed_through() -> None:
    service = _ConcreteService()
    messages = [_Message(role=_Role.user, content="hi")]
    result = asyncio.run(
        service.create_structured_output_async(
            messages, _Output, use_cache=False
        )
    )
    assert result.name == "async-generated"


def test_async_cache_hit_skips_generate() -> None:
    service = _ConcreteService()
    key = _ResponseCache.make_key(
        [_Message(role=_Role.user, content="async-cached")],
        _Output,
        "_ConcreteService",
        "test-model",
    )
    _default_cache.set(key, {"name": "async-from-cache"})

    with patch.object(
        _ConcreteService, _ConcreteService._generate_async.__name__
    ) as mock_gen:
        result = asyncio.run(
            service.create_structured_output_async(
                "async-cached", _Output, use_cache=True
            )
        )
        mock_gen.assert_not_called()

    assert result.name == "async-from-cache"
    _default_cache.data.pop(key, None)


def test_async_cache_miss_stores_result() -> None:
    service = _ConcreteService()
    with (
        patch.object(
            _ResponseCache, _ResponseCache.get.__name__, return_value=None
        ),
        patch.object(_ResponseCache, _ResponseCache.set.__name__) as mock_set,
    ):
        result = asyncio.run(
            service.create_structured_output_async(
                "miss", _Output, use_cache=True
            )
        )
        mock_set.assert_called_once()
    assert result.name == "async-generated"


def test_async_non_pydantic_type_wrapped() -> None:
    class _AsyncStrService(_ConcreteService):
        def _generate(
            self,
            _messages: list[_Message],
            output_type: type[PydanticType],
        ) -> PydanticType:
            return output_type.model_validate({"value": "async-hello"})

        async def _generate_async(
            self,
            _messages: list[_Message],
            output_type: type[PydanticType],
        ) -> PydanticType:
            return output_type.model_validate({"value": "async-hello"})

    service = _AsyncStrService()
    result = asyncio.run(
        service.create_structured_output_async("prompt", str, use_cache=False)  # type: ignore[arg-type]
    )
    assert result == "async-hello"
