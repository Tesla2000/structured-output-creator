from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from structured_output_creator._base_service import (
    PydanticType,
    _BaseService,
)
from structured_output_creator._cache import _ResponseCache
from structured_output_creator._models import (
    _ErrorObject,
    _Message,
    _RefusalError,
    _Role,
)


class _ConcreteService(_BaseService):
    model: str = "test-model"
    response_name: str = "generated"

    def _generate(
        self,
        _messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType | _ErrorObject:
        if self.response_name == "__error__":
            return _RefusalError(message="nope")
        return output_type.model_validate({"name": self.response_name})

    async def _generate_async(
        self,
        _messages: list[_Message],
        output_type: type[PydanticType],
    ) -> PydanticType | _ErrorObject:
        if self.response_name == "__error__":
            return _RefusalError(message="nope")
        return output_type.model_validate(
            {"name": "async-" + self.response_name}
        )


class _Output(BaseModel):
    name: str


def test_field_value_used_in_generate() -> None:
    service = _ConcreteService(response_name="custom")
    result = service.create_structured_output(
        "hello", _Output, use_cache=False
    )
    assert isinstance(result, _Output)
    assert result.name == "custom"


def test_field_value_used_in_generate_async() -> None:
    service = _ConcreteService(response_name="custom")
    result = asyncio.run(
        service.create_structured_output_async(
            "hello", _Output, use_cache=False
        )
    )
    assert isinstance(result, _Output)
    assert result.name == "async-custom"


def test_error_object_from_generate_propagates() -> None:
    service = _ConcreteService(response_name="__error__")
    result = service.create_structured_output(
        "hello", _Output, use_cache=False
    )
    assert isinstance(result, _RefusalError)
    assert result.message == "nope"


def test_error_object_from_generate_is_not_cached() -> None:
    service = _ConcreteService(response_name="__error__")
    with patch.object(_ResponseCache, _ResponseCache.set.__name__) as mock_set:
        result = service.create_structured_output(
            "hello", _Output, use_cache=True
        )
        mock_set.assert_not_called()
    assert isinstance(result, _RefusalError)


def test_error_object_from_generate_async_propagates() -> None:
    service = _ConcreteService(response_name="__error__")
    result = asyncio.run(
        service.create_structured_output_async(
            "hello", _Output, use_cache=False
        )
    )
    assert isinstance(result, _RefusalError)


def test_string_prompt_converted_to_message() -> None:
    service = _ConcreteService()
    result = service.create_structured_output(
        "hello", _Output, use_cache=False
    )
    assert isinstance(result, _Output)
    assert result.name == "generated"


def test_message_list_passed_through() -> None:
    service = _ConcreteService()
    messages = [_Message(role=_Role.user, content="hi")]
    result = service.create_structured_output(
        messages, _Output, use_cache=False
    )
    assert isinstance(result, _Output)
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
        service,
    )
    service.cache.set(key, {"name": "from-cache"})

    with patch.object(
        _ConcreteService,
        _ConcreteService._generate.__name__,  # noqa: SLF001
    ) as mock_gen:
        result = service.create_structured_output(
            "cached", _Output, use_cache=True
        )
        mock_gen.assert_not_called()

    assert isinstance(result, _Output)
    assert result.name == "from-cache"
    service.cache.data.pop(key, None)


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
    assert isinstance(result, _Output)
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
    result = service.create_structured_output("prompt", str, use_cache=False)
    assert result == "hello"


def test_non_pydantic_type_propagates_error_object() -> None:
    service = _ConcreteService(response_name="__error__")
    result = service.create_structured_output("prompt", str, use_cache=False)
    assert isinstance(result, _RefusalError)


def test_async_non_pydantic_type_propagates_error_object() -> None:
    service = _ConcreteService(response_name="__error__")
    result = asyncio.run(
        service.create_structured_output_async("prompt", str, use_cache=False)
    )
    assert isinstance(result, _RefusalError)


def test_base_service_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        _BaseService(model="x")


def test_async_string_prompt_converted() -> None:
    service = _ConcreteService()
    result = asyncio.run(
        service.create_structured_output_async(
            "hello", _Output, use_cache=False
        )
    )
    assert isinstance(result, _Output)
    assert result.name == "async-generated"


def test_async_message_list_passed_through() -> None:
    service = _ConcreteService()
    messages = [_Message(role=_Role.user, content="hi")]
    result = asyncio.run(
        service.create_structured_output_async(
            messages, _Output, use_cache=False
        )
    )
    assert isinstance(result, _Output)
    assert result.name == "async-generated"


def test_async_cache_hit_skips_generate() -> None:
    service = _ConcreteService()
    key = _ResponseCache.make_key(
        [_Message(role=_Role.user, content="async-cached")],
        _Output,
        service,
    )
    service.cache.set(key, {"name": "async-from-cache"})

    with patch.object(
        _ConcreteService,
        _ConcreteService._generate_async.__name__,  # noqa: SLF001
    ) as mock_gen:
        result = asyncio.run(
            service.create_structured_output_async(
                "async-cached", _Output, use_cache=True
            )
        )
        mock_gen.assert_not_called()

    assert isinstance(result, _Output)
    assert result.name == "async-from-cache"
    service.cache.data.pop(key, None)


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
    assert isinstance(result, _Output)
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
        service.create_structured_output_async("prompt", str, use_cache=False)
    )
    assert result == "async-hello"
