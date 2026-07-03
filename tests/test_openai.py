from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from structured_output_creator._models import (
    _Message,
    _NoContentError,
    _RefusalError,
    _Role,
)
from structured_output_creator._openai import _OpenAIService
from structured_output_creator._types import _ProviderType


class _Output(BaseModel):
    name: str


def _parsed_response(obj: _Output) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(parsed=obj))])


def test_openai_default_model() -> None:
    service = _OpenAIService.model_construct(
        client=MagicMock(spec=OpenAI),
        async_client=MagicMock(spec=AsyncOpenAI),
    )
    assert service.model == "gpt-5.4-mini"


def test_openai_service_type_field() -> None:
    service = _OpenAIService.model_construct(
        client=MagicMock(spec=OpenAI),
        async_client=MagicMock(spec=AsyncOpenAI),
    )
    assert service.service_type == _ProviderType.openai


def test_openai_generate_calls_parse_with_correct_args() -> None:
    mock_client = MagicMock(spec=OpenAI)
    mock_client.beta.chat.completions.parse.return_value = _parsed_response(
        _Output(name="Alice")
    )
    service = _OpenAIService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncOpenAI),
        model="gpt-5.4-mini",
    )
    messages = [_Message(role=_Role.user, content="hello")]

    result = service._generate(messages, _Output)  # noqa: SLF001

    mock_client.beta.chat.completions.parse.assert_called_once_with(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "hello"}],
        response_format=_Output,
    )
    assert isinstance(result, _Output)
    assert result.name == "Alice"


def test_openai_generate_custom_temperature_via_kwargs() -> None:
    mock_client = MagicMock(spec=OpenAI)
    mock_client.beta.chat.completions.parse.return_value = _parsed_response(
        _Output(name="Bob")
    )
    service = _OpenAIService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncOpenAI),
        model="gpt-5.4-mini",
    )
    custom_temperature = 0.5
    service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="test")],
        _Output,
        kwargs={"temperature": custom_temperature},
    )
    assert (
        mock_client.beta.chat.completions.parse.call_args.kwargs["temperature"]
        == custom_temperature
    )


def test_openai_generate_omits_unset_optional_params() -> None:
    mock_client = MagicMock(spec=OpenAI)
    mock_client.beta.chat.completions.parse.return_value = _parsed_response(
        _Output(name="C")
    )
    service = _OpenAIService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncOpenAI),
        model="gpt-5.4-mini",
    )
    service._generate([_Message(role=_Role.user, content="t")], _Output)  # noqa: SLF001
    call_kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    assert "top_p" not in call_kwargs


def test_openai_generate_returns_error_object_on_refusal() -> None:
    mock_client = MagicMock(spec=OpenAI)
    mock_client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(parsed=None, refusal="cannot help with that")
            )
        ]
    )
    service = _OpenAIService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncOpenAI),
        model="gpt-5.4-mini",
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hello")], _Output
    )
    assert isinstance(result, _RefusalError)
    assert result.message == "cannot help with that"


def test_openai_generate_returns_error_object_on_no_content() -> None:
    mock_client = MagicMock(spec=OpenAI)
    mock_client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=None, refusal=None))]
    )
    service = _OpenAIService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncOpenAI),
        model="gpt-5.4-mini",
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hello")], _Output
    )
    assert isinstance(result, _NoContentError)
    assert result.message is None


def test_openai_generate_async_returns_error_object_on_refusal() -> None:
    mock_async_client = MagicMock(spec=AsyncOpenAI)
    mock_async_client.beta.chat.completions.parse = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        parsed=None, refusal="cannot help with that"
                    )
                )
            ]
        )
    )
    service = _OpenAIService.model_construct(
        client=MagicMock(spec=OpenAI),
        async_client=mock_async_client,
        model="gpt-5.4-mini",
    )
    result = asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="hello")], _Output
        )
    )
    assert isinstance(result, _RefusalError)
    assert result.message == "cannot help with that"


def test_openai_generate_async_calls_parse() -> None:
    mock_async_client = MagicMock(spec=AsyncOpenAI)
    mock_async_client.beta.chat.completions.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="Async"))
    )
    service = _OpenAIService.model_construct(
        client=MagicMock(spec=OpenAI),
        async_client=mock_async_client,
        model="gpt-5.4-mini",
    )
    messages = [_Message(role=_Role.user, content="hello")]

    result = asyncio.run(service._generate_async(messages, _Output))  # noqa: SLF001

    mock_async_client.beta.chat.completions.parse.assert_called_once_with(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "hello"}],
        response_format=_Output,
    )
    assert isinstance(result, _Output)
    assert result.name == "Async"


def test_openai_generate_async_custom_temperature_via_kwargs() -> None:
    mock_async_client = MagicMock(spec=AsyncOpenAI)
    mock_async_client.beta.chat.completions.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="C"))
    )
    service = _OpenAIService.model_construct(
        client=MagicMock(spec=OpenAI),
        async_client=mock_async_client,
        model="gpt-5.4-mini",
    )
    custom_temperature = 0.7
    asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="t")],
            _Output,
            kwargs={"temperature": custom_temperature},
        )
    )
    assert (
        mock_async_client.beta.chat.completions.parse.call_args.kwargs[
            "temperature"
        ]
        == custom_temperature
    )
