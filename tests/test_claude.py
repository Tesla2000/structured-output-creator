from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from anthropic import AsyncAnthropic, Anthropic
from pydantic import BaseModel

from structured_output_creator._claude import _ClaudeService
from structured_output_creator._models import _Message, _Role
from structured_output_creator._types import _ProviderType


class _Output(BaseModel):
    name: str


def _parsed_response(obj: _Output) -> MagicMock:
    return MagicMock(parsed_output=obj)


def test_claude_default_model() -> None:
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.model == "claude-haiku-4-5"


def test_claude_service_type_field() -> None:
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.service_type == _ProviderType.claude


def test_claude_default_max_tokens() -> None:
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.max_tokens == 4096


def test_claude_generate_calls_parse() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="Alice")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=4096,
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hello")], _Output
    )

    mock_client.beta.messages.parse.assert_called_once_with(
        model="claude-haiku-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": "hello"}],
        output_format=_Output,
    )
    assert result.name == "Alice"


def test_claude_generate_passes_messages() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="B")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=4096,
    )
    service._generate(  # noqa: SLF001
        [
            _Message(role=_Role.system, content="sys"),
            _Message(role=_Role.user, content="hi"),
        ],
        _Output,
    )
    assert mock_client.beta.messages.parse.call_args.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_claude_generate_custom_max_tokens() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="C")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=1024,
    )
    service._generate([_Message(role=_Role.user, content="test")], _Output)  # noqa: SLF001
    assert mock_client.beta.messages.parse.call_args.kwargs["max_tokens"] == 1024


def test_claude_generate_custom_temperature() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="D")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=4096,
        temperature=0.5,
    )
    service._generate([_Message(role=_Role.user, content="t")], _Output)  # noqa: SLF001
    assert (
        mock_client.beta.messages.parse.call_args.kwargs["temperature"] == 0.5
    )


def test_claude_generate_omits_none_optional_params() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="E")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=4096,
    )
    service._generate([_Message(role=_Role.user, content="t")], _Output)  # noqa: SLF001
    call_kwargs = mock_client.beta.messages.parse.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "top_p" not in call_kwargs
    assert "top_k" not in call_kwargs
    assert "system" not in call_kwargs
    assert "stop_sequences" not in call_kwargs


def test_claude_generate_async_calls_parse() -> None:
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="Async"))
    )
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
        model="claude-haiku-4-5",
        max_tokens=4096,
    )
    result = asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="hello")], _Output
        )
    )

    mock_async_client.beta.messages.parse.assert_called_once_with(
        model="claude-haiku-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": "hello"}],
        output_format=_Output,
    )
    assert result.name == "Async"


def test_claude_generate_async_custom_max_tokens() -> None:
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="D"))
    )
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
        model="claude-haiku-4-5",
        max_tokens=512,
    )
    asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="t")], _Output
        )
    )
    assert (
        mock_async_client.beta.messages.parse.call_args.kwargs["max_tokens"] == 512
    )
