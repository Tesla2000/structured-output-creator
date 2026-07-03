from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import Anthropic, AsyncAnthropic, omit
from pydantic import BaseModel, ValidationError

from structured_output_creator._claude import _ClaudeService
from structured_output_creator._models import (
    _Message,
    _NoContentError,
    _RefusalError,
    _Role,
)
from structured_output_creator._types import _ProviderType

_MAX_TOKENS = 4096


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


def test_claude_max_tokens_looked_up_for_default_model() -> None:
    haiku_max_tokens = 64_000
    service = _ClaudeService(
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.max_tokens == haiku_max_tokens


def test_claude_max_tokens_looked_up_for_explicit_known_model() -> None:
    opus_4_1_max_tokens = 32_000
    service = _ClaudeService(
        model="claude-opus-4-1",
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.max_tokens == opus_4_1_max_tokens


def test_claude_max_tokens_raises_for_unknown_model() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        _ClaudeService(
            model="claude-definitely-not-a-real-model",
            client=MagicMock(spec=Anthropic),
            async_client=MagicMock(spec=AsyncAnthropic),
        )


def test_claude_explicit_max_tokens_overrides_lookup() -> None:
    explicit_max_tokens = 1234
    service = _ClaudeService(
        model="claude-opus-4-1",
        max_tokens=explicit_max_tokens,
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.max_tokens == explicit_max_tokens


def test_claude_explicit_max_tokens_from_another_models_table_value_is_kept() -> (
    None
):
    haiku_max_tokens = 64_000
    service = _ClaudeService(
        model="claude-opus-4-1",
        max_tokens=haiku_max_tokens,
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.max_tokens == haiku_max_tokens


def test_claude_explicit_max_tokens_allowed_for_unknown_model() -> None:
    explicit_max_tokens = 123
    service = _ClaudeService(
        model="claude-definitely-not-a-real-model",
        max_tokens=explicit_max_tokens,
        client=MagicMock(spec=Anthropic),
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    assert service.max_tokens == explicit_max_tokens


def test_claude_generate_uses_looked_up_max_tokens_for_known_model() -> None:
    sonnet_5_max_tokens = 128_000
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="F")
    )
    service = _ClaudeService(
        model="claude-sonnet-5",
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hi")], _Output
    )
    assert (
        mock_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == sonnet_5_max_tokens
    )
    assert isinstance(result, _Output)


def test_claude_generate_uses_default_models_looked_up_max_tokens() -> None:
    haiku_max_tokens = 64_000
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="G")
    )
    service = _ClaudeService(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    service._generate([_Message(role=_Role.user, content="hi")], _Output)  # noqa: SLF001
    assert (
        mock_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == haiku_max_tokens
    )


def test_claude_generate_uses_explicit_max_tokens_over_lookup() -> None:
    explicit_max_tokens = 4242
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="H")
    )
    service = _ClaudeService(
        model="claude-opus-4-1",
        max_tokens=explicit_max_tokens,
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
    )
    service._generate([_Message(role=_Role.user, content="hi")], _Output)  # noqa: SLF001
    assert (
        mock_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == explicit_max_tokens
    )


def test_claude_generate_async_uses_looked_up_max_tokens() -> None:
    sonnet_5_max_tokens = 128_000
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="I"))
    )
    service = _ClaudeService(
        model="claude-sonnet-5",
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
    )
    asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="hi")], _Output
        )
    )
    assert (
        mock_async_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == sonnet_5_max_tokens
    )


def test_claude_generate_async_uses_explicit_max_tokens_over_lookup() -> None:
    explicit_max_tokens = 4343
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="J"))
    )
    service = _ClaudeService(
        model="claude-opus-4-1",
        max_tokens=explicit_max_tokens,
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
    )
    asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="hi")], _Output
        )
    )
    assert (
        mock_async_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == explicit_max_tokens
    )


def test_claude_generate_calls_parse() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="Alice")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hello")],
        _Output,
    )

    mock_client.beta.messages.parse.assert_called_once_with(
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": "hello"}],
        output_format=_Output,
        temperature=omit,
        top_p=omit,
        top_k=omit,
        system=omit,
        stop_sequences=omit,
    )
    assert isinstance(result, _Output)
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
        max_tokens=_MAX_TOKENS,
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
    custom_max_tokens = 1024
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=custom_max_tokens,
    )
    service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="test")],
        _Output,
    )
    assert (
        mock_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == custom_max_tokens
    )


def test_claude_generate_custom_temperature() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="D")
    )
    custom_temperature = 0.5
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
        temperature=custom_temperature,
    )
    service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="t")],
        _Output,
    )
    assert (
        mock_client.beta.messages.parse.call_args.kwargs["temperature"]
        == custom_temperature
    )


def test_claude_generate_omits_unset_optional_params() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = _parsed_response(
        _Output(name="E")
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )
    service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="t")],
        _Output,
    )
    call_kwargs = mock_client.beta.messages.parse.call_args.kwargs
    assert call_kwargs["temperature"] is omit
    assert call_kwargs["top_p"] is omit
    assert call_kwargs["top_k"] is omit
    assert call_kwargs["system"] is omit
    assert call_kwargs["stop_sequences"] is omit


def test_claude_generate_async_calls_parse() -> None:
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="Async"))
    )
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )
    result = asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="hello")],
            _Output,
        )
    )

    mock_async_client.beta.messages.parse.assert_called_once_with(
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": "hello"}],
        output_format=_Output,
        temperature=omit,
        top_p=omit,
        top_k=omit,
        system=omit,
        stop_sequences=omit,
    )
    assert isinstance(result, _Output)
    assert result.name == "Async"


def test_claude_generate_returns_error_object_on_refusal() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = MagicMock(
        parsed_output=None,
        stop_reason="refusal",
        stop_details=MagicMock(explanation="policy violation"),
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hello")],
        _Output,
    )
    assert isinstance(result, _RefusalError)
    assert result.message == "policy violation"


def test_claude_generate_returns_error_object_on_no_content() -> None:
    mock_client = MagicMock(spec=Anthropic)
    mock_client.beta.messages.parse.return_value = MagicMock(
        parsed_output=None,
        stop_reason="tool_use",
        stop_details=None,
    )
    service = _ClaudeService.model_construct(
        client=mock_client,
        async_client=MagicMock(spec=AsyncAnthropic),
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )
    result = service._generate(  # noqa: SLF001
        [_Message(role=_Role.user, content="hello")],
        _Output,
    )
    assert isinstance(result, _NoContentError)
    assert result.message == "stop_reason=tool_use"


def test_claude_generate_async_returns_error_object_on_refusal() -> None:
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=MagicMock(
            parsed_output=None,
            stop_reason="refusal",
            stop_details=MagicMock(explanation="policy violation"),
        )
    )
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
        model="claude-haiku-4-5",
        max_tokens=_MAX_TOKENS,
    )
    result = asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="hello")],
            _Output,
        )
    )
    assert isinstance(result, _RefusalError)
    assert result.message == "policy violation"


def test_claude_generate_async_custom_max_tokens() -> None:
    mock_async_client = MagicMock(spec=AsyncAnthropic)
    mock_async_client.beta.messages.parse = AsyncMock(
        return_value=_parsed_response(_Output(name="D"))
    )
    custom_max_tokens = 512
    service = _ClaudeService.model_construct(
        client=MagicMock(spec=Anthropic),
        async_client=mock_async_client,
        model="claude-haiku-4-5",
        max_tokens=custom_max_tokens,
    )
    asyncio.run(
        service._generate_async(  # noqa: SLF001
            [_Message(role=_Role.user, content="t")],
            _Output,
        )
    )
    assert (
        mock_async_client.beta.messages.parse.call_args.kwargs["max_tokens"]
        == custom_max_tokens
    )
