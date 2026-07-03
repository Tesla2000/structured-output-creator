from __future__ import annotations

from typing import ClassVar, Literal, TypeVar, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ParsedChatCompletionMessage,
)
from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from typing_extensions import TypedDict

from structured_output_creator._base_service import _NO_KWARGS, _BaseService
from structured_output_creator._models import (
    _ErrorObject,
    _Message,
    _NoContentError,
    _RefusalError,
    _Role,
)
from structured_output_creator._types import _ProviderType

T = TypeVar("T", bound=BaseModel)
_ContentT = TypeVar("_ContentT")


def _error_from_message(
    message: ParsedChatCompletionMessage[_ContentT],
) -> _ErrorObject:
    if message.refusal:
        return _RefusalError(message=message.refusal)
    return _NoContentError()


class _OpenAIKwargs(TypedDict, total=False):
    temperature: float
    max_tokens: int
    top_p: float
    n: int
    frequency_penalty: float
    presence_penalty: float
    seed: int


_NO_OPENAI_KWARGS = cast("_OpenAIKwargs", _NO_KWARGS)


def _as_oai_param(msg: _Message) -> ChatCompletionMessageParam:
    if msg.role == _Role.user:
        return ChatCompletionUserMessageParam(role="user", content=msg.content)
    if msg.role == _Role.assistant:
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=msg.content
        )
    return ChatCompletionSystemMessageParam(role="system", content=msg.content)


class _OpenAIService(_BaseService[_OpenAIKwargs]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service_type: Literal[_ProviderType.openai] = _ProviderType.openai
    model: str = "gpt-5.4-mini"
    client: InstanceOf[OpenAI] = Field(default_factory=OpenAI, exclude=True)
    async_client: InstanceOf[AsyncOpenAI] = Field(
        default_factory=AsyncOpenAI, exclude=True
    )

    def _generate(
        self,
        messages: list[_Message],
        output_type: type[T],
        kwargs: _OpenAIKwargs = _NO_OPENAI_KWARGS,
    ) -> T | _ErrorObject:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[_as_oai_param(m) for m in messages],
            response_format=output_type,
            **kwargs,
        )
        message = completion.choices[0].message
        if message.parsed is not None:
            return message.parsed
        return _error_from_message(message)

    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[T],
        kwargs: _OpenAIKwargs = _NO_OPENAI_KWARGS,
    ) -> T | _ErrorObject:
        completion = await self.async_client.beta.chat.completions.parse(
            model=self.model,
            messages=[_as_oai_param(m) for m in messages],
            response_format=output_type,
            **kwargs,
        )
        message = completion.choices[0].message
        if message.parsed is not None:
            return message.parsed
        return _error_from_message(message)
