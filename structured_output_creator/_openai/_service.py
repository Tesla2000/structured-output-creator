from __future__ import annotations

from typing import ClassVar, Literal, TypeVar

from openai import AsyncOpenAI, OpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ParsedChatCompletionMessage,
)
from pydantic import ConfigDict, Field, InstanceOf
from pydantic.json_schema import SkipJsonSchema

from structured_output_creator._base_service import _BaseService
from structured_output_creator._models import (
    _ErrorObject,
    _Message,
    _NoContentError,
    _RefusalError,
    _Role,
)
from structured_output_creator._openai._compatible import (
    _OpenAICompatibleModel,
)
from structured_output_creator._types import _ProviderType

_ContentT = TypeVar("_ContentT")


def _error_from_message(
    message: ParsedChatCompletionMessage[_ContentT],
) -> _ErrorObject:
    if message.refusal:
        return _RefusalError(message=message.refusal)
    return _NoContentError()


def _as_oai_param(msg: _Message) -> ChatCompletionMessageParam:
    if msg.role == _Role.user:
        return ChatCompletionUserMessageParam(role="user", content=msg.content)
    if msg.role == _Role.assistant:
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=msg.content
        )
    return ChatCompletionSystemMessageParam(role="system", content=msg.content)


class _OpenAIService(_BaseService[_OpenAICompatibleModel]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service_type: Literal[_ProviderType.openai] = _ProviderType.openai
    model: str = "gpt-5.4-mini"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    n: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    client: SkipJsonSchema[InstanceOf[OpenAI]] = Field(
        default_factory=OpenAI, exclude=True
    )
    async_client: SkipJsonSchema[InstanceOf[AsyncOpenAI]] = Field(
        default_factory=AsyncOpenAI, exclude=True
    )

    def _generate(
        self,
        messages: list[_Message],
        output_type: type[_OpenAICompatibleModel],
    ) -> _OpenAICompatibleModel | _ErrorObject:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[_as_oai_param(m) for m in messages],
            response_format=output_type,
            temperature=omit if self.temperature is None else self.temperature,
            max_tokens=omit if self.max_tokens is None else self.max_tokens,
            top_p=omit if self.top_p is None else self.top_p,
            n=omit if self.n is None else self.n,
            frequency_penalty=omit
            if self.frequency_penalty is None
            else self.frequency_penalty,
            presence_penalty=omit
            if self.presence_penalty is None
            else self.presence_penalty,
            seed=omit if self.seed is None else self.seed,
        )
        message = completion.choices[0].message
        if message.parsed is not None:
            return message.parsed
        return _error_from_message(message)

    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[_OpenAICompatibleModel],
    ) -> _OpenAICompatibleModel | _ErrorObject:
        completion = await self.async_client.beta.chat.completions.parse(
            model=self.model,
            messages=[_as_oai_param(m) for m in messages],
            response_format=output_type,
            temperature=omit if self.temperature is None else self.temperature,
            max_tokens=omit if self.max_tokens is None else self.max_tokens,
            top_p=omit if self.top_p is None else self.top_p,
            n=omit if self.n is None else self.n,
            frequency_penalty=omit
            if self.frequency_penalty is None
            else self.frequency_penalty,
            presence_penalty=omit
            if self.presence_penalty is None
            else self.presence_penalty,
            seed=omit if self.seed is None else self.seed,
        )
        message = completion.choices[0].message
        if message.parsed is not None:
            return message.parsed
        return _error_from_message(message)
