from __future__ import annotations

from typing import ClassVar, Literal, TypeVar

from anthropic import Anthropic, AsyncAnthropic, omit
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from typing_extensions import TypedDict

from structured_output_creator._base_service import _BaseService
from structured_output_creator._models import _ErrorObject, _Message
from structured_output_creator._types import _ProviderType

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MAX_TOKENS = 4096


def _error_from_response(response: BetaMessage) -> _ErrorObject:
    if response.stop_reason == "refusal":
        explanation = (
            response.stop_details.explanation
            if response.stop_details is not None
            else None
        )
        return _ErrorObject(reason="refusal", message=explanation)
    return _ErrorObject(
        reason="no_content", message=f"stop_reason={response.stop_reason}"
    )


class _ClaudeKwargs(TypedDict, total=False):
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    system: str
    stop_sequences: list[str]


class _ClaudeService(_BaseService[_ClaudeKwargs]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service_type: Literal[_ProviderType.claude] = _ProviderType.claude
    model: str = "claude-haiku-4-5"
    client: InstanceOf[Anthropic] = Field(
        default_factory=Anthropic, exclude=True
    )
    async_client: InstanceOf[AsyncAnthropic] = Field(
        default_factory=AsyncAnthropic, exclude=True
    )

    def _generate(
        self,
        messages: list[_Message],
        output_type: type[T],
        kwargs: _ClaudeKwargs | None = None,
    ) -> T | _ErrorObject:
        resolved = kwargs or {}
        response = self.client.beta.messages.parse(
            model=self.model,
            messages=[
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            output_format=output_type,
            max_tokens=resolved.get("max_tokens", _DEFAULT_MAX_TOKENS),
            temperature=resolved.get("temperature", omit),
            top_p=resolved.get("top_p", omit),
            top_k=resolved.get("top_k", omit),
            system=resolved.get("system", omit),
            stop_sequences=resolved.get("stop_sequences", omit),
        )
        if response.parsed_output is not None:
            return response.parsed_output
        return _error_from_response(response)

    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[T],
        kwargs: _ClaudeKwargs | None = None,
    ) -> T | _ErrorObject:
        resolved = kwargs or {}
        response = await self.async_client.beta.messages.parse(
            model=self.model,
            messages=[
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            output_format=output_type,
            max_tokens=resolved.get("max_tokens", _DEFAULT_MAX_TOKENS),
            temperature=resolved.get("temperature", omit),
            top_p=resolved.get("top_p", omit),
            top_k=resolved.get("top_k", omit),
            system=resolved.get("system", omit),
            stop_sequences=resolved.get("stop_sequences", omit),
        )
        if response.parsed_output is not None:
            return response.parsed_output
        return _error_from_response(response)
