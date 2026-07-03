from __future__ import annotations

from typing import ClassVar, Literal, TypeVar

from anthropic import Anthropic, AsyncAnthropic, omit
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, model_validator

from structured_output_creator._base_service import _BaseService
from structured_output_creator._models import (
    _ErrorObject,
    _Message,
    _NoContentError,
    _RefusalError,
)
from structured_output_creator._types import _ProviderType

T = TypeVar("T", bound=BaseModel)

# https://platform.claude.com/docs/en/about-claude/models/overview
_MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "claude-fable-5": 128_000,
    "claude-mythos-5": 128_000,
    "claude-mythos-preview": 128_000,
    "claude-opus-4-8": 128_000,
    "claude-sonnet-5": 128_000,
    "claude-haiku-4-5": 64_000,
    "claude-haiku-4-5-20251001": 64_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-6": 128_000,
    "claude-sonnet-4-6": 128_000,
    "claude-sonnet-4-5": 64_000,
    "claude-sonnet-4-5-20250929": 64_000,
    "claude-opus-4-5": 64_000,
    "claude-opus-4-5-20251101": 64_000,
    "claude-opus-4-1": 32_000,
    "claude-opus-4-1-20250805": 32_000,
}


def _error_from_response(response: BetaMessage) -> _ErrorObject:
    if response.stop_reason == "refusal":
        explanation = (
            response.stop_details.explanation
            if response.stop_details is not None
            else None
        )
        return _RefusalError(message=explanation)
    return _NoContentError(message=f"stop_reason={response.stop_reason}")


class _ClaudeService(_BaseService):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service_type: Literal[_ProviderType.claude] = _ProviderType.claude
    model: str = "claude-haiku-4-5"
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    system: str | None = None
    stop_sequences: list[str] | None = None
    client: InstanceOf[Anthropic] = Field(
        default_factory=Anthropic, exclude=True
    )
    async_client: InstanceOf[AsyncAnthropic] = Field(
        default_factory=AsyncAnthropic, exclude=True
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_default_max_tokens(cls, data: object) -> object:
        if not isinstance(data, dict) or "max_tokens" in data:
            return data
        model_name = data.get("model", cls.model_fields["model"].default)
        if model_name not in _MODEL_MAX_OUTPUT_TOKENS:
            raise ValueError(
                "max_tokens must be provided explicitly for model "
                f"{model_name!r} (not in the known-models table)"
            )
        data["max_tokens"] = _MODEL_MAX_OUTPUT_TOKENS[model_name]
        return data

    def _generate(
        self,
        messages: list[_Message],
        output_type: type[T],
    ) -> T | _ErrorObject:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            output_format=output_type,
            temperature=omit if self.temperature is None else self.temperature,
            top_p=omit if self.top_p is None else self.top_p,
            top_k=omit if self.top_k is None else self.top_k,
            system=omit if self.system is None else self.system,
            stop_sequences=omit
            if self.stop_sequences is None
            else self.stop_sequences,
        )
        if response.parsed_output is not None:
            return response.parsed_output
        return _error_from_response(response)

    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[T],
    ) -> T | _ErrorObject:
        response = await self.async_client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            output_format=output_type,
            temperature=omit if self.temperature is None else self.temperature,
            top_p=omit if self.top_p is None else self.top_p,
            top_k=omit if self.top_k is None else self.top_k,
            system=omit if self.system is None else self.system,
            stop_sequences=omit
            if self.stop_sequences is None
            else self.stop_sequences,
        )
        if response.parsed_output is not None:
            return response.parsed_output
        return _error_from_response(response)
