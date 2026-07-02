from __future__ import annotations

from typing import ClassVar, Literal, TypeVar

from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from structured_output_creator._base_service import _BaseService
from structured_output_creator._models import _Message
from structured_output_creator._types import _ProviderType

T = TypeVar("T", bound=BaseModel)


class _ClaudeService(_BaseService):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    service_type: Literal[_ProviderType.claude] = _ProviderType.claude
    model: str = "claude-haiku-4-5"
    max_tokens: int = 4096
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

    def _generate(
        self,
        messages: list[_Message],
        output_type: type[T],
    ) -> T:
        optional: dict[str, object] = {
            k: v
            for k, v in {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "system": self.system,
                "stop_sequences": self.stop_sequences,
            }.items()
            if v is not None
        }
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[  # type: ignore[arg-type]
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            output_format=output_type,
            **optional,  # type: ignore[arg-type]
        )
        return response.parsed_output  # type: ignore[return-value]

    async def _generate_async(
        self,
        messages: list[_Message],
        output_type: type[T],
    ) -> T:
        optional: dict[str, object] = {
            k: v
            for k, v in {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "system": self.system,
                "stop_sequences": self.stop_sequences,
            }.items()
            if v is not None
        }
        response = await self.async_client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[  # type: ignore[arg-type]
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            output_format=output_type,
            **optional,  # type: ignore[arg-type]
        )
        return response.parsed_output  # type: ignore[return-value]
