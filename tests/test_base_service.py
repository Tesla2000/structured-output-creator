from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from structured_output_creator._base_service import _BaseService
from structured_output_creator._models import _ErrorObject, _Message, _Role


class _Output(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: str


class _RecordingService(_BaseService[BaseModel]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    def _generate(
        self,
        _messages: list[_Message],
        output_type: type[BaseModel],
    ) -> BaseModel | _ErrorObject:
        return output_type.model_validate({"value": "hello"})

    async def _generate_async(
        self,
        _messages: list[_Message],
        output_type: type[BaseModel],
    ) -> BaseModel | _ErrorObject:
        return output_type.model_validate({"value": "async-hello"})


def _make_service() -> _RecordingService:
    return _RecordingService(model="test-model")


def test_base_service_generates_schema_for_pydantic_output_type() -> None:
    schema = _Output.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"name"}


def test_base_service_wraps_scalar_output_type_into_schema_generating_model() -> (
    None
):
    service = _make_service()
    result = service.create_structured_output(
        [_Message(role=_Role.user, content="hi")], str
    )
    assert result == "hello"


def test_base_service_async_wraps_scalar_output_type_into_schema_generating_model() -> (
    None
):
    service = _make_service()
    result = asyncio.run(
        service.create_structured_output_async(
            [_Message(role=_Role.user, content="hi")], str
        )
    )
    assert result == "async-hello"
