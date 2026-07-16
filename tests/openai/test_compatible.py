from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_output_creator import (
    IncompatibleFieldTypeError,
    OpenAICompatibleModel,
)


def test_allows_compatible_model() -> None:
    class Good(OpenAICompatibleModel):
        name: str
        count: int | None

    Good(name="a", count=None)


def test_rejects_dict_field_at_class_definition() -> None:
    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(OpenAICompatibleModel):
            data: dict[str, int]


def test_rejects_set_field_at_class_definition() -> None:
    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(OpenAICompatibleModel):
            tags: set[str]


def test_allows_non_optional_union() -> None:
    class Good(OpenAICompatibleModel):
        value: str | int

    Good(value="a")


def test_allows_compatible_nested_model_field() -> None:
    class Inner(BaseModel):
        label: str

    class Good(OpenAICompatibleModel):
        inner: Inner

    Good(inner=Inner(label="a"))


def test_rejects_incompatible_field_in_nested_model() -> None:
    class Inner(BaseModel):
        data: dict[str, int]

    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(OpenAICompatibleModel):
            inner: Inner


def test_generates_json_schema() -> None:
    class Good(OpenAICompatibleModel):
        name: str
        count: int | None
        value: str | int

    schema = Good.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"name", "count", "value"}
    assert "anyOf" in schema["properties"]["value"]
