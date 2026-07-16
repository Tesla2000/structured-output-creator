from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_output_creator import (
    ClaudeCompatibleModel,
    IncompatibleFieldTypeError,
)


def test_allows_compatible_model() -> None:
    class Good(ClaudeCompatibleModel):
        name: str
        count: int | None

    Good(name="a", count=None)


def test_rejects_dict_field_at_class_definition() -> None:
    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(ClaudeCompatibleModel):
            data: dict[str, int]


def test_rejects_set_field_at_class_definition() -> None:
    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(ClaudeCompatibleModel):
            tags: set[str]


def test_rejects_non_optional_union_at_class_definition() -> None:
    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(ClaudeCompatibleModel):
            value: str | int


def test_allows_compatible_nested_model_field() -> None:
    class Inner(BaseModel):
        label: str

    class Good(ClaudeCompatibleModel):
        inner: Inner

    Good(inner=Inner(label="a"))


def test_rejects_incompatible_field_in_nested_model() -> None:
    class Inner(BaseModel):
        data: dict[str, int]

    with pytest.raises(IncompatibleFieldTypeError):

        class Bad(ClaudeCompatibleModel):
            inner: Inner


def test_generates_json_schema() -> None:
    class Good(ClaudeCompatibleModel):
        name: str
        count: int | None

    schema = Good.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"name", "count"}
