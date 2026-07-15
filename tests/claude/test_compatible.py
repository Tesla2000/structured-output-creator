from typing import Optional, Union

import pytest

from structured_output_creator import ClaudeCompatibleModel, IncompatibleFieldTypeError


def test_allows_compatible_model() -> None:
    class Good(ClaudeCompatibleModel):
        name: str
        count: Optional[int]

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
            value: Union[str, int]
