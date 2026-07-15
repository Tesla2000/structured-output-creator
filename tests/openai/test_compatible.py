from typing import Optional, Union

import pytest

from structured_output_creator import IncompatibleFieldTypeError, OpenAICompatibleModel


def test_allows_compatible_model() -> None:
    class Good(OpenAICompatibleModel):
        name: str
        count: Optional[int]

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
        value: Union[str, int]

    Good(value="a")
