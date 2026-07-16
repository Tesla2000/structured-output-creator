from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_output_creator import IncompatibleFieldTypeError
from structured_output_creator._type_compatibility import (
    _JSON_INCOMPATIBLE_ORIGINS,
    _RecursiveTypeCompatibilityChecker,
)


def _make_checker() -> _RecursiveTypeCompatibilityChecker:
    return _RecursiveTypeCompatibilityChecker(
        forbidden_origins=_JSON_INCOMPATIBLE_ORIGINS,
        provider_name="TestProvider",
    )


def test_allows_plain_model() -> None:
    class Plain(BaseModel):
        name: str
        count: int

    _make_checker().check_model(Plain)


def test_allows_optional_field() -> None:
    class WithOptional(BaseModel):
        name: str | None
        other: int | None

    _make_checker().check_model(WithOptional)


def test_rejects_dict_field() -> None:
    class WithDict(BaseModel):
        data: dict[str, int]

    with pytest.raises(IncompatibleFieldTypeError):
        _make_checker().check_model(WithDict)


def test_rejects_set_field() -> None:
    class WithSet(BaseModel):
        tags: set[str]

    with pytest.raises(IncompatibleFieldTypeError):
        _make_checker().check_model(WithSet)


def test_rejects_tuple_field() -> None:
    class WithTuple(BaseModel):
        pair: tuple[str, int]

    with pytest.raises(IncompatibleFieldTypeError):
        _make_checker().check_model(WithTuple)


def test_allows_homogeneous_variadic_tuple() -> None:
    class WithHomogeneousTuple(BaseModel):
        items: tuple[str, ...]

    _make_checker().check_model(WithHomogeneousTuple)


def test_rejects_non_optional_union() -> None:
    class WithUnion(BaseModel):
        value: str | int

    with pytest.raises(IncompatibleFieldTypeError):
        _make_checker().check_model(WithUnion)


def test_allow_unions_permits_non_optional_union() -> None:
    class WithUnion(BaseModel):
        value: str | int

    checker = _RecursiveTypeCompatibilityChecker(
        forbidden_origins=_JSON_INCOMPATIBLE_ORIGINS,
        provider_name="TestProvider",
        allow_unions=True,
    )
    checker.check_model(WithUnion)


def test_allow_unions_still_rejects_forbidden_members() -> None:
    class WithBadUnion(BaseModel):
        value: str | dict[str, int]

    checker = _RecursiveTypeCompatibilityChecker(
        forbidden_origins=_JSON_INCOMPATIBLE_ORIGINS,
        provider_name="TestProvider",
        allow_unions=True,
    )
    with pytest.raises(IncompatibleFieldTypeError):
        checker.check_model(WithBadUnion)


def test_recurses_into_nested_model() -> None:
    class BadNested(BaseModel):
        data: dict[str, int]

    class Outer(BaseModel):
        nested: BadNested

    with pytest.raises(IncompatibleFieldTypeError):
        _make_checker().check_model(Outer)


def test_recurses_into_generic_containers() -> None:
    class BadNested(BaseModel):
        data: dict[str, int]

    class Outer(BaseModel):
        items: list[BadNested]

    with pytest.raises(IncompatibleFieldTypeError):
        _make_checker().check_model(Outer)
