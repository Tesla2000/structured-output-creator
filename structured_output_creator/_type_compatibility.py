import sys
import types
import typing
from collections import abc as collections_abc
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

_PEP604_UNION_ORIGINS: frozenset[object] = (
    frozenset({types.UnionType})
    if sys.version_info >= (3, 10)
    else frozenset()
)

_UNION_ORIGINS: frozenset[object] = (
    frozenset({typing.Union}) | _PEP604_UNION_ORIGINS
)

_TUPLE_ORIGINS: frozenset[object] = frozenset({tuple})

_JSON_INCOMPATIBLE_ORIGINS: frozenset[object] = frozenset(
    {
        dict,
        collections_abc.Mapping,
        collections_abc.MutableMapping,
        set,
        frozenset,
    }
)

_OPTIONAL_UNION_ARG_COUNT = 2
_HOMOGENEOUS_TUPLE_ARG_COUNT = 2


class _IncompatibleFieldTypeError(TypeError):
    def __init__(
        self, field_path: str, annotation: object, provider_name: str
    ) -> None:
        super().__init__(
            f"{field_path}: {annotation!r} is not supported by {provider_name}"
            " structured outputs"
        )


class _RecursiveTypeCompatibilityChecker(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    forbidden_origins: frozenset[object]
    provider_name: str
    allow_unions: bool = False

    def check_model(self, model: type[BaseModel]) -> None:
        for name, field in model.model_fields.items():
            self._check_annotation(
                field.annotation, f"{model.__name__}.{name}"
            )

    def _check_annotation(self, annotation: object, field_path: str) -> None:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin in _UNION_ORIGINS:
            if self.allow_unions or self._is_optional(args):
                for arg in args:
                    if arg is not type(None):
                        self._check_annotation(arg, field_path)
                return
            raise _IncompatibleFieldTypeError(
                field_path, annotation, self.provider_name
            )
        if origin in _TUPLE_ORIGINS:
            if self._is_homogeneous_tuple(args):
                self._check_annotation(args[0], field_path)
                return
            raise _IncompatibleFieldTypeError(
                field_path, annotation, self.provider_name
            )
        if (
            origin in self.forbidden_origins
            or annotation in self.forbidden_origins
        ):
            raise _IncompatibleFieldTypeError(
                field_path, annotation, self.provider_name
            )
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            self.check_model(annotation)
            return
        for arg in args:
            self._check_annotation(arg, field_path)

    @staticmethod
    def _is_optional(args: tuple[object, ...]) -> bool:
        return type(None) in args and len(args) == _OPTIONAL_UNION_ARG_COUNT

    @staticmethod
    def _is_homogeneous_tuple(args: tuple[object, ...]) -> bool:
        return (
            len(args) == _HOMOGENEOUS_TUPLE_ARG_COUNT and args[1] is Ellipsis
        )
