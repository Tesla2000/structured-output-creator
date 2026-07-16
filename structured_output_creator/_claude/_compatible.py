from typing import Any, cast

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

from structured_output_creator._type_compatibility import (
    _JSON_INCOMPATIBLE_ORIGINS,
    _RecursiveTypeCompatibilityChecker,
)


class _ClaudeCompatibleMeta(ModelMetaclass):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "_ClaudeCompatibleMeta":
        cls = super().__new__(
            mcs,
            name,
            bases,
            namespace,
            **kwargs,
        )
        typed_cls = cast("type[BaseModel]", cls)
        typed_cls.model_rebuild(force=True, _parent_namespace_depth=3)
        _RecursiveTypeCompatibilityChecker(
            forbidden_origins=_JSON_INCOMPATIBLE_ORIGINS,
            provider_name="Claude",
        ).check_model(typed_cls)
        return cast("_ClaudeCompatibleMeta", cls)


class _ClaudeCompatibleModel(BaseModel, metaclass=_ClaudeCompatibleMeta):
    pass
