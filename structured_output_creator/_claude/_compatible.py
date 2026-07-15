from typing import Any

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
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if bases:
            _RecursiveTypeCompatibilityChecker(
                forbidden_origins=_JSON_INCOMPATIBLE_ORIGINS,
                provider_name="Claude",
            ).check_model(cls)
        return cls


class _ClaudeCompatibleModel(BaseModel, metaclass=_ClaudeCompatibleMeta):
    pass
