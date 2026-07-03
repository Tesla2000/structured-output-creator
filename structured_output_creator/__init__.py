from __future__ import annotations

from typing import Union

from structured_output_creator._models import _ErrorObject as ErrorObject
from structured_output_creator._models import _Message as Message
from structured_output_creator._models import _NoContentError as NoContentError
from structured_output_creator._models import _RefusalError as RefusalError
from structured_output_creator._models import _Role as Role
from structured_output_creator._types import _ProviderType as ProviderType

__all__: list[str] = [
    "ErrorObject",
    "Message",
    "NoContentError",
    "ProviderType",
    "RefusalError",
    "Role",
]

try:
    from structured_output_creator._openai import (
        _OpenAIService as OpenAIService,
    )

    __all__ += ["OpenAIService"]
except ImportError:
    pass

try:
    from structured_output_creator._claude import (
        _ClaudeService as ClaudeService,
    )

    __all__ += ["ClaudeService"]
except ImportError:
    pass

try:
    AnyProviderService = Union[OpenAIService, ClaudeService]
    __all__ += ["AnyProviderService"]
except NameError:
    pass
