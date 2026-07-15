from __future__ import annotations

from contextlib import suppress
from typing import Union

from structured_output_creator._claude._compatible import (
    _ClaudeCompatibleModel as ClaudeCompatibleModel,
)
from structured_output_creator._models import LLMError as LLMError
from structured_output_creator._models import (
    LLMNoContentError as LLMNoContentError,
)
from structured_output_creator._models import (
    LLMRefusalError as LLMRefusalError,
)
from structured_output_creator._models import _ErrorObject as ErrorObject
from structured_output_creator._models import _Message as Message
from structured_output_creator._models import _NoContentError as NoContentError
from structured_output_creator._models import _RefusalError as RefusalError
from structured_output_creator._models import _Role as Role
from structured_output_creator._openai._compatible import (
    _OpenAICompatibleModel as OpenAICompatibleModel,
)
from structured_output_creator._raising_service import (
    _RaisingService as RaisingService,
)
from structured_output_creator._type_compatibility import (
    _IncompatibleFieldTypeError as IncompatibleFieldTypeError,
)
from structured_output_creator._types import _ProviderType as ProviderType

__all__: list[str] = [
    "ClaudeCompatibleModel",
    "ErrorObject",
    "IncompatibleFieldTypeError",
    "LLMError",
    "LLMNoContentError",
    "LLMRefusalError",
    "Message",
    "NoContentError",
    "OpenAICompatibleModel",
    "ProviderType",
    "RaisingService",
    "RefusalError",
    "Role",
]

with suppress(ImportError):
    from structured_output_creator._openai._service import (
        _OpenAIService as OpenAIService,
    )

    __all__ += ["OpenAIService"]

with suppress(ImportError):
    from structured_output_creator._claude._service import (
        _ClaudeService as ClaudeService,
    )

    __all__ += ["ClaudeService"]

with suppress(NameError):
    AnyProviderService = Union[OpenAIService, ClaudeService]
    __all__ += ["AnyProviderService"]
