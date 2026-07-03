from __future__ import annotations

import enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class _Role(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class _Message(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    role: _Role
    content: str


class _ErrorObject(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid"
    )

    message: str | None = None


class _RefusalError(_ErrorObject):
    pass


class _NoContentError(_ErrorObject):
    pass
