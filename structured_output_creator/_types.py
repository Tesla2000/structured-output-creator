from __future__ import annotations

import enum


class _ProviderType(str, enum.Enum):
    openai = "openai"
    claude = "claude"
