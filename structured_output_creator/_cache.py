from __future__ import annotations

import hashlib
import json
import threading
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from structured_output_creator._models import _Message

_LockType = type(threading.Lock())


class _ResponseCache(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    data: InstanceOf[dict] = Field(default_factory=dict, exclude=True)  # type: ignore[type-arg]
    lock: InstanceOf[_LockType] = Field(  # type: ignore[valid-type]
        default_factory=threading.Lock, exclude=True
    )

    @staticmethod
    def make_key(
        messages: list[_Message],
        output_type: type[BaseModel],
        service_name: str,
        model: str,
        **kwargs: object,
    ) -> str:
        payload = json.dumps(
            {
                "messages": [m.model_dump() for m in messages],
                "schema": output_type.model_json_schema(),
                "service": service_name,
                "model": model,
                "kwargs": kwargs,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict[str, object] | None:
        with self.lock:  # type: ignore[attr-defined]
            return self.data.get(key)

    def set(self, key: str, value: dict[str, object]) -> None:
        with self.lock:  # type: ignore[attr-defined]
            self.data[key] = value


_default_cache = _ResponseCache()
