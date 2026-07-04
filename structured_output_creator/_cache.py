from __future__ import annotations

import hashlib
import json
import threading
from functools import cache
from types import TracebackType
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from structured_output_creator._models import _Message


@runtime_checkable
class _Lock(Protocol):
    def __enter__(self) -> bool: ...

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None: ...


class _ResponseCache(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    data: dict[str, dict[str, object]] = Field(
        default_factory=dict, exclude=True
    )
    lock: InstanceOf[_Lock] = Field(
        default_factory=threading.Lock, exclude=True
    )

    @staticmethod
    def make_key(
        messages: list[_Message],
        output_type: type[BaseModel],
        service: BaseModel,
    ) -> str:
        payload = json.dumps(
            {
                "messages": [m.model_dump_json() for m in messages],
                "schema": output_type.model_json_schema(),
                "service": type(service).__name__,
                "state": service.model_dump_json(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict[str, object] | None:
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: dict[str, object]) -> None:
        with self.lock:
            self.data[key] = value


@cache
def _make_default_cache() -> _ResponseCache:
    return _ResponseCache()
