from __future__ import annotations

import threading
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from structured_output_creator._cache import _ResponseCache
from structured_output_creator._models import _Message, _Role


class _SampleModel(BaseModel):
    name: str


class _FakeService(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    model: str = "default"
    temperature: float | None = None


class _OtherService(_FakeService):
    pass


def _make_messages() -> list[_Message]:
    return [_Message(role=_Role.user, content="hello")]


def test_make_key_deterministic() -> None:
    service = _FakeService(model="gpt-4o")
    messages = _make_messages()
    assert _ResponseCache.make_key(
        messages, _SampleModel, service
    ) == _ResponseCache.make_key(messages, _SampleModel, service)


def test_make_key_differs_on_messages() -> None:
    service = _FakeService()
    msgs_a = [_Message(role=_Role.user, content="hello")]
    msgs_b = [_Message(role=_Role.user, content="world")]
    assert _ResponseCache.make_key(
        msgs_a, _SampleModel, service
    ) != _ResponseCache.make_key(msgs_b, _SampleModel, service)


def test_make_key_differs_on_service_name() -> None:
    messages = _make_messages()
    key_a = _ResponseCache.make_key(messages, _SampleModel, _FakeService())
    key_b = _ResponseCache.make_key(messages, _SampleModel, _OtherService())
    assert key_a != key_b


def test_make_key_differs_on_model() -> None:
    messages = _make_messages()
    key_a = _ResponseCache.make_key(
        messages, _SampleModel, _FakeService(model="gpt-4o")
    )
    key_b = _ResponseCache.make_key(
        messages, _SampleModel, _FakeService(model="gpt-3.5")
    )
    assert key_a != key_b


def test_make_key_differs_on_state() -> None:
    messages = _make_messages()
    key_a = _ResponseCache.make_key(
        messages, _SampleModel, _FakeService(temperature=0.5)
    )
    key_b = _ResponseCache.make_key(
        messages, _SampleModel, _FakeService(temperature=0.9)
    )
    assert key_a != key_b


def test_make_key_stable_without_extra_state() -> None:
    messages = _make_messages()
    service = _FakeService()
    key_a = _ResponseCache.make_key(messages, _SampleModel, service)
    key_b = _ResponseCache.make_key(messages, _SampleModel, service)
    assert key_a == key_b


def test_cache_miss_returns_none() -> None:
    cache = _ResponseCache()
    assert cache.get("nonexistent") is None


def test_cache_set_and_get() -> None:
    cache = _ResponseCache()
    cache.set("key1", {"name": "Alice"})
    assert cache.get("key1") == {"name": "Alice"}


def test_cache_thread_safety() -> None:
    cache = _ResponseCache()
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            cache.set(f"key{i}", {"value": i})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(cache.get(f"key{i}") == {"value": i} for i in range(50))
