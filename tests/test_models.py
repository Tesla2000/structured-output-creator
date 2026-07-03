from __future__ import annotations

import pytest
from pydantic import ValidationError

from structured_output_creator._models import _Message, _Role


def test_role_values() -> None:
    assert _Role.user.value == "user"
    assert _Role.assistant.value == "assistant"
    assert _Role.system.value == "system"


def test_message_creation() -> None:
    msg = _Message(role=_Role.user, content="hello")
    assert msg.role == _Role.user
    assert msg.content == "hello"


def test_message_frozen() -> None:
    msg = _Message(role=_Role.user, content="hello")
    with pytest.raises(ValidationError):
        msg.role = _Role.assistant


def test_message_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        _Message(role=_Role.user, content="hello", extra_field="bad")
