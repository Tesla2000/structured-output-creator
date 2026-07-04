from __future__ import annotations

import pytest
from pydantic import ValidationError

from structured_output_creator import Message, Role


def test_role_values() -> None:
    assert Role.user.value == "user"
    assert Role.assistant.value == "assistant"
    assert Role.system.value == "system"


def test_message_creation() -> None:
    msg = Message(role=Role.user, content="hello")
    assert msg.role == Role.user
    assert msg.content == "hello"


def test_message_frozen() -> None:
    msg = Message(role=Role.user, content="hello")
    with pytest.raises(ValidationError):
        msg.role = Role.assistant


def test_message_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        Message(role=Role.user, content="hello", extra_field="bad")
