from __future__ import annotations

from pathlib import Path

import mypy.api
import pytest

_INCOMPATIBLE_MODEL_SNIPPET = """\
from pydantic import BaseModel

from structured_output_creator import {service}


class Incompatible(BaseModel):
    tags: dict[str, int]


def use(service: {service}) -> None:
    service.create_structured_output("prompt", Incompatible)
"""

_COMPATIBLE_MODEL_SNIPPET = """\
from structured_output_creator import {compatible_model}, {service}


class Good({compatible_model}):
    name: str


def use(service: {service}) -> None:
    service.create_structured_output("prompt", Good)
"""

_SCALAR_SNIPPET = """\
from structured_output_creator import {service}


def use(service: {service}) -> None:
    service.create_structured_output("prompt", str)
"""


def _run_mypy(tmp_path: Path, source: str) -> str:
    module = tmp_path / "snippet.py"
    module.write_text(source)
    stdout, _stderr, _exit_status = mypy.api.run(["--strict", str(module)])
    return stdout


@pytest.mark.parametrize("service", ["OpenAIService", "ClaudeService"])
def test_incompatible_model_is_rejected_at_type_check_time(
    tmp_path: Path, service: str
) -> None:
    stdout = _run_mypy(
        tmp_path, _INCOMPATIBLE_MODEL_SNIPPET.format(service=service)
    )
    assert "error" in stdout, stdout


@pytest.mark.parametrize(
    ("service", "compatible_model"),
    [
        ("OpenAIService", "OpenAICompatibleModel"),
        ("ClaudeService", "ClaudeCompatibleModel"),
    ],
)
def test_compatible_model_type_checks_cleanly(
    tmp_path: Path, service: str, compatible_model: str
) -> None:
    stdout = _run_mypy(
        tmp_path,
        _COMPATIBLE_MODEL_SNIPPET.format(
            service=service, compatible_model=compatible_model
        ),
    )
    assert "Success" in stdout, stdout


@pytest.mark.parametrize("service", ["OpenAIService", "ClaudeService"])
def test_scalar_output_type_type_checks_cleanly(
    tmp_path: Path, service: str
) -> None:
    stdout = _run_mypy(tmp_path, _SCALAR_SNIPPET.format(service=service))
    assert "Success" in stdout, stdout
