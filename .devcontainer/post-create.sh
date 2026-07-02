#!/usr/bin/env bash
set -e

git config --global --add safe.directory /workspace

if [ ! -f /home/dev/.claude/settings.json ]; then
    cp /workspace/.devcontainer/claude-settings.json /home/dev/.claude/settings.json
fi

uv sync --group dev
echo 'source /workspace/.venv/bin/activate' >> /home/dev/.bashrc
echo 'eval "$(fzf --bash)"' >> /home/dev/.bashrc
uv run pre-commit install --overwrite --hook-type pre-commit --hook-type pre-push
