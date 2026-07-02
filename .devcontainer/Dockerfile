FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        bubblewrap \
        socat \
        nodejs \
        npm \
        openssh-client \
        locales-all \
        micro \
        fzf \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN npm install -g @anthropic-ai/claude-code

RUN groupadd --gid 1000 dev && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home dev

ENV UV_LINK_MODE=copy
ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PRE_COMMIT_HOME="/.jbdevcontainer/pre-commit"
ENV EDITOR=micro

RUN mkdir -p /.jbdevcontainer/pre-commit && chown -R dev:dev /.jbdevcontainer

WORKDIR /workspace
RUN chown dev:dev /workspace

USER dev

RUN mkdir -p /home/dev/.claude

COPY --chown=dev:dev pyproject.toml uv.lock .pre-commit-config.yaml README.md ./
RUN uv sync --group dev --no-install-project
RUN git config --global user.email "build@example.com" && \
    git config --global user.name "Build" && \
    git init && git add -A && git commit -m init && \
    pre-commit run --all-files; rm -rf .git
