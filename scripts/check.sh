#!/usr/bin/env bash
# Every check in one place: formatting, lints and tests for the Rust core and the Python worker.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== cargo fmt";    cargo fmt --all --check
echo "== cargo clippy"; cargo clippy --all-targets -q -- -D warnings
echo "== cargo test";   cargo test -q 2>&1 | grep -E "^test result|FAILED|panicked" || true
cargo test -q >/dev/null
PY=${M2S_PYTHON:-.venv/bin/python}
echo "== pytest";       "$PY" -m pytest -q
if command -v uvx >/dev/null; then echo "== ruff"; uvx ruff check m2s tools tests --select E,F,W --ignore E501; fi
echo "all checks passed"
