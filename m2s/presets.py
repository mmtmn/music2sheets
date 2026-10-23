"""Instrument presets (shared YAML in presets/ at the repo root; the Rust core embeds the same files)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


@lru_cache(maxsize=None)
def load_preset(name: str) -> dict[str, Any]:
    path = PRESET_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(available())}")
    return yaml.safe_load(path.read_text())


def available() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))
