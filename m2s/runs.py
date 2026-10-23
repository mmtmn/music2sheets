"""Run directories: one folder per input, keyed by the audio's content hash.

Each stage records the hash of its inputs and config in ``manifest.json`` so a
rerun skips every stage whose inputs haven't changed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

RUNS_ROOT = Path("runs")


def file_hash(path: str | Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def obj_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


class Run:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {"stages": {}}
        )

    @staticmethod
    def for_audio(audio_path: str | Path, runs_root: str | Path = RUNS_ROOT) -> "Run":
        return Run(Path(runs_root) / file_hash(audio_path, 12))

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- stage bookkeeping -------------------------------------------------
    def is_fresh(self, stage: str, key: Any, outputs: list[str]) -> bool:
        rec = self.manifest["stages"].get(stage)
        if not rec or rec.get("key") != obj_hash(key):
            return False
        return all((self.root / o).exists() for o in outputs)

    def mark(self, stage: str, key: Any, **info: Any) -> None:
        self.manifest["stages"][stage] = {"key": obj_hash(key), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        self.save_manifest()

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1))

    def stage_key(self, stage: str) -> str | None:
        rec = self.manifest["stages"].get(stage)
        return rec.get("key") if rec else None

    # -- json helpers ------------------------------------------------------
    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=1, default=_json_default))
        return p

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text())


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
