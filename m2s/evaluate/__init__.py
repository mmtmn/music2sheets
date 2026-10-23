"""Reference scores for evaluation: loading (with repeat expansion) and score-to-audio alignment.

Scoring itself (note F1, hand agreement, playability) runs in the Rust core
(`m2s eval`); the worker provides the aligned reference (`python -m m2s.worker align`)
and the score-level metrics that need Python or Java (MV2H, musicdiff).
"""

from m2s.evaluate.reference import load_reference

__all__ = ["load_reference"]
