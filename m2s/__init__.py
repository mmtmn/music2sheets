"""music2sheets model worker.

The pipeline is the Rust `m2s` binary (crates/). This package is the part that
needs Python: the neural models (analysis, separation, transcription and the
verification front ends), engraving, inspection images, and reference loading
and alignment for evaluation. The Rust side calls it as `python -m m2s.worker`.
"""

__version__ = "0.1.3"
