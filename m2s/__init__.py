"""music2sheets: audio to playable sheet music.

Pipeline: ingest -> analyze -> separate -> transcribe -> decode (playability)
-> verify -> notate -> engrave, with every stage reading and writing the
shared note format in :mod:`m2s.ir`.
"""

__version__ = "0.1.0"
