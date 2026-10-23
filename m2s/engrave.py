"""Engraving: MusicXML -> SVG pages (Verovio) -> PDF, or MuseScore when installed."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from pathlib import Path

MUSESCORE = ("mscore4portable", "mscore4", "musescore4", "mscore", "musescore")


def musescore() -> str | None:
    for exe in MUSESCORE:
        if shutil.which(exe):
            return exe
    return None


def verovio_svgs(musicxml: str | Path, options: dict | None = None) -> list[str]:
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": False,
                   "footer": "none", "breaks": "auto", **(options or {})})
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError(f"verovio could not read {musicxml}")
    return [tk.renderToSVG(i + 1) for i in range(tk.getPageCount())]


# SMuFL text glyphs (metronome marks) that cairo cannot draw from Verovio's web font,
# mapped to plain Unicode note symbols every system font set carries
SMUFL_TEXT = {"\uECA5": "\u2669", "\uECA7": "\u266A", "\uECA3": "\U0001D15E", "\uECA2": "\U0001D15D",
              "\uECB7": ".", "\uECA9": "\u266B"}
_SMUFL_SPAN = re.compile(r'<tspan font-family="(?:Leipzig|Bravura|Gootville)"([^>]*)>([^<]*)</tspan>')


def _plain_text_glyphs(svg: str) -> str:
    def sub(m: re.Match) -> str:
        txt = "".join(SMUFL_TEXT.get(ch, ch) for ch in m.group(2))
        attrs = re.sub(r'font-size="(\d+)px"', lambda k: f'font-size="{int(int(k.group(1)) * 0.6)}px"', m.group(1))
        return f'<tspan font-family="DejaVu Sans"{attrs}>{txt}</tspan>'

    return _SMUFL_SPAN.sub(sub, svg)


def svgs_to_pdf(svgs: list[str], out: str | Path) -> None:
    import cairosvg
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for svg in svgs:
        buf = io.BytesIO()
        cairosvg.svg2pdf(bytestring=_plain_text_glyphs(svg).encode(), write_to=buf)
        buf.seek(0)
        for page in PdfReader(buf).pages:
            writer.add_page(page)
    with open(out, "wb") as f:
        writer.write(f)


def engrave(musicxml: str | Path, pdf: str | Path, engine: str = "auto") -> str:
    """Render ``musicxml`` to ``pdf``. Returns the engine used."""
    exe = musescore() if engine in ("auto", "musescore") else None
    if exe:
        r = subprocess.run([exe, "-o", str(pdf), str(musicxml)], capture_output=True, text=True,
                           env={"QT_QPA_PLATFORM": "offscreen", **_env()})
        if r.returncode == 0 and Path(pdf).exists():
            return "musescore"
        if engine == "musescore":
            raise RuntimeError(r.stderr[-2000:])
    svgs_to_pdf(verovio_svgs(musicxml), pdf)
    return "verovio"


def render_bars_png(musicxml: str | Path, bars: tuple[int, int], out: str | Path, width: int = 2400) -> None:
    """Engrave only bars a..b (for `m2s inspect`)."""
    import cairosvg
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": width, "adjustPageHeight": True, "scale": 45, "footer": "none", "header": "none",
                   "breaks": "auto"})
    tk.loadFile(str(musicxml))
    tk.select({"measureRange": f"{bars[0]}-{bars[1]}"})
    tk.redoLayout()
    svg = _plain_text_glyphs(tk.renderToSVG(1))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out), output_width=width // 2, background_color="white")


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)
