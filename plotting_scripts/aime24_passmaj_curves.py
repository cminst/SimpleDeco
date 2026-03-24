"""Compatibility wrapper for the generic pass@k / maj@k curve generator."""
from __future__ import annotations

from pathlib import Path

from passmaj_curves import main as _main


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figure" / "aime24_passmaj_curves.pdf"


if __name__ == "__main__":
    _main(default_datasets=["aime24"], default_output=DEFAULT_OUTPUT)
