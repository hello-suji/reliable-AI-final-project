from __future__ import annotations

from pathlib import Path


def read_text(path: Path) -> str:
    """Read a text file with UTF-8, falling back to Latin-1 on decode errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
