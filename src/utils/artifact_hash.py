from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    """Return a stable sha256 for generated patch artifacts."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    return sha256_text(p.read_text(encoding="utf-8"))
