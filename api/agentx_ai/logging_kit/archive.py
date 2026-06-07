"""Phase 6 — compressed, rotating on-disk log archive.

The ring buffer is in-memory and bounded; this adds durable retention so history
survives restarts and can be browsed/downloaded from the Log panel. A
``RotatingFileHandler`` (owned by the QueueListener) writes the **plain, redacted,
ANSI-free** stream to ``data/logs/agentx.log`` and gzips rotated segments.
"""

from __future__ import annotations

import gzip
import logging
import logging.handlers
import os
import shutil
from pathlib import Path
from typing import Optional

from .flags import LogFlags

ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "logs"
ARCHIVE_BASENAME = "agentx.log"
_ARCHIVE_FMT = "{levelname} {asctime} {module} {message}"


def archive_path() -> Path:
    return ARCHIVE_DIR / ARCHIVE_BASENAME


class GzipRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that gzips each rotated segment (``agentx.log.1.gz``)."""

    def rotation_filename(self, default_name: str) -> str:
        return default_name + ".gz"

    def rotate(self, source: str, dest: str) -> None:
        with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(source)


def build_archive_handler(flags: LogFlags) -> Optional[logging.Handler]:
    if not flags.archive_enabled:
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    handler = GzipRotatingFileHandler(
        filename=str(archive_path()),
        maxBytes=max(1, flags.archive_max_mb) * 1024 * 1024,
        backupCount=max(0, flags.archive_backups),
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter(_ARCHIVE_FMT, style="{"))
    return handler


def list_segments() -> list[dict[str, object]]:
    """List the archive segments (newest first) for ``GET /api/logs/archive``."""
    if not ARCHIVE_DIR.exists():
        return []
    out: list[dict[str, object]] = []
    for p in sorted(ARCHIVE_DIR.glob(ARCHIVE_BASENAME + "*")):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            {
                "name": p.name,
                "size": st.st_size,
                "modified": st.st_mtime,
                "compressed": p.suffix == ".gz",
            }
        )
    out.sort(key=lambda e: e["modified"], reverse=True)  # type: ignore[arg-type,return-value]
    return out


def resolve_segment(name: str) -> Optional[Path]:
    """Safely resolve a segment name to a path inside ARCHIVE_DIR (no traversal)."""
    if "/" in name or "\\" in name or ".." in name:
        return None
    candidate = (ARCHIVE_DIR / name).resolve()
    try:
        candidate.relative_to(ARCHIVE_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None
