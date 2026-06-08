"""Phase 6 — compressed, rotating on-disk log archive (daily chunks).

The ring buffer is in-memory and bounded; this adds durable retention so history
survives restarts and can be browsed/downloaded from the Log panel. A
``TimedRotatingFileHandler`` (owned by the QueueListener) writes the **plain,
redacted, ANSI-free** stream to ``data/logs/agentx.log`` and, at each midnight,
gzips the completed day into ``agentx-YYYY-MM-DD.log.gz``.

When a keyring exists and the DEK has been unlocked (a user has logged in), each
completed day is additionally **sealed** into authenticated ciphertext
(``agentx-YYYY-MM-DD.log.gz.enc``) by :mod:`.archive_crypto`. Sealing is lazy:
the hot logging path never needs the key, so days that roll while locked stay as
redacted-plaintext ``.gz`` until the next :func:`archive_crypto.seal_pending`
(triggered on login / ``task logs:seal``).
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


def _seal_after_rotate() -> None:
    """Best-effort: seal completed days + prune, if the vault is unlocked.

    Imported lazily to avoid an import cycle (``archive_crypto`` imports paths
    from this module). Crypto must never break the logging hot path.
    """
    try:
        from . import archive_crypto

        if archive_crypto.get_cached_dek() is None:
            return
        archive_crypto.seal_pending()
        archive_crypto.prune_old(read_retention_days())
    except Exception:  # noqa: BLE001 — logging must survive any crypto failure
        logging.getLogger(__name__).debug("post-rotate seal/prune skipped", exc_info=True)


def read_retention_days() -> int:
    from .flags import read_flags

    return read_flags().archive_retention_days


class DailyArchiveHandler(logging.handlers.TimedRotatingFileHandler):
    """Rotates at midnight, gzipping each day to ``agentx-YYYY-MM-DD.log.gz``.

    Retention is **not** delegated to ``backupCount``: the custom ``namer`` (and
    the later ``.enc`` rename by the sealer) make segments invisible to
    ``getFilesToDelete()``'s regex, so it would silently never prune. Pruning is
    owned by :func:`archive_crypto.prune_old`, invoked from the sealer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.namer = self._name_segment

    @staticmethod
    def _name_segment(default_name: str) -> str:
        # ``default_name`` looks like ``.../agentx.log.2026-06-08``; reshape to
        # ``.../agentx-2026-06-08.log.gz``.
        p = Path(default_name)
        date = p.name[len(ARCHIVE_BASENAME) + 1 :]  # strip "agentx.log."
        return str(p.with_name(f"agentx-{date}.log.gz")) if date else default_name + ".gz"

    def rotate(self, source: str, dest: str) -> None:
        with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(source)
        _seal_after_rotate()


class _ArchiveFormatter(logging.Formatter):
    """Plain archive formatter that appends any oversized ``llm_detail`` payload.

    The console only shows the one-line summary, but the durable archive keeps the
    full (already-redacted) content on the following lines, so it can be browsed
    or downloaded later.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        detail = getattr(record, "llm_detail", None)
        return f"{base}\n{detail}" if detail else base


def build_archive_handler(flags: LogFlags) -> Optional[logging.Handler]:
    if not flags.archive_enabled:
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    handler = DailyArchiveHandler(
        filename=str(archive_path()),
        when="midnight",
        utc=True,
        backupCount=0,  # retention owned by archive_crypto.prune_old — see class docstring
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(_ArchiveFormatter(_ARCHIVE_FMT, style="{"))
    return handler


def list_segments() -> list[dict[str, object]]:
    """List the archive segments (newest first) for ``GET /api/logs/archive``.

    Globs ``agentx*`` so it captures both the daily ``agentx-YYYY-MM-DD.log.gz``
    scheme, the sealed ``*.gz.enc`` siblings, and legacy size-based
    ``agentx.log.N.gz`` segments. The open active ``agentx.log`` is included too.
    """
    if not ARCHIVE_DIR.exists():
        return []
    out: list[dict[str, object]] = []
    for p in sorted(ARCHIVE_DIR.glob("agentx*")):
        if p.name.endswith(".tmp") or not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        encrypted = p.name.endswith(".enc")
        out.append(
            {
                "name": p.name,
                "size": st.st_size,
                "modified": st.st_mtime,
                "compressed": encrypted or p.suffix == ".gz",
                "encrypted": encrypted,
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
