"""Envelope encryption for durable log archives.

The on-disk archive (``archive.py``) rolls one **redacted-plaintext** gzip per
calendar day. This module *seals* completed days into authenticated ciphertext
(``<segment>.gz.enc``) so history at rest is unreadable without the user's login
password — defense-in-depth on top of capture-time redaction.

Design (see the plan): **envelope encryption**.

* A random 256-bit **DEK** (data-encryption key) encrypts every archive segment.
* The user password only derives a **KEK** (via Scrypt) that *wraps* the DEK in a
  tiny ``keyring.json``. Changing the password re-wraps that one small key — an
  ``O(1)`` operation that touches no archive (:func:`rewrap_dek`). A full
  re-encrypt (:func:`reencrypt_all`) stays available for "assume the DEK leaked".

The logger starts at boot before anyone logs in, so the hot path never holds the
key: sealing is **lazy**. The unwrapped DEK is cached in process memory once a
user authenticates (:func:`set_cached_dek`); :func:`seal_pending` then encrypts
any day that rolled while we were locked.

Container format (``.enc``), all integers little-endian::

    magic   b"AXLOG1\\0"            (7 bytes)
    flags   uint8                   (bit0 = inner payload is gzip)
    frame   uint32                  (plaintext frame size used by the writer)
    repeated frames:
        nonce   12 bytes
        ct_len  uint32
        ct      ct_len bytes        (AES-256-GCM of one plaintext frame)
    terminator frame: ct_len == 0   (an empty AAD-bound frame marking clean EOF)

Each frame's AAD is ``magic || frame_index`` so frames cannot be reordered or
duplicated; the explicit zero-length terminator makes truncation (dropped
trailing frames) detectable too.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .archive import ARCHIVE_BASENAME, ARCHIVE_DIR

logger = logging.getLogger(__name__)

KEYRING_PATH = ARCHIVE_DIR / "keyring.json"

_MAGIC = b"AXLOG1\0"
_FLAG_GZIP = 0x01
_FRAME_SIZE = 1024 * 1024  # 1 MiB plaintext frames bound GCM memory on large days
_DEK_BYTES = 32  # AES-256
_SALT_BYTES = 16
_NONCE_BYTES = 12
_KEK_AAD = b"AXLOG-KEK"

# Scrypt work factors (interactive-ish; the keyring is unwrapped at most once per
# login, never in a hot loop).
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1

_lock = threading.RLock()
_dek: Optional[bytes] = None


class BadPassword(Exception):
    """Raised when a password fails to unwrap the keyring (GCM auth failure)."""


class VaultLocked(Exception):
    """Raised when an operation needs the DEK but none is available/cached."""


# --------------------------------------------------------------------------- #
# Key cache (process-memory only — never persisted to disk or Redis)
# --------------------------------------------------------------------------- #
def set_cached_dek(dek: bytes) -> None:
    global _dek
    with _lock:
        _dek = dek


def get_cached_dek() -> Optional[bytes]:
    with _lock:
        return _dek


def clear_cached_dek() -> None:
    global _dek
    with _lock:
        _dek = None


def is_encryption_active() -> bool:
    """True when a keyring exists (i.e. archives should be sealed)."""
    return KEYRING_PATH.exists()


# --------------------------------------------------------------------------- #
# Keyring (wrapped DEK)
# --------------------------------------------------------------------------- #
def derive_kek(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=_DEK_BYTES, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64

    return base64.b64decode(text.encode("ascii"))


def _write_keyring(dek: bytes, password: str, *, created_at: str) -> None:
    salt = secrets.token_bytes(_SALT_BYTES)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    kek = derive_kek(password, salt)
    wrapped = AESGCM(kek).encrypt(nonce, dek, _KEK_AAD)
    payload = {
        "version": 1,
        "kdf": "scrypt",
        "scrypt": {"n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P},
        "salt": _b64(salt),
        "wrap_nonce": _b64(nonce),
        "wrapped_dek": _b64(wrapped),
        "created_at": created_at,
        "rotated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(KEYRING_PATH, json.dumps(payload, indent=2).encode("utf-8"))


def create_keyring(password: str) -> bytes:
    """Create the keyring with a fresh random DEK if none exists; return the DEK.

    No-op (returns the existing unwrapped DEK) when a keyring is already present,
    so calling this on every setup/first-login is safe.
    """
    with _lock:
        if KEYRING_PATH.exists():
            return unwrap_dek(password)
        dek = secrets.token_bytes(_DEK_BYTES)
        _write_keyring(dek, password, created_at=datetime.now(timezone.utc).isoformat())
        logger.info("Log-archive keyring created")
        return dek


def unwrap_dek(password: str) -> bytes:
    """Decrypt the DEK from the keyring; raise :class:`BadPassword` on mismatch."""
    if not KEYRING_PATH.exists():
        raise VaultLocked("no keyring")
    data = json.loads(KEYRING_PATH.read_text("utf-8"))
    scrypt = data.get("scrypt", {})
    kdf = Scrypt(
        salt=_unb64(data["salt"]),
        length=_DEK_BYTES,
        n=scrypt.get("n", _SCRYPT_N),
        r=scrypt.get("r", _SCRYPT_R),
        p=scrypt.get("p", _SCRYPT_P),
    )
    kek = kdf.derive(password.encode("utf-8"))
    try:
        return AESGCM(kek).decrypt(_unb64(data["wrap_nonce"]), _unb64(data["wrapped_dek"]), _KEK_AAD)
    except InvalidTag as exc:
        raise BadPassword("incorrect password for log-archive keyring") from exc


def rewrap_dek(new_password: str, *, dek: Optional[bytes] = None, old_password: Optional[str] = None) -> None:
    """Re-wrap the existing DEK under ``new_password`` (``O(1)``, touches no archive).

    Prefers an in-memory ``dek`` (the cached key from login) so rotation doesn't
    depend on the old wrapping being intact; otherwise unwraps with
    ``old_password``.
    """
    with _lock:
        if dek is None:
            if old_password is None:
                raise ValueError("rewrap_dek needs either dek= or old_password=")
            dek = unwrap_dek(old_password)
        created_at = datetime.now(timezone.utc).isoformat()
        if KEYRING_PATH.exists():
            try:
                created_at = json.loads(KEYRING_PATH.read_text("utf-8")).get("created_at", created_at)
            except (OSError, ValueError):
                pass
        _write_keyring(dek, new_password, created_at=created_at)
    logger.info("Log-archive keyring re-wrapped under new password")


# --------------------------------------------------------------------------- #
# Frame codec
# --------------------------------------------------------------------------- #
def _seal_stream(src: BinaryIO, dst: BinaryIO, dek: bytes, *, gzip_inner: bool) -> None:
    aead = AESGCM(dek)
    dst.write(_MAGIC)
    dst.write(struct.pack("<B", _FLAG_GZIP if gzip_inner else 0))
    dst.write(struct.pack("<I", _FRAME_SIZE))
    index = 0
    while True:
        chunk = src.read(_FRAME_SIZE)
        if not chunk:
            break
        nonce = secrets.token_bytes(_NONCE_BYTES)
        ct = aead.encrypt(nonce, chunk, _MAGIC + struct.pack("<I", index))
        dst.write(nonce)
        dst.write(struct.pack("<I", len(ct)))
        dst.write(ct)
        index += 1
    # Zero-length terminator frame → clean-EOF / truncation detection.
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ct = aead.encrypt(nonce, b"", _MAGIC + struct.pack("<I", index))
    dst.write(nonce)
    dst.write(struct.pack("<I", len(ct)))
    dst.write(ct)


def _read_exact(src: BinaryIO, n: int) -> bytes:
    buf = src.read(n)
    if len(buf) != n:
        raise ValueError("truncated archive segment")
    return buf


def unseal_iter(path: Path, dek: bytes) -> Iterator[bytes]:
    """Yield decrypted plaintext frames (the inner gzip bytes) from a ``.enc`` file."""
    aead = AESGCM(dek)
    with open(path, "rb") as src:
        if _read_exact(src, len(_MAGIC)) != _MAGIC:
            raise ValueError("not an AgentX sealed archive")
        _read_exact(src, 1)  # flags (gzip_inner) — informational; payload is served as-is
        _read_exact(src, 4)  # frame size — informational
        index = 0
        saw_terminator = False
        while True:
            head = src.read(_NONCE_BYTES + 4)
            if not head:
                break
            if len(head) != _NONCE_BYTES + 4:
                raise ValueError("truncated archive segment")
            nonce = head[:_NONCE_BYTES]
            (ct_len,) = struct.unpack("<I", head[_NONCE_BYTES:])
            ct = _read_exact(src, ct_len)
            try:
                pt = aead.decrypt(nonce, ct, _MAGIC + struct.pack("<I", index))
            except InvalidTag as exc:
                raise ValueError("archive segment failed authentication (tampered or wrong key)") from exc
            index += 1
            if pt == b"":
                saw_terminator = True
                break
            yield pt
        if not saw_terminator:
            raise ValueError("archive segment is missing its terminator (truncated)")


def unseal_bytes(path: Path, dek: bytes) -> bytes:
    return b"".join(unseal_iter(path, dek))


# --------------------------------------------------------------------------- #
# Sealing / pruning over the archive directory
# --------------------------------------------------------------------------- #
def _enc_path(gz: Path) -> Path:
    return gz.with_name(gz.name + ".enc")


def _dated_gz_segments() -> list[Path]:
    """Rotated, gzipped day segments (excludes the open active ``agentx.log``)."""
    if not ARCHIVE_DIR.exists():
        return []
    return [p for p in ARCHIVE_DIR.glob(ARCHIVE_BASENAME.split(".")[0] + "*.gz") if p.is_file()]


def seal_segment(gz: Path, dek: bytes) -> Optional[Path]:
    """Encrypt one ``.gz`` to ``<name>.gz.enc`` and remove the plaintext source.

    Atomic (tmp + ``os.replace``); the ``.gz`` is removed only after the ``.enc``
    is durably written. Idempotent/race-tolerant: a missing source or a
    pre-existing ``.enc`` is treated as already-sealed.
    """
    enc = _enc_path(gz)
    if enc.exists():
        # Already sealed by a concurrent sealer; drop the stale plaintext.
        gz.unlink(missing_ok=True)
        return enc
    tmp = enc.with_suffix(enc.suffix + ".tmp")
    try:
        with open(gz, "rb") as src, open(tmp, "wb") as dst:
            _seal_stream(src, dst, dek, gzip_inner=True)
            dst.flush()
            os.fsync(dst.fileno())
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, enc)
    gz.unlink(missing_ok=True)
    return enc


def seal_pending(dek: Optional[bytes] = None) -> int:
    """Seal every dated ``.gz`` lacking a sibling ``.enc``. Returns the count sealed."""
    dek = dek or get_cached_dek()
    if dek is None:
        raise VaultLocked("no DEK available to seal pending segments")
    count = 0
    for gz in _dated_gz_segments():
        try:
            if seal_segment(gz, dek) is not None:
                count += 1
        except Exception as exc:  # best-effort: one bad segment must not stop the rest
            logger.warning("Failed to seal log segment %s: %s", gz.name, exc)
    if count:
        logger.info("Sealed %d log archive segment(s)", count)
    return count


def prune_old(retention_days: int) -> int:
    """Delete sealed/plaintext day segments older than ``retention_days``.

    Our own retention — ``TimedRotatingFileHandler.backupCount`` can't see the
    custom ``namer`` + ``.enc`` rename, so it would silently never prune.
    """
    if retention_days <= 0 or not ARCHIVE_DIR.exists():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    patterns = (ARCHIVE_BASENAME.split(".")[0] + "*.gz", ARCHIVE_BASENAME.split(".")[0] + "*.enc")
    for pattern in patterns:
        for p in ARCHIVE_DIR.glob(pattern):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
    if removed:
        logger.info("Pruned %d log archive segment(s) older than %d days", removed, retention_days)
    return removed


def reencrypt_all(old_password: str, new_password: str) -> int:
    """Deep rotation: decrypt every ``.enc`` with the old DEK, re-encrypt under a
    brand-new DEK, then re-wrap that DEK with ``new_password``.

    Use when the old DEK is assumed compromised. Returns the number of segments
    re-encrypted.
    """
    with _lock:
        old_dek = unwrap_dek(old_password)
        new_dek = secrets.token_bytes(_DEK_BYTES)
        count = 0
        for enc in sorted(ARCHIVE_DIR.glob("*.enc")):
            tmp = enc.with_suffix(enc.suffix + ".tmp")
            try:
                with open(tmp, "wb") as dst:
                    # Stream plaintext frames straight from old → new sealing.
                    aead = AESGCM(new_dek)
                    dst.write(_MAGIC)
                    dst.write(struct.pack("<B", _FLAG_GZIP))
                    dst.write(struct.pack("<I", _FRAME_SIZE))
                    index = 0
                    for frame in unseal_iter(enc, old_dek):
                        nonce = secrets.token_bytes(_NONCE_BYTES)
                        ct = aead.encrypt(nonce, frame, _MAGIC + struct.pack("<I", index))
                        dst.write(nonce + struct.pack("<I", len(ct)) + ct)
                        index += 1
                    nonce = secrets.token_bytes(_NONCE_BYTES)
                    ct = aead.encrypt(nonce, b"", _MAGIC + struct.pack("<I", index))
                    dst.write(nonce + struct.pack("<I", len(ct)) + ct)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(tmp, enc)
                count += 1
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        created_at = datetime.now(timezone.utc).isoformat()
        _write_keyring(new_dek, new_password, created_at=created_at)
        set_cached_dek(new_dek)
    logger.info("Re-encrypted %d log archive segment(s) under a new DEK", count)
    return count


def keyring_status() -> dict[str, object]:
    """Summary for the management command / status surfaces."""
    sealed = len(list(ARCHIVE_DIR.glob("*.enc"))) if ARCHIVE_DIR.exists() else 0
    pending = len([p for p in _dated_gz_segments() if not _enc_path(p).exists()])
    info: dict[str, object] = {
        "keyring_present": KEYRING_PATH.exists(),
        "unlocked": get_cached_dek() is not None,
        "sealed_segments": sealed,
        "pending_segments": pending,
    }
    if KEYRING_PATH.exists():
        try:
            data = json.loads(KEYRING_PATH.read_text("utf-8"))
            info["created_at"] = data.get("created_at")
            info["rotated_at"] = data.get("rotated_at")
        except (OSError, ValueError):
            pass
    return info
