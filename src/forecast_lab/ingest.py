"""Download a source and store an immutable raw snapshot plus a manifest.

Layout:
    data/raw/<source_id>/<UTC timestamp>_<sha12>.csv    exact downloaded bytes
    data/raw/<source_id>/<UTC timestamp>_<sha12>.json   manifest (provenance)

The SHA-256 hash is the snapshot's identity. Experiments record it, and
`load_snapshot` re-checks it, so a past experiment's input can always be
re-created byte for byte (or we find out loudly that it can't).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from .config import RAW_DIR
from .sources import Source

Fetcher = Callable[[str], bytes]


class SnapshotIntegrityError(RuntimeError):
    """The bytes on disk no longer match the hash recorded in the manifest."""


def fetch_bytes(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"forecast_lab/{__version__} (educational)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_snapshots(source_id: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """All manifests for a source, oldest first."""
    folder = Path(raw_dir) / source_id
    if not folder.exists():
        return []
    manifests = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    return sorted(manifests, key=lambda m: m["retrieved_at_utc"])


def find_snapshot(source_id: str, sha_prefix: str | None = None, raw_dir: Path = RAW_DIR) -> dict:
    """Latest snapshot, or the one whose sha256 starts with `sha_prefix`."""
    snaps = list_snapshots(source_id, raw_dir)
    if sha_prefix:
        snaps = [m for m in snaps if m["sha256"].startswith(sha_prefix)]
    if not snaps:
        hint = f" matching '{sha_prefix}'" if sha_prefix else ""
        raise FileNotFoundError(f"No snapshot of '{source_id}'{hint} in {raw_dir}. Run `forecast-lab ingest` first.")
    return snaps[-1]


def snapshot_source(
    source: Source,
    raw_dir: Path = RAW_DIR,
    fetch: Fetcher = fetch_bytes,
    now: datetime | None = None,
) -> tuple[dict, bool]:
    """Download `source` and store it. Returns (manifest, is_new).

    If the downloaded bytes are identical to an existing snapshot, nothing is
    written and the existing manifest is returned with is_new=False.
    """
    data = fetch(source.url)
    if not data:
        raise ValueError(f"Download from {source.url} returned zero bytes; refusing to store an empty snapshot.")
    digest = sha256_bytes(data)

    for existing in list_snapshots(source.source_id, raw_dir):
        if existing["sha256"] == digest:
            return existing, False

    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    folder = Path(raw_dir) / source.source_id
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / f"{stamp}_{digest[:12]}.csv"
    data_path.write_bytes(data)
    data_path.chmod(0o444)  # read-only: raw data is never edited

    manifest = {
        "source_id": source.source_id,
        "url": source.url,
        "license": source.license,
        "attribution": source.attribution,
        "retrieved_at_utc": now.isoformat(timespec="seconds"),
        "sha256": digest,
        "n_bytes": len(data),
        "n_lines": data.count(b"\n"),
        "file": data_path.name,
        "tool_version": __version__,
    }
    data_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, True


def load_snapshot(manifest: dict, raw_dir: Path = RAW_DIR) -> bytes:
    """Read a snapshot's bytes, verifying they still match the recorded hash."""
    path = Path(raw_dir) / manifest["source_id"] / manifest["file"]
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != manifest["sha256"]:
        raise SnapshotIntegrityError(
            f"{path} has sha256 {actual[:12]}..., manifest says {manifest['sha256'][:12]}... "
            "The raw file was modified; re-ingest instead of editing raw data."
        )
    return data
