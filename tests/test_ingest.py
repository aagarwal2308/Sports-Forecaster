"""AT-1: raw snapshots are exact, provenance is recorded, and tampering is detected."""

import hashlib
from datetime import datetime, timezone

import pytest

from forecast_lab.ingest import SnapshotIntegrityError, find_snapshot, list_snapshots, load_snapshot, snapshot_source
from forecast_lab.sources import NFLVERSE_SCHEDULES

T1 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def fake_fetch(payload: bytes):
    return lambda url: payload  # stands in for the network in tests


def test_snapshot_stores_exact_bytes_and_manifest(tmp_path):
    payload = b"game_id,season\nX,2020\n"
    manifest, is_new = snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(payload), now=T1)
    assert is_new
    assert load_snapshot(manifest, tmp_path) == payload
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    for field in ("url", "license", "attribution", "retrieved_at_utc", "n_bytes", "file"):
        assert manifest[field]
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["retrieved_at_utc"] == "2026-10-01T12:00:00+00:00"


def test_identical_download_is_not_duplicated(tmp_path):
    payload = b"a,b\n1,2\n"
    snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(payload), now=T1)
    _, is_new = snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(payload), now=T2)
    assert not is_new
    assert len(list_snapshots(NFLVERSE_SCHEDULES.source_id, tmp_path)) == 1


def test_changed_source_keeps_both_versions(tmp_path):
    m1, _ = snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(b"v1\n"), now=T1)
    m2, _ = snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(b"v2\n"), now=T2)
    assert find_snapshot(NFLVERSE_SCHEDULES.source_id, raw_dir=tmp_path)["sha256"] == m2["sha256"]  # latest
    old = find_snapshot(NFLVERSE_SCHEDULES.source_id, m1["sha256"][:8], raw_dir=tmp_path)
    assert load_snapshot(old, tmp_path) == b"v1\n"  # an old experiment's input is still reproducible


def test_tampered_raw_file_is_detected(tmp_path):
    manifest, _ = snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(b"original\n"), now=T1)
    path = tmp_path / manifest["source_id"] / manifest["file"]
    path.chmod(0o644)
    path.write_bytes(b"edited\n")
    with pytest.raises(SnapshotIntegrityError):
        load_snapshot(manifest, tmp_path)


def test_empty_download_is_refused(tmp_path):
    with pytest.raises(ValueError):
        snapshot_source(NFLVERSE_SCHEDULES, tmp_path, fake_fetch(b""), now=T1)


def test_missing_snapshot_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        find_snapshot("nflverse_schedules", raw_dir=tmp_path)
