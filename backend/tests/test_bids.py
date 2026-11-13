"""
BIDS ZIP upload, end to end.

The fixture writes a real BIDS dataset with mne_bids.write_raw_bids (so it is
spec-conformant by construction), then zips it the way macOS "Compress" does:
everything inside one top-level folder, plus __MACOSX resource-fork junk.
"""
import asyncio
import io
import zipfile

import pytest
from fastapi import HTTPException, UploadFile
from mne_bids import BIDSPath, write_raw_bids

from backend.core.detector import detect_capabilities
from backend.loaders import zip_handler as zip_module
from backend.loaders.bids_loader import load_bids, scan_bids_entities
from backend.loaders.zip_handler import zip_handler

# sub-01 has two runs, sub-02 one: lets the tests check combinations are real, not a
# cross product of every subject x every run
RECORDINGS = [("01", "1"), ("01", "2"), ("02", "1")]


@pytest.fixture
def bids_dataset(tmp_path, synthetic_raw):
    root = tmp_path / "src" / "MyStudy"
    raw = synthetic_raw.copy()
    raw.info["line_freq"] = 50          # BIDS requires PowerLineFrequency in eeg.json
    for subject, run in RECORDINGS:
        # https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html (MNE-BIDS 0.19)
        # allow_preload + format are required for in-memory RawArray data
        write_raw_bids(
            raw, BIDSPath(subject=subject, task="imagery", run=run, datatype="eeg", root=root),
            allow_preload=True, format="BrainVision", overwrite=True, verbose=False,
        )
    return root


def _zip_dir(root, zip_path, with_macos_junk=True):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in root.rglob("*"):
            if f.is_file():
                zf.write(f, f"{root.name}/{f.relative_to(root).as_posix()}")
        if with_macos_junk:
            zf.writestr(f"__MACOSX/{root.name}/._dataset_description.json", b"junk")
            zf.writestr(f"{root.name}/.DS_Store", b"junk")
    return zip_path


@pytest.fixture
def bids_zip(tmp_path, bids_dataset):
    return _zip_dir(bids_dataset, tmp_path / "MyStudy.zip")


# ── zip_handler ───────────────────────────────────────────────────────────────

def test_zip_handler_finds_nested_root_and_skips_junk(tmp_path, bids_zip):
    root = zip_handler(bids_zip, "sid", base_dir=tmp_path / "out")

    assert root.name == "MyStudy"
    assert (root / "dataset_description.json").exists()
    assert not list((tmp_path / "out").rglob("__MACOSX")), "__MACOSX must not be extracted"
    assert not list((tmp_path / "out").rglob(".DS_Store"))
    assert not bids_zip.exists(), "the ZIP is deleted once extracted"


def test_zip_handler_rejects_non_zip(tmp_path):
    fake = tmp_path / "data.zip"
    fake.write_bytes(b"definitely not a zip")
    with pytest.raises(HTTPException) as exc:
        zip_handler(fake, "sid", base_dir=tmp_path / "out")
    assert exc.value.status_code == 400


def test_zip_handler_rejects_zip_without_bids(tmp_path):
    z = tmp_path / "notes.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("notes/readme.txt", "hello")
    with pytest.raises(HTTPException) as exc:
        zip_handler(z, "sid", base_dir=tmp_path / "out")
    assert exc.value.status_code == 400
    assert "dataset_description.json" in exc.value.detail


def test_zip_handler_refuses_oversized_archive_before_extracting(tmp_path, bids_zip, monkeypatch):
    monkeypatch.setattr(zip_module, "MAX_UNCOMPRESSED_BYTES", 10)
    with pytest.raises(HTTPException) as exc:
        zip_handler(bids_zip, "sid", base_dir=tmp_path / "out")
    assert exc.value.status_code == 413
    assert not list((tmp_path / "out").rglob("*.vhdr")), "nothing extracted"


# ── scanning + loading ────────────────────────────────────────────────────────

def test_scan_returns_flat_real_combinations(bids_dataset):
    combos = scan_bids_entities(bids_dataset)

    assert combos == [
        {"subject": "01", "session": None, "task": "imagery", "run": "1"},
        {"subject": "01", "session": None, "task": "imagery", "run": "2"},
        {"subject": "02", "session": None, "task": "imagery", "run": "1"},
    ]
    # sub-02 run 2 doesn't exist and must not be offered
    assert {"subject": "02", "session": None, "task": "imagery", "run": "2"} not in combos


def test_loaded_recording_has_events_and_motor_channels(bids_dataset):
    raw = load_bids(bids_dataset, subject="01", task="imagery", run="2")
    caps = detect_capabilities(raw)

    assert caps["events_found"] is True
    assert {"left_hand", "right_hand"} <= set(caps["event_classes"])
    assert "motor_imagery" in caps["available_modes"]


def test_load_missing_recording_is_a_400(bids_dataset):
    with pytest.raises(HTTPException) as exc:
        load_bids(bids_dataset, subject="02", task="imagery", run="2")
    assert exc.value.status_code == 400


# ── routes: upload -> select -> analyze ───────────────────────────────────────

def test_full_bids_flow_through_routes(bids_zip):
    """
    Goes through the real route functions, so it writes under /tmp/neuroscope like
    the app does. The session is deleted at the end either way.
    """
    from backend.models import AnalysisRequest, BidsSelection
    from backend.routes.analyze import analyze
    from backend.routes.upload import configure_bids, upload_bids
    from backend.utils.file_utils import cleanup_session
    from backend.utils.session_registry import get_session

    upload = UploadFile(file=io.BytesIO(bids_zip.read_bytes()), filename="MyStudy.zip")
    resp = asyncio.run(upload_bids(upload))
    try:
        assert resp.needs_bids_selection is True
        assert get_session(resp.session_id)["is_ready"] is False
        assert len(resp.bids_entities) == 3

        # analysing before choosing a recording must be refused cleanly
        req = AnalysisRequest(session_id=resp.session_id, mode="motor_imagery",
                              classifiers={"selected": ["lda"]})
        with pytest.raises(HTTPException) as exc:
            asyncio.run(analyze(req))
        assert exc.value.status_code == 400

        configured = asyncio.run(configure_bids(BidsSelection(
            session_id=resp.session_id, subject="01", task="imagery", run="1")))
        assert configured.is_ready is True
        assert "motor_imagery" in configured.available_modes

        out = asyncio.run(analyze(req))
        assert out["classifiers"][0]["accuracy_mean"] is not None
        assert set(out["dataset_summary"]["classes"]) >= {"left_hand", "right_hand"}
    finally:
        cleanup_session(resp.session_id)


def test_upload_rejects_non_zip_filename():
    from backend.routes.upload import upload_bids
    upload = UploadFile(file=io.BytesIO(b"x"), filename="recording.edf")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_bids(upload))
    assert exc.value.status_code == 400
