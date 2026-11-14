"""
Checks the recorded files the online demo is built from (backend/demo/record_demo.py →
frontend/public/demo/). The demo has no server, so a missing or inconsistent file would
surface as a broken page for a visitor rather than an error anyone sees. Skipped until the
recording has been made.
"""
import json
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo"
pytestmark = pytest.mark.skipif(not (DEMO / "manifest.json").exists(), reason="demo not recorded yet")


def load(name: str):
    return json.loads((DEMO / name).read_text())


@pytest.fixture(scope="module")
def manifest():
    return load("manifest.json")


def test_every_manifest_entry_has_its_files(manifest):
    for ds, modes in manifest["results"].items():
        assert (DEMO / f"fetch__{ds}.json").exists()
        for mode, evaluations in modes.items():
            assert (DEMO / f"preset__{ds}__{mode}.json").exists(), f"{ds}/{mode} preset"
            for ev in evaluations:
                assert (DEMO / f"result__{ds}__{mode}__{ev}.json").exists(), f"{ds}/{mode}/{ev}"


def test_each_recorded_mode_is_offered_by_the_fetch(manifest):
    # demoApi refuses a mode with no recording; every mode the UI offers should have one
    for ds in manifest["datasets"]:
        offered = set(load(f"fetch__{ds}.json")["available_modes"])
        recorded = {m for m, evs in manifest["results"][ds].items() if "single" in evs}
        assert offered == recorded, f"{ds}: UI offers {offered}, recorded {recorded}"


def test_results_have_scores_and_the_recorded_subject(manifest):
    for ds, info in manifest["datasets"].items():
        for mode, evaluations in manifest["results"][ds].items():
            for ev in evaluations:
                res = load(f"result__{ds}__{mode}__{ev}.json")
                assert res["evaluation"] == ev
                scored = [c for c in res["classifiers"] if c["accuracy_mean"] is not None]
                assert scored, f"{ds}/{mode}/{ev}: no decoder succeeded"
                if ev == "single":
                    assert res["dataset_summary"]["subjects"] == [info["subject"]]
                else:
                    assert res["subjects"] == list(range(1, 10))


def test_catalogue_points_at_the_recorded_subject(manifest):
    listed = {}
    for source in ("mne", "moabb"):
        for d in load(f"datasets__{source}.json")["datasets"]:
            listed[d["id"]] = d["demo_subject"]
    assert listed == {ds: info["subject"] for ds, info in manifest["datasets"].items()}


def test_playback_binary_matches_its_header(manifest):
    for ds, info in manifest["datasets"].items():
        key = info.get("playback")
        if not key:
            continue
        head = load(f"playback__{key}.json")
        samples = np.fromfile(DEMO / f"playback__{key}.bin", dtype="<i2")
        assert samples.size == head["n_samples"] * len(head["channels"])
        assert head["duration_s"] == pytest.approx(head["n_samples"] / head["sfreq"])
        assert all(0 <= e["t"] < head["duration_s"] for e in head["events"])
        assert set(head["default_channels"]) <= set(head["channels"])
        # µV after scaling: real EEG, not silence or a clipped rail
        uv = samples.astype(float) / head["scale"]
        assert 1 < uv.std() < 500
        assert np.mean(np.abs(samples) == 32767) < 0.001
