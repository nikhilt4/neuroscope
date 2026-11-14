"""Playback: event timing, frame contents, seeking, and which sources are allowed."""
import json

import mne
import numpy as np
import pytest
from fastapi import HTTPException

from backend.core import playback


def _frames(gen):
    return [json.loads(chunk[len("data: "):]) for chunk in gen]


def test_stream_carries_every_sample_once_in_microvolts(synthetic_raw):
    frames = _frames(playback.stream_frames(synthetic_raw, ["C3", "C4"], pace=False))
    assert frames[0]["type"] == "start" and frames[-1]["type"] == "end"
    data = [f for f in frames if f["type"] == "data"]
    samples = np.concatenate([np.array(f["samples"]) for f in data], axis=1)
    assert samples.shape == (2, synthetic_raw.n_times)
    expected = synthetic_raw.get_data(picks=["C3", "C4"])[:, :10] * 1e6
    assert np.allclose(samples[:, :10], np.round(expected, 2))
    # ~25 frames per second of recording
    assert len(data) == pytest.approx(synthetic_raw.n_times / synthetic_raw.info["sfreq"] * 25, rel=0.05)


def test_events_arrive_in_order_before_the_frame_that_passes_them(synthetic_raw):
    frames = _frames(playback.stream_frames(synthetic_raw, pace=False))
    events = [f for f in frames if f["type"] == "event"]
    assert len(events) == len(synthetic_raw.annotations)
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)
    assert {e["label"] for e in events} == {"left_hand", "right_hand"}
    # each event is emitted just before the data frame that contains its time
    for i, f in enumerate(frames):
        if f["type"] == "event":
            nxt = next(x for x in frames[i:] if x["type"] == "data")
            assert nxt["t"] <= f["t"] < nxt["t"] + 0.05


def test_seek_starts_mid_recording_and_skips_earlier_events(synthetic_raw):
    frames = _frames(playback.stream_frames(synthetic_raw, start=30.0, pace=False))
    assert frames[0]["t"] == pytest.approx(30.0)
    assert min(f["t"] for f in frames if f["type"] == "event") >= 30.0


def test_default_channels_put_motor_sites_first():
    names = ["Fp1", "Fz", "Pz", "O1", "C4", "C3", "Cz", "F3", "P3", "O2"]
    assert playback.default_channels(names)[:3] == ["C3", "Cz", "C4"]
    assert len(playback.default_channels(names)) == 8


def test_unknown_channels_are_a_clear_error(synthetic_raw):
    with pytest.raises(HTTPException) as exc:
        list(playback.stream_frames(synthetic_raw, ["Nope"], pace=False))
    assert exc.value.status_code == 400


def test_pre_epoched_and_upload_sessions_are_refused(monkeypatch, synthetic_epochs):
    monkeypatch.setattr(playback, "reload_data", lambda s: synthetic_epochs)
    with pytest.raises(HTTPException) as exc:
        playback.load_for_playback({"source": "mne", "dataset_name": "limo"})
    assert "already cut into trials" in exc.value.detail
    with pytest.raises(HTTPException) as exc:
        playback.load_for_playback({"source": "upload"})
    assert "coming later" in exc.value.detail


def test_load_applies_display_filter_and_renames(monkeypatch, info):
    """eegbci-style T1/T2 become left/right hand; a 0.5–40 Hz display filter is applied."""
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((len(info.ch_names), int(info["sfreq"] * 30))) * 1e-5, info)
    raw.set_annotations(mne.Annotations([2, 6, 10], [4, 4, 4], ["T0", "T1", "T2"]))
    monkeypatch.setattr(playback, "reload_data", lambda s: raw)
    out = playback.load_for_playback({"source": "mne", "dataset_name": "eegbci", "runs": [4, 8, 12]})
    assert {e["label"] for e in playback.event_list(out)} == {"T0", "left_hand", "right_hand"}
    assert out.info["highpass"] == pytest.approx(0.5)
    assert out.info["lowpass"] == pytest.approx(40.0)


def test_prefill_sends_the_previous_window_at_once(synthetic_raw):
    """After a seek the screen should be full immediately, like scrubbing a video."""
    sf = synthetic_raw.info["sfreq"]
    frames = _frames(playback.stream_frames(synthetic_raw, ["C3"], start=30.0, prefill=10.0, pace=False))
    first_data = next(f for f in frames if f["type"] == "data")
    assert first_data["prefill"] is True
    assert first_data["t"] == pytest.approx(20.0)
    assert len(first_data["samples"][0]) == int(10 * sf)
    pre_events = [f for f in frames if f["type"] == "event" and f.get("prefill")]
    assert pre_events and all(20.0 <= e["t"] < 30.0 for e in pre_events)


def test_preview_stops_after_the_prefill(synthetic_raw):
    frames = _frames(playback.stream_frames(synthetic_raw, ["C3"], start=30.0, prefill=10.0,
                                            preview=True, pace=False))
    assert [f["type"] for f in frames if f["type"] != "event"] == ["start", "data", "end"]
    assert frames[-1]["preview"] is True


def test_prefill_at_the_very_start_is_harmless(synthetic_raw):
    frames = _frames(playback.stream_frames(synthetic_raw, ["C3"], start=0.0, prefill=10.0, pace=False))
    assert not any(f.get("prefill") for f in frames if f["type"] == "data")
