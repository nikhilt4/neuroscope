"""
Page viewer (core/playback.py: page_window, montages, channel_meta) and uploads in playback.
Synthetic 10-20 recordings, so montage arithmetic can be checked exactly.
"""
import mne
import numpy as np
import pytest
from fastapi import HTTPException

from backend.core import playback

mne.set_log_level("WARNING")

EEG = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
       "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
SFREQ = 250.0


def make_raw(names=EEG, extra=(("ECG", "ecg"),), seconds=40, events=True):
    ch_names = list(names) + [n for n, _ in extra]
    ch_types = ["eeg"] * len(names) + [t for _, t in extra]
    info = mne.create_info(ch_names, SFREQ, ch_types)
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((len(ch_names), int(SFREQ * seconds))) * 1e-5, info)
    if events:
        raw.set_annotations(mne.Annotations([5.0, 15.0, 25.0], [1, 1, 1], ["eyes_closed", "eyes_open", "eyes_closed"]))
    return raw


def test_montages_offered_match_the_electrodes():
    assert playback.available_montages(make_raw()) == ["referential", "average", "bipolar"]
    # Sleep-EDF style: two channels that are already bipolar
    two = make_raw(names=["Fpz-Cz", "Pz-Oz"], extra=())
    assert playback.available_montages(two) == ["referential"]


def test_bipolar_is_the_difference_of_each_electrode_pair_in_clinical_order():
    raw = make_raw()
    page = playback.page_window(raw, 0, 4, montage="bipolar")
    names = [c["name"] for c in page["channels"]]
    assert names[:4] == ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]       # right parasagittal first
    assert names[-1] == "ECG"                                       # non-EEG channels follow
    raw_uv = raw.get_data() * 1e6
    i = {n: k for k, n in enumerate(raw.ch_names)}
    step = int(SFREQ / page["sfreq"])
    expected = (raw_uv[i["Fp2"]] - raw_uv[i["F4"]])[: int(4 * SFREQ)][::step]
    np.testing.assert_allclose(page["data"][0], np.round(expected, 1), atol=0.051)


def test_old_10_20_names_still_make_a_bipolar_montage():
    old = [{"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(n, n) for n in EEG]
    page = playback.page_window(make_raw(names=old), 0, 2, montage="bipolar")
    assert "F7-T3" in [c["name"] for c in page["channels"]]


def test_average_reference_leaves_eeg_summing_to_zero_and_ecg_untouched():
    raw = make_raw()
    page = playback.page_window(raw, 0, 2, montage="average")
    data = np.array(page["data"])
    eeg_rows = [k for k, c in enumerate(page["channels"]) if c["type"] == "eeg"]
    assert np.abs(data[eeg_rows].mean(axis=0)).max() < 0.1          # µV, rounding only
    ecg = raw.get_data(picks="ECG")[0][: int(2 * SFREQ)] * 1e6
    step = int(SFREQ / page["sfreq"])
    np.testing.assert_allclose(data[-1], np.round(ecg[::step], 1), atol=0.051)


def test_channel_groups_follow_10_20_conventions():
    meta = {m["name"]: m["group"] for m in playback.channel_meta(make_raw())}
    assert meta["Fp1"] == "left" and meta["F4"] == "right" and meta["Cz"] == "midline"
    assert meta["ECG"] == "ecg"


def test_window_returns_its_events_and_is_clamped_to_the_recording():
    raw = make_raw(seconds=40)
    page = playback.page_window(raw, 10, 10)
    assert [e["label"] for e in page["events"]] == ["eyes_open"]
    end = playback.page_window(raw, 1e6, 10)                       # past the end → last page
    assert end["start"] == pytest.approx(30.0)
    assert len(end["data"][0]) == pytest.approx(10 * end["sfreq"], abs=1)


def test_decimation_never_drops_below_the_display_floor():
    page = playback.page_window(make_raw(seconds=60), 0, 60)
    assert page["sfreq"] >= playback.MIN_VIEW_SFREQ
    assert len(page["data"][0]) <= 60 * SFREQ


def test_channel_selection_and_bad_requests():
    raw = make_raw()
    page = playback.page_window(raw, 0, 2, channels=["Cz", "ECG"])
    assert [c["name"] for c in page["channels"]] == ["Cz", "ECG"]
    with pytest.raises(HTTPException):
        playback.page_window(raw, 0, 2, channels=["nope"])
    with pytest.raises(HTTPException) as err:
        playback.page_window(make_raw(names=["Fpz-Cz", "Pz-Oz"], extra=()), 0, 2, montage="bipolar")
    assert "isn't available" in err.value.detail


def test_uploaded_recordings_play_with_eeg_eog_ecg_only(monkeypatch):
    raw = make_raw(extra=(("ECG", "ecg"), ("EOG", "eog"), ("EMG", "emg")))
    monkeypatch.setattr(playback, "reload_data", lambda s: raw)
    out = playback.load_for_playback({"source": "upload", "session_id": "x"})
    assert "ECG" in out.ch_names and "EOG" in out.ch_names and "EMG" not in out.ch_names
    info = playback.viewer_info(out)
    assert "bipolar" in info["montages"] and info["channel_meta"][0]["group"] == "left"
