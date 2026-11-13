"""
detect_capabilities must handle Raw and every BaseEpochs flavour.

The bug this guards: EpochsFIF (from mne.read_epochs) and EpochsArray (from
MOABB) are siblings of mne.Epochs under mne.BaseEpochs, not subclasses, so an
isinstance(obj, mne.Epochs) check sent kiloword down the Raw branch, where it
crashed on obj.n_times.
"""
import mne
import pytest

from backend.core.detector import detect_capabilities


def test_raw_with_annotations(synthetic_raw):
    caps = detect_capabilities(synthetic_raw)

    assert caps["is_epochs"] is False
    assert caps["n_epochs"] is None
    assert caps["events_found"] is True
    assert sorted(caps["event_classes"]) == ["left_hand", "right_hand"]
    assert caps["duration_seconds"] == pytest.approx(60.0)
    assert caps["needs_event_setup"] is False
    assert set(caps["available_modes"]) == {"basic", "spectral", "motor_imagery"}


def test_epochs_array(synthetic_epochs):
    assert not isinstance(synthetic_epochs, mne.Epochs)   # the trap, made explicit
    caps = detect_capabilities(synthetic_epochs)

    assert caps["is_epochs"] is True
    assert caps["n_epochs"] == 60
    assert caps["event_classes"] == ["left_hand", "right_hand"]
    assert "motor_imagery" in caps["available_modes"]


def test_epochs_read_back_from_fif(tmp_path, synthetic_epochs):
    # MNE requires epochs filenames to end in -epo.fif
    # https://mne.tools/1.12/generated/mne.read_epochs.html
    path = tmp_path / "synthetic-epo.fif"
    synthetic_epochs.save(path)
    epochs_fif = mne.read_epochs(path)

    assert type(epochs_fif).__name__ == "EpochsFIF"
    caps = detect_capabilities(epochs_fif)
    assert caps["is_epochs"] is True
    assert caps["n_epochs"] == 60


def test_motor_imagery_needs_both_c3_and_c4(synthetic_raw):
    raw = synthetic_raw.copy().rename_channels({"C4": "T8"})
    caps = detect_capabilities(raw)

    assert caps["motor_channels"]["motor_imagery_possible"] is False
    assert "motor_imagery" not in caps["available_modes"]
    assert "spectral" in caps["available_modes"]      # still 5 standard channels


def test_raw_without_events_needs_event_setup(synthetic_raw):
    raw = synthetic_raw.copy().set_annotations(None)
    caps = detect_capabilities(raw)

    assert caps["events_found"] is False
    assert caps["needs_event_setup"] is True
    assert "motor_imagery" not in caps["available_modes"]
