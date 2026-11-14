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
    assert set(caps["available_modes"]) == {"basic", "spectral", "motor_imagery", "erp"}


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


def test_without_c3_c4_motor_imagery_is_available_but_not_recommended(synthetic_raw):
    """
    13 Nov change (NEUROSCOPE_CONTEXT.md 2.3): C3+C4 is a RECOMMENDATION, not a gate.
    CSP needs events and >= 2 EEG channels; BNCI2014_002 (channels EEG1–EEG15) was
    locked out of the mode its task needs.
    """
    raw = synthetic_raw.copy().rename_channels({"C4": "T8"})
    caps = detect_capabilities(raw)

    assert caps["motor_channels"]["motor_imagery_possible"] is False
    assert "motor_imagery" in caps["available_modes"]
    assert caps["recommended_mode"] == "spectral"


def test_motor_imagery_needs_two_eeg_channels_and_events():
    from backend.core.detector import get_available_modes
    assert "motor_imagery" not in get_available_modes(["Cz"], True, False, n_eeg_channels=1)
    assert "motor_imagery" not in get_available_modes(["C3", "C4"], False, True, n_eeg_channels=2)
    assert "motor_imagery" in get_available_modes(["EEG1", "EEG2"], True, False, n_eeg_channels=2)


def test_raw_without_events_needs_event_setup(synthetic_raw):
    raw = synthetic_raw.copy().set_annotations(None)
    caps = detect_capabilities(raw)

    assert caps["events_found"] is False
    assert caps["needs_event_setup"] is True
    assert "motor_imagery" not in caps["available_modes"]
