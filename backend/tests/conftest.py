"""
Synthetic EEG fixtures for fast unit tests.

The signal is deliberately trivial: two classes, where one class has 3x the
variance on C4 and the other has 3x the variance on C3. That is the spatial
power difference CSP exists to find (Blankertz et al. 2008, "Optimizing
Spatial Filters for Robust EEG Single-Trial Analysis"), so any correctly
wired classification pipeline must score far above chance on it. If one
doesn't, the wiring is broken, not the data.
"""
import mne
import numpy as np
import pytest

mne.set_log_level("WARNING")

SFREQ = 128.0
# All five are standard 10-20 names, so detector.check_standard_montage counts
# them and C3 + C4 makes motor_imagery available.
CH_NAMES = ["C3", "Cz", "C4", "Fz", "Pz"]
EVENT_ID = {"left_hand": 1, "right_hand": 2}
N_PER_CLASS = 30
N_TIMES = int(SFREQ)        # 1-second trials
NOISE_V = 10e-6             # ~10 µV, a realistic scalp EEG amplitude (MNE stores volts)


def _make_trials(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns data (n_trials, n_chans, n_times) in volts and integer event codes."""
    c3, c4 = CH_NAMES.index("C3"), CH_NAMES.index("C4")
    trials, codes = [], []
    for code in (1, 2):
        for _ in range(N_PER_CLASS):
            x = rng.standard_normal((len(CH_NAMES), N_TIMES)) * NOISE_V
            # left-hand imagery -> stronger C4 here; right-hand -> stronger C3.
            # (The direction is arbitrary for the test; only the asymmetry matters.)
            x[c4 if code == 1 else c3] *= 3.0
            trials.append(x)
            codes.append(code)
    order = rng.permutation(len(codes))
    return np.stack(trials)[order], np.array(codes)[order]


@pytest.fixture
def info() -> mne.Info:
    # https://mne.tools/1.12/generated/mne.create_info.html
    return mne.create_info(CH_NAMES, SFREQ, ch_types="eeg")


@pytest.fixture
def synthetic_epochs(info) -> mne.EpochsArray:
    """
    EpochsArray — the same class MOABB's paradigm.get_data(return_epochs=True)
    returns, which is a sibling of mne.Epochs under BaseEpochs, not a subclass.
    """
    data, codes = _make_trials(np.random.default_rng(0))
    # events array: (sample, previous value, event code)
    # https://mne.tools/1.12/generated/mne.EpochsArray.html
    events = np.column_stack([
        np.arange(len(codes)) * N_TIMES,
        np.zeros(len(codes), dtype=int),
        codes,
    ])
    return mne.EpochsArray(data, info, events=events, tmin=0.0, event_id=EVENT_ID)


@pytest.fixture
def synthetic_raw(info) -> mne.io.RawArray:
    """
    Continuous RawArray with one annotation per trial, the way EDF/GDF uploads
    usually arrive (events in annotations, no stim channel).
    """
    data, codes = _make_trials(np.random.default_rng(1))
    continuous = np.concatenate(list(data), axis=1)       # (n_chans, n_trials * n_times)
    # https://mne.tools/1.12/generated/mne.io.RawArray.html
    raw = mne.io.RawArray(continuous, info)
    names = {v: k for k, v in EVENT_ID.items()}
    # https://mne.tools/1.12/generated/mne.Annotations.html
    raw.set_annotations(mne.Annotations(
        onset=np.arange(len(codes)) * (N_TIMES / SFREQ),
        duration=np.full(len(codes), N_TIMES / SFREQ),
        description=[names[c] for c in codes],
    ))
    return raw


@pytest.fixture
def labels(synthetic_epochs) -> np.ndarray:
    return synthetic_epochs.events[:, 2]
