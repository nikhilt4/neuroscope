"""
Regression tests for the correctness fixes of 13 Nov 2026
(NEUROSCOPE_CONTEXT.md section 5 items 1, 3, 4 and section 7 Correctness).
"""
import asyncio

import mne
import numpy as np
import pytest
from fastapi import HTTPException

from backend.core.features import FREQUENCY_BANDS
from backend.core.presets import DATASET_PRESETS, get_preset
from backend.core.visualizer import spectral_summary
from backend.fetchers.moabb_fetcher import restore_class_names
from backend.models import AnalysisRequest
from backend.routes import analyze as analyze_route


# ── MOABB class names ─────────────────────────────────────────────────────────

def _numeric_epochs(epochs):
    """Mimic MOABB: same data, but event_id {'1': 1, '2': 2}."""
    out = epochs.copy()
    out.event_id = {"1": 1, "2": 2}
    return out


def test_restore_class_names_from_moabb_labels(synthetic_epochs):
    epochs = _numeric_epochs(synthetic_epochs)
    names = {1: "left_hand", 2: "right_hand"}
    labels = np.array([names[c] for c in epochs.events[:, 2]])

    restored = restore_class_names(epochs, labels)
    assert restored.event_id == {"left_hand": 1, "right_hand": 2}


def test_restore_class_names_refuses_inconsistent_labels(synthetic_epochs):
    epochs = _numeric_epochs(synthetic_epochs)
    labels = np.array(["left_hand"] * len(epochs))     # both codes -> one name
    assert restore_class_names(epochs, labels).event_id == {"1": 1, "2": 2}

    labels = np.array(["a", "b"] * (len(epochs) // 2))  # one code -> two names
    assert restore_class_names(epochs, labels).event_id == {"1": 1, "2": 2}


def test_restore_class_names_refuses_length_mismatch(synthetic_epochs):
    epochs = _numeric_epochs(synthetic_epochs)
    assert restore_class_names(epochs, np.array(["x"])).event_id == {"1": 1, "2": 2}


# ── get_preset returns a copy ─────────────────────────────────────────────────

@pytest.mark.parametrize("dataset, mode", [("eegbci", "motor_imagery"), (None, "spectral")])
def test_get_preset_mutation_does_not_leak(dataset, mode):
    first = get_preset(dataset, mode)
    snapshot = repr(get_preset(dataset, mode))
    first["bandpass_low"] = -999
    for value in first.values():
        if isinstance(value, dict):
            value["__poisoned__"] = True     # nested, e.g. eegbci annotation_rename
    assert repr(get_preset(dataset, mode)) == snapshot


def test_eegbci_preset_has_nested_dict():
    # the deepcopy (not a shallow .copy()) only matters if nested dicts exist;
    # keep this true so the test above actually exercises it
    assert any(isinstance(v, dict) for v in DATASET_PRESETS["eegbci"].values())


# ── spectral visuals in every mode ────────────────────────────────────────────

def test_spectral_summary_computes_psd_when_not_given(synthetic_epochs, labels):
    psd, topo = spectral_summary(synthetic_epochs, labels, ["left_hand", "right_hand"],
                                 FREQUENCY_BANDS)
    assert psd is not None
    assert set(psd["curves"]) == {"left_hand", "right_hand"}
    assert len(psd["curves"]["left_hand"]) == len(psd["freqs"])
    assert psd["freqs"][0] >= 0.5 and psd["freqs"][-1] <= 45
    assert topo is None          # synthetic fixture has no montage -> no positions


def test_spectral_summary_topomaps_with_montage(synthetic_epochs, labels):
    epochs = synthetic_epochs.copy()
    # https://mne.tools/1.12/generated/mne.channels.make_standard_montage.html
    epochs.set_montage(mne.channels.make_standard_montage("standard_1020"))
    psd, topo = spectral_summary(epochs, labels, ["left_hand", "right_hand"], FREQUENCY_BANDS)
    assert psd is not None
    assert topo and "alpha" in topo
    assert topo["alpha"]["image"][:10]          # non-empty base64 PNG


def test_spectral_summary_never_raises(synthetic_epochs, labels, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("plotting exploded")

    monkeypatch.setattr(type(synthetic_epochs), "compute_psd", broken)
    assert spectral_summary(synthetic_epochs, labels, ["a", "b"], FREQUENCY_BANDS) == (None, None)


# ── preprocessing_applied tells the truth ─────────────────────────────────────

def test_describe_preprocessing_for_pre_epoched(synthetic_epochs):
    config = {"bandpass_low": 7.0, "bandpass_high": 30.0, "epoch_tmin": -0.2}
    applied = analyze_route.describe_preprocessing(synthetic_epochs, config,
                                                   synthetic_epochs, "moabb")
    assert applied["pre_epoched"] is True
    assert "bandpass_low" not in applied, "preset values that never ran must not be reported"
    assert applied["epoch_window_s"] == [synthetic_epochs.tmin, synthetic_epochs.tmax]
    assert applied["lowpass_hz"] == synthetic_epochs.info["lowpass"]


def test_describe_preprocessing_for_raw(synthetic_raw, synthetic_epochs):
    config = {"bandpass_low": 7.0, "_internal": 1}
    applied = analyze_route.describe_preprocessing(synthetic_raw, config,
                                                   synthetic_epochs, "upload")
    assert applied == {"bandpass_low": 7.0}


# ── band power is log-scaled so trees can split on it ─────────────────────────

def test_band_power_features_are_log_scaled(synthetic_epochs):
    from backend.core.features import extract_band_power_features
    X, extras = extract_band_power_features(synthetic_epochs)
    assert np.isfinite(X).all()
    # log10 of ~1e-12 V^2/Hz lands around -12; linear values would be ~1e-12
    assert X.min() < -5
    # sklearn trees ignore features with range <= 1e-7 (FEATURE_THRESHOLD)
    assert (X.max(axis=0) - X.min(axis=0)).min() > 1e-7
    assert extras["raw_psds"].max() < 1e-6, "raw_psds must stay linear for the plots"


def test_random_forest_learns_from_band_power(synthetic_epochs, labels):
    # The synthetic classes differ in C3/C4 power, which band power captures.
    # Before the log fix RF sat at exactly chance on this kind of input.
    from backend.core.features import extract_band_power_features
    from backend.core.classifiers import run_classifier
    X, _ = extract_band_power_features(synthetic_epochs)
    r = run_classifier("rf", "spectral", {}, X=X, y=labels, cv_folds=5)
    assert r["accuracy_mean"] > 0.8


def test_flat_channel_does_not_produce_inf(synthetic_epochs):
    from backend.core.features import extract_band_power_features
    epochs = synthetic_epochs.copy()
    epochs._data[:, 0, :] = 0.0         # one dead electrode
    X, _ = extract_band_power_features(epochs)
    assert np.isfinite(X).all()


# ── /analyze end to end (no HTTP, no disk) ────────────────────────────────────

def _request(mode="motor_imagery", selected=("lda",)):
    return AnalysisRequest(session_id="test", mode=mode,
                           classifiers={"selected": list(selected), "n_folds": 5})


def test_analyze_rejects_unready_session(monkeypatch):
    monkeypatch.setattr(analyze_route, "get_session",
                        lambda sid: {"is_ready": False, "source": "upload", "format": "csv"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(analyze_route.analyze(_request()))
    assert exc.value.status_code == 400
    assert "isn't ready" in exc.value.detail


def test_analyze_motor_imagery_returns_visuals_and_honest_preprocessing(
        monkeypatch, synthetic_epochs):
    session = {"is_ready": True, "source": "moabb", "dataset_name": "SyntheticSet",
               "subjects": [1],
               "capabilities": {"available_modes": ["basic", "spectral", "motor_imagery"]}}
    monkeypatch.setattr(analyze_route, "get_session", lambda sid: session)
    monkeypatch.setattr(analyze_route, "reload_data", lambda s: synthetic_epochs.copy())

    out = asyncio.run(analyze_route.analyze(_request("motor_imagery")))

    assert out["psd"] is not None, "PSD must be computed outside spectral mode too"
    assert set(out["psd"]["curves"]) == {"left_hand", "right_hand"}
    assert out["dataset_summary"]["classes"] == ["left_hand", "right_hand"]
    assert out["preprocessing_applied"]["pre_epoched"] is True
    assert out["classifiers"][0]["accuracy_mean"] > 0.8


def test_analyze_missing_is_ready_key_still_runs(monkeypatch, synthetic_epochs):
    # sessions written before is_ready existed must keep working
    session = {"source": "moabb", "capabilities": {}}
    monkeypatch.setattr(analyze_route, "get_session", lambda sid: session)
    monkeypatch.setattr(analyze_route, "reload_data", lambda s: synthetic_epochs.copy())
    out = asyncio.run(analyze_route.analyze(_request("motor_imagery")))
    assert out["classifiers"][0]["accuracy_mean"] is not None


# ── eegbci: classes follow the run variant (fetch audit, 13 Nov) ───────────────

@pytest.mark.parametrize("runs, expected", [([4, 8, 12], ["left_hand", "right_hand"]),
                                            ([6, 10, 14], ["hands", "feet"])])
def test_eegbci_classes_follow_run_variant(info, runs, expected):
    """
    eegbci's T1/T2 mean left/right hand on runs 4/8/12 but hands/feet on 6/10/14.
    The preset used to hardcode include_events=[left_hand, right_hand], so the
    hands/feet variant failed with "Could not find any of the events you specified".
    """
    from backend.core.pipeline import prepare_epochs
    rng = np.random.default_rng(3)
    n = 30
    raw = mne.io.RawArray(rng.standard_normal((len(info.ch_names), int(info["sfreq"]) * 6 * n)) * 1e-5, info)
    raw.set_annotations(mne.Annotations(onset=np.arange(n) * 6.0 + 1.0, duration=[4.1] * n,
                                        description=[["T0", "T1", "T2"][i % 3] for i in range(n)]))
    config = {**get_preset("eegbci", "motor_imagery"), "montage": None}
    epochs, used = prepare_epochs(raw, config, runs=runs)
    assert sorted(epochs.event_id) == sorted(expected), "T0 (rest) must be dropped"
    assert used["include_events"] == expected


# ── sleep staging: 30-s windows, channel types, spectral unlocked (13 Nov) ─────

def test_event_chunking_makes_one_trial_per_window(info):
    """
    One hypnogram annotation can cover 40 minutes. Without chunk_duration it became ONE
    trial (Sleep-EDF subject 1: 125 trials instead of ~1100).
    """
    from backend.core.pipeline import prepare_epochs
    sf = info["sfreq"]
    raw = mne.io.RawArray(np.random.default_rng(4).standard_normal((len(info.ch_names), int(sf * 600))) * 1e-5, info)
    # 300 s of "N2" then 300 s of "wake": 10 windows of 30 s each
    raw.set_annotations(mne.Annotations(onset=[0, 300], duration=[300, 300], description=["N2", "wake"]))
    base = {"epoch_tmin": 0.0, "epoch_tmax": 30.0 - 1.0 / sf, "baseline_enabled": False,
            "reference": "none", "notch_enabled": False}

    chunked, _ = prepare_epochs(raw, {**base, "event_chunk_duration": 30.0})
    unchunked, _ = prepare_epochs(raw, base)
    _, _, counts = get_labels_counts(chunked)
    assert len(unchunked) == 2
    assert counts == {"N2": 10, "wake": 9} or counts == {"N2": 10, "wake": 10}   # last window may hit the end


def get_labels_counts(epochs):
    from backend.core.pipeline import get_labels
    return get_labels(epochs)


def test_spectral_available_for_bipolar_eeg_names():
    """Sleep-EDF's EEG channels are 'Fpz-Cz' and 'Pz-Oz' — no standard 10-20 names, but band power works."""
    from backend.core.detector import detect_capabilities
    info = mne.create_info(["Fpz-Cz", "Pz-Oz", "EOG horizontal", "Temp rectal"], 100.0,
                           ch_types=["eeg", "eeg", "eog", "temperature"])
    raw = mne.io.RawArray(np.random.default_rng(5).standard_normal((4, 6000)) * 1e-5, info)
    raw.set_annotations(mne.Annotations([0, 30], [30, 30], ["N2", "wake"]))
    caps = detect_capabilities(raw)
    assert "spectral" in caps["available_modes"]
    assert caps["recommended_mode"] == "spectral"


def test_spectral_locked_without_any_eeg_channel():
    from backend.core.detector import get_available_modes
    assert "spectral" not in get_available_modes(["EMG", "ECG"], True, False, n_eeg_channels=0)


def test_sleep_annotations_without_every_stage():
    """Subject 4 has no 'Sleep stage 4'; renaming crashed with 'not present in data'."""
    from backend.fetchers.mne_fetcher import prepare_sleep_annotations
    a = mne.Annotations(
        onset=[0, 3600, 3900, 4500, 5400, 9000, 30000],
        duration=[3600, 300, 600, 900, 3600, 21000, 30],
        description=["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
                     "Sleep stage R", "Sleep stage W", "Movement time"],
    )
    out = prepare_sleep_annotations(a)
    assert set(out.description) == {"wake", "N1", "N2", "N3", "REM"}
    assert "Movement time" not in out.description
    # wake trimmed to 30 min before the first sleep stage (onset 3600 -> crop starts at 1800)
    assert out.onset.min() == pytest.approx(1800)
    assert out.duration[0] == pytest.approx(1800)


def test_sleep_stage_4_merges_into_n3():
    from backend.fetchers.mne_fetcher import prepare_sleep_annotations
    a = mne.Annotations([0, 30, 60], [30, 30, 30], ["Sleep stage 3", "Sleep stage 4", "Sleep stage 2"])
    assert list(prepare_sleep_annotations(a).description) == ["N3", "N3", "N2"]


# ── analysis_classes / n_eeg_channels for the UI (13 Nov) ──────────────────────

@pytest.mark.parametrize("runs, expected", [([4, 8, 12], ["left_hand", "right_hand"]),
                                            ([6, 10, 14], ["hands", "feet"]),
                                            (None, ["T0", "T1", "T2"])])   # unresolvable -> raw names
def test_analysis_class_names_for_eegbci(runs, expected):
    from backend.core.presets import analysis_class_names
    assert analysis_class_names("eegbci", "motor_imagery", ["T0", "T1", "T2"], runs) == expected


def test_analysis_class_names_passthrough_without_preset():
    from backend.core.presets import analysis_class_names
    assert analysis_class_names(None, "spectral", ["a", "b"]) == ["a", "b"]


def test_detector_reports_eeg_channel_count():
    from backend.core.detector import detect_capabilities
    info = mne.create_info(["Fpz-Cz", "Pz-Oz", "EOG", "Temp"], 100.0, ["eeg", "eeg", "eog", "temperature"])
    caps = detect_capabilities(mne.io.RawArray(np.zeros((4, 1000)), info))
    assert caps["n_channels"] == 4 and caps["n_eeg_channels"] == 2


# ── group 1 quick fixes (13 Nov) ───────────────────────────────────────────────

def test_basic_features_are_in_microvolts_so_trees_can_split(synthetic_epochs):
    """Variance in volts (~1e-10 V^2) sat below sklearn's 1e-7 tree split threshold."""
    from backend.core.features import extract_basic_features
    X, extras = extract_basic_features(synthetic_epochs)
    var_idx = [i for i, n in enumerate(extras["feature_names"]) if n.endswith("_var")]
    assert (X[:, var_idx].max(0) - X[:, var_idx].min(0)).min() > 1e-7


def test_ssvep_markers_become_frequencies():
    from backend.core.presets import analysis_class_names
    assert analysis_class_names("ssvep", "spectral", ["Stimulus/S155", "Stimulus/S255"]) == ["15hz", "12hz"]


def test_kiloword_median_split_into_two_classes(info):
    import pandas as pd
    from backend.fetchers.mne_fetcher import label_kiloword_by_concreteness
    n = 10
    md = pd.DataFrame({"WORD": [f"w{i}" for i in range(n)], "Concreteness": np.linspace(1.5, 7.0, n)})
    ev = np.column_stack([np.arange(n) * 100, np.zeros(n, int), np.arange(1, n + 1)])   # one event per word
    epochs = mne.EpochsArray(np.zeros((n, len(info.ch_names), 50)), info, events=ev,
                             event_id={f"w{i}": i + 1 for i in range(n)}, metadata=md)
    out = label_kiloword_by_concreteness(epochs)
    assert out.event_id == {"abstract": 1, "concrete": 2}
    assert (out.events[:, 2] == 1).sum() == 5 and (out.events[:, 2] == 2).sum() == 5


@pytest.mark.parametrize("task, available, default, expected", [
    ("motor_imagery", ["basic", "spectral", "motor_imagery"], "spectral", "motor_imagery"),   # BNCI2014_002
    ("ssvep", ["basic", "spectral", "motor_imagery"], "motor_imagery", "spectral"),          # SSVEP has C3/C4
    ("spectral", ["basic", "spectral"], "spectral", "spectral"),
    ("erp", ["basic", "spectral", "motor_imagery"], "motor_imagery", "motor_imagery"),      # no ERP mode yet
    ("motor_imagery", ["basic"], "basic", "basic"),                                          # unavailable -> keep
    (None, ["basic", "spectral"], "spectral", "spectral"),                                   # uploads
])
def test_recommend_for_task(task, available, default, expected):
    from backend.core.detector import recommend_for_task
    assert recommend_for_task(task, available, default) == expected


# ── ERP mode + balanced accuracy (session 2) ───────────────────────────────────

def test_erp_features_are_time_bins_in_microvolts(synthetic_epochs):
    from backend.core.features import extract_erp_features
    X, extras = extract_erp_features(synthetic_epochs, bin_ms=50)
    sfreq = synthetic_epochs.info["sfreq"]                      # 128 Hz, 1-s epochs
    per_bin = round(sfreq * 0.05)                                # 6 samples
    n_bins = len(synthetic_epochs.times) // per_bin
    assert extras["n_bins"] == n_bins
    assert X.shape == (len(synthetic_epochs), len(synthetic_epochs.ch_names) * n_bins)
    # first feature = mean of the first bin of the first channel, in µV
    expected = synthetic_epochs.get_data()[0, 0, :per_bin].mean() * 1e6
    assert X[0, 0] == pytest.approx(expected)
    assert extras["feature_names"][0].startswith("C3_")


def test_erp_mode_available_and_recommended_only_for_erp_tasks():
    from backend.core.detector import get_available_modes, get_recommended_mode, recommend_for_task
    modes = get_available_modes(["C3", "C4", "Cz", "Pz"], True, True, n_eeg_channels=4)
    assert "erp" in modes
    assert get_recommended_mode(modes, True) == "motor_imagery", "never the automatic default"
    assert recommend_for_task("erp", modes, "motor_imagery") == "erp"
    assert "erp" not in get_available_modes(["C3"], False, False, n_eeg_channels=1), "needs events"


def test_flankers_responses_become_correct_or_error():
    from backend.fetchers.mne_fetcher import label_flankers_responses
    a = mne.Annotations(
        onset=[1.0, 1.4, 5.0, 5.3, 9.0, 9.2, 20.0, 30.0],
        duration=[0] * 8,
        description=["stimulus/compatible/target_left", "response/left",      # correct
                     "stimulus/incompatible/target_right", "response/left",   # error
                     "stimulus/compatible/target_right", "response/right",    # correct
                     "response/left",                                         # 11 s after a stimulus: dropped
                     "stimulus/compatible/target_left"],
    )
    out = label_flankers_responses(a)
    assert list(out.description) == ["correct", "error", "correct"]
    assert list(out.onset) == [1.4, 5.3, 9.2]


def test_balanced_accuracy_exposes_majority_guessing():
    """A 90/10 split where the model can't separate the classes: plain accuracy looks fine."""
    from backend.core.classifiers import run_classifier
    rng = np.random.default_rng(7)
    y = np.array([1] * 90 + [2] * 10)
    X = rng.standard_normal((100, 5))                            # no signal at all
    r = run_classifier("lda", "basic", {}, X=X, y=y, cv_folds=5)
    assert r["accuracy_mean"] > 0.8
    assert r["balanced_accuracy_mean"] < 0.6
