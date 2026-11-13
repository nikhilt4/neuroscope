"""
End-to-end on real data: BNCI2014_001 (BCI Competition IV 2a), SUBJECT 1.

Follows the same path as POST /analyze (reload_data -> get_preset ->
prepare_epochs -> get_labels -> run_all_classifiers) without HTTP.

Reference numbers are this project's own verified run, recorded in
NEUROSCOPE_CONTEXT.md section 4 (riemann_ts 83.3%, lda 73.4%, 5-fold, defaults).
They sit inside the single-subject range reported for this dataset, but they are
a regression baseline for THIS code, not a literature benchmark. A failure here
means something changed — check which subject was fetched before blaming code.

Must run from the repo root: MOABB caches to <cwd>/C-/Users/... on Windows
(see NEUROSCOPE_CONTEXT.md section 6), and that cache lives in the repo root.

Run with:  python -m pytest -m slow -s
"""
import pytest

from backend.core.classifiers import run_all_classifiers
from backend.core.pipeline import get_labels, prepare_epochs, reload_data
from backend.core.presets import get_preset

pytestmark = pytest.mark.slow

SUBJECT = 1
TOLERANCE = 0.03     # StratifiedKFold(random_state=42) is deterministic; this absorbs
                     # small numeric drift across library versions only


@pytest.fixture(scope="module")
def bnci_subject1():
    session = {"source": "moabb", "dataset_name": "BNCI2014_001", "subjects": [SUBJECT]}
    data = reload_data(session)
    config = get_preset("BNCI2014_001", "motor_imagery")
    epochs, config = prepare_epochs(data, config)
    y, class_names, counts = get_labels(epochs)
    print(f"\n[BNCI2014_001 subject {SUBJECT}] {len(epochs)} epochs, "
          f"{len(epochs.ch_names)} channels, event_id={epochs.event_id}, "
          f"class_names={class_names}, counts={counts}")
    return epochs, y, class_names


def test_shape_matches_known_dataset(bnci_subject1):
    epochs, y, _ = bnci_subject1
    assert len(epochs) == 576            # 2 sessions x 6 runs x 48 trials
    assert len(epochs.ch_names) == 22    # EEG only; MOABB drops the 3 EOG
    assert len(set(y.tolist())) == 4


def test_class_names_are_real_labels(bnci_subject1):
    # Was '1','2','3','4' until moabb_fetcher.restore_class_names (13 Nov 2026).
    # Ordered by event code: 1 left_hand, 2 right_hand, 3 feet, 4 tongue.
    _, _, class_names = bnci_subject1
    assert class_names == ["left_hand", "right_hand", "feet", "tongue"]


@pytest.mark.parametrize("name, expected", [("riemann_ts", 0.833), ("lda", 0.734)])
def test_accuracy_matches_verified_baseline(bnci_subject1, name, expected):
    epochs, y, _ = bnci_subject1
    (result,) = run_all_classifiers([name], "motor_imagery", {}, y=y, epochs_obj=epochs,
                                    csp_config={"n_components": 4, "reg": None}, cv_folds=5)
    acc = result["accuracy_mean"]
    print(f"[subject {SUBJECT}] {name}: {acc:.3f} ± {result['accuracy_std']:.3f}")
    assert acc is not None, result.get("error")
    assert acc == pytest.approx(expected, abs=TOLERANCE)


# ── fetch-audit regressions (13 Nov) ───────────────────────────────────────────

def test_braindecode_route_uses_every_run_with_real_names():
    """braindecode used only datasets[0] (48 of 576 trials), labelled 1-4, and scored 0.167."""
    from backend.fetchers.braindecode_fetcher import fetch_braindecode_dataset
    raw = fetch_braindecode_dataset("BNCI2014_001", [SUBJECT])
    epochs, _ = prepare_epochs(raw, get_preset("BNCI2014_001", "motor_imagery"))
    y, names, counts = get_labels(epochs)
    assert len(epochs) == 576
    assert sorted(names) == ["feet", "left_hand", "right_hand", "tongue"]
    (r,) = run_all_classifiers(["lda"], "motor_imagery", {}, y=y, epochs_obj=epochs,
                               csp_config={"n_components": 4, "reg": None}, cv_folds=5)
    print(f"[braindecode subject {SUBJECT}] lda: {r['accuracy_mean']:.3f}")
    # same data as the MOABB route (0.734). UNVERIFIED: the 0.008 gap is probably filter
    # design (our FIR vs MOABB's filtering) — not yet checked.
    assert r["accuracy_mean"] == pytest.approx(0.726, abs=TOLERANCE)


def test_eegbci_hands_feet_variant_runs():
    from backend.fetchers.mne_fetcher import fetch_mne_dataset
    raw = fetch_mne_dataset("eegbci", [SUBJECT], [6, 10, 14])
    epochs, _ = prepare_epochs(raw, get_preset("eegbci", "motor_imagery"), runs=[6, 10, 14])
    _, names, _ = get_labels(epochs)
    assert names == ["hands", "feet"] or sorted(names) == ["feet", "hands"]


def test_sleep_physionet_is_staged_in_30s_windows():
    """
    Sleep-EDF subject 1 used to give 125 trials (one per annotation), 7 'eeg' channels
    including temperature, and basic mode only (LDA 0.576 vs chance 0.32).
    """
    from backend.core.detector import detect_capabilities
    from backend.core.features import extract_features
    from backend.fetchers.mne_fetcher import fetch_mne_dataset
    raw = fetch_mne_dataset("sleep_physionet", [SUBJECT])
    caps = detect_capabilities(raw)
    assert caps["recommended_mode"] == "spectral"
    assert sorted(caps["event_classes"]) == ["N1", "N2", "N3", "REM", "wake"]

    epochs, _ = prepare_epochs(raw, get_preset("sleep_physionet", "spectral"))
    y, names, counts = get_labels(epochs)
    assert epochs.ch_names == ["Fpz-Cz", "Pz-Oz"], "only the two EEG channels go into features"
    assert len(epochs) == 1103
    assert counts["N3"] == 105          # stage 3 (96) + stage 4 (9) merged
    X, _ = extract_features(epochs, "spectral", labels=y)
    from backend.core.classifiers import run_classifier
    r = run_classifier("lda", "spectral", {}, X=X, y=y)
    print(f"[sleep_physionet subject {SUBJECT}] lda: {r['accuracy_mean']:.3f} (chance {max(counts.values())/len(y):.3f})")
    assert r["accuracy_mean"] == pytest.approx(0.845, abs=TOLERANCE)
