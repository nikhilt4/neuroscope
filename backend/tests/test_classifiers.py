"""
Guards the three non-negotiables in core/classifiers.py:
  1. CSP and StandardScaler are refit inside every CV fold (no test-fold leakage)
  2. One classifier failing never takes the others down
  3. The result dict has the shape the frontend reads
"""
import numpy as np
import pytest
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler

from backend.core import classifiers
from backend.core.classifiers import run_all_classifiers, run_classifier

RESULT_KEYS = {
    "classifier", "category", "accuracy_mean", "accuracy_std", "fold_scores",
    "n_folds", "confusion_matrix", "confusion_labels",
    "balanced_accuracy_mean", "balanced_accuracy_std",
}


# ── 1. leakage ────────────────────────────────────────────────────────────────

class RecordingCSP(CSP):
    """
    CSP that records how many trials each fit() sees.

    No __init__ override on purpose: sklearn's clone() rebuilds estimators from
    get_params(), which introspects the class's __init__ signature. Inheriting
    CSP's keeps clone() working exactly as it does for the real class.
    https://scikit-learn.org/1.9/modules/generated/sklearn.base.clone.html
    """
    fit_sizes: list = []

    def fit(self, X, y):
        RecordingCSP.fit_sizes.append(len(X))
        return super().fit(X, y)


class RecordingScaler(StandardScaler):
    fit_sizes: list = []

    def fit(self, X, y=None, sample_weight=None):
        RecordingScaler.fit_sizes.append(len(X))
        return super().fit(X, y, sample_weight=sample_weight)


def test_csp_is_fit_once_per_fold_never_on_all_trials(monkeypatch, synthetic_epochs, labels):
    # run_classifier looks CSP up as a module global, so patching the name in
    # backend.core.classifiers swaps it for the recording subclass.
    RecordingCSP.fit_sizes = []
    monkeypatch.setattr(classifiers, "CSP", RecordingCSP)

    n = len(labels)
    result = run_classifier("lda", "motor_imagery", {}, y=labels,
                            epochs_obj=synthetic_epochs, cv_folds=5)

    assert len(RecordingCSP.fit_sizes) == 5, "CSP should be fit exactly once per fold"
    assert all(size < n for size in RecordingCSP.fit_sizes), (
        f"CSP saw all {n} trials in one fit — it was fit outside CV and leaks test data"
    )
    # 5-fold stratified on 60 trials -> every training split holds 48
    assert RecordingCSP.fit_sizes == [n * 4 // 5] * 5
    assert result["accuracy_mean"] > 0.8


def test_scaler_is_refit_per_fold_for_scale_sensitive_models(monkeypatch, labels):
    RecordingScaler.fit_sizes = []
    monkeypatch.setattr(classifiers, "StandardScaler", RecordingScaler)

    X = np.random.default_rng(0).standard_normal((len(labels), 10))
    run_classifier("svm", "basic", {}, X=X, y=labels, cv_folds=5)

    assert "svm" in classifiers.SCALE_SENSITIVE
    assert len(RecordingScaler.fit_sizes) == 5
    assert all(size < len(labels) for size in RecordingScaler.fit_sizes)


def test_lda_without_scale_sensitivity_gets_no_scaler(monkeypatch, labels):
    RecordingScaler.fit_sizes = []
    monkeypatch.setattr(classifiers, "StandardScaler", RecordingScaler)

    X = np.random.default_rng(0).standard_normal((len(labels), 10))
    run_classifier("lda", "basic", {}, X=X, y=labels, cv_folds=5)

    assert RecordingScaler.fit_sizes == []


# ── 2. failure isolation ──────────────────────────────────────────────────────

def test_unknown_classifier_does_not_block_others(synthetic_epochs, labels):
    results = run_all_classifiers(["lda", "not_a_model", "riemann_ts"], "motor_imagery", {},
                                  y=labels, epochs_obj=synthetic_epochs, cv_folds=5)

    by_name = {r["classifier"]: r for r in results}
    assert list(by_name) == ["lda", "not_a_model", "riemann_ts"], "order must be preserved"
    assert by_name["not_a_model"]["accuracy_mean"] is None
    assert "Unknown classifier" in by_name["not_a_model"]["error"]
    assert by_name["lda"]["accuracy_mean"] > 0.8
    assert by_name["riemann_ts"]["accuracy_mean"] > 0.8


def test_crash_inside_a_builder_is_isolated(monkeypatch, synthetic_epochs, labels):
    def exploding_builder(config):
        raise RuntimeError("boom")

    monkeypatch.setitem(classifiers.FEATURE_CLASSIFIERS, "gnb", exploding_builder)

    results = run_all_classifiers(["gnb", "lda"], "motor_imagery", {},
                                  y=labels, epochs_obj=synthetic_epochs, cv_folds=5)

    gnb, lda = results
    assert gnb["accuracy_mean"] is None
    assert "boom" in gnb["error"]
    assert lda["accuracy_mean"] > 0.8


def test_feature_classifier_without_features_reports_clean_error(labels):
    # spectral/basic modes need pre-extracted X; forgetting it should be an
    # actionable message, not a numpy traceback
    results = run_all_classifiers(["lda"], "spectral", {}, X=None, y=labels, cv_folds=5)
    assert results[0]["accuracy_mean"] is None
    assert "pre-extracted features" in results[0]["error"]


# ── 3. result shape ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, category", [("lda", "feature"), ("riemann_ts", "epoch")])
def test_result_shape(name, category, synthetic_epochs, labels):
    r = run_classifier(name, "motor_imagery", {}, y=labels,
                       epochs_obj=synthetic_epochs, cv_folds=5)

    assert set(r) == RESULT_KEYS
    assert r["category"] == category
    assert r["n_folds"] == 5 and len(r["fold_scores"]) == 5
    assert np.isclose(r["accuracy_mean"], np.mean(r["fold_scores"]))

    cm = np.array(r["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(labels), "out-of-fold predictions must cover every trial once"
    assert r["confusion_labels"] == [1, 2]


def test_fgmdm_reports_too_few_channels_clearly():
    """Sleep-EDF: 2 EEG channels, 5 stages. FgMDM's internal LDA needs n_classes-1 = 4
    tangent-space features but 2 channels give only 3 — the raw sklearn error was opaque."""
    import mne
    rng = np.random.default_rng(6)
    info = mne.create_info(["Fpz-Cz", "Pz-Oz"], 100.0, "eeg")
    y = np.repeat([1, 2, 3, 4, 5], 12)
    epochs = mne.EpochsArray(rng.standard_normal((60, 2, 100)) * 1e-5, info,
                             events=np.column_stack([np.arange(60) * 100, np.zeros(60, int), y]))
    (r,) = run_all_classifiers(["riemann_fgmdm"], "spectral", {}, y=y, epochs_obj=epochs, cv_folds=3)
    assert r["accuracy_mean"] is None
    assert "2 channels with 5 classes" in r["error"]
