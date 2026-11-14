"""
Multi-subject evaluation (NEUROSCOPE_CONTEXT.md section 8).

The non-negotiable: in cross-subject mode the held-out subject must never reach
training — not the classifier, and not CSP (which is supervised, so fitting it on the
test subject would leak just as badly).
"""
import mne
import numpy as np
import pytest
from fastapi import HTTPException

from backend.core import classifiers, multisubject
from backend.core.classifiers import run_classifier
from backend.models import AnalysisRequest

from .conftest import CH_NAMES, EVENT_ID, SFREQ, _make_trials
from .test_classifiers import RecordingCSP


def _subject_epochs(seed: int) -> mne.EpochsArray:
    data, codes = _make_trials(np.random.default_rng(seed))
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    events = np.column_stack([np.arange(len(codes)) * 200, np.zeros(len(codes), int), codes])
    return mne.EpochsArray(data, info, events=events, event_id=EVENT_ID)


@pytest.fixture
def three_subjects(monkeypatch):
    """reload_data returns a different synthetic subject (60 trials) per subject id."""
    monkeypatch.setattr(multisubject, "reload_data", lambda sess: _subject_epochs(100 + sess["subjects"][0]))
    return {"source": "moabb", "dataset_name": "SyntheticSet", "subjects": [1], "is_ready": True}


def _req(evaluation, subjects=(1, 2, 3), selected=("lda",)):
    return AnalysisRequest(session_id="t", mode="motor_imagery", evaluation=evaluation,
                           subjects=list(subjects), classifiers={"selected": list(selected), "n_folds": 3})


# ── leave-one-subject-out folds ───────────────────────────────────────────────

def test_groups_switch_to_leave_one_subject_out(monkeypatch):
    RecordingCSP.fit_sizes = []
    monkeypatch.setattr(classifiers, "CSP", RecordingCSP)
    pooled = mne.concatenate_epochs([_subject_epochs(s) for s in (1, 2, 3)], verbose=False)
    y = pooled.events[:, 2]
    groups = np.repeat([1, 2, 3], 60)

    r = run_classifier("lda", "motor_imagery", {}, y=y, epochs_obj=pooled, groups=groups)

    assert r["n_folds"] == 3 and r["fold_groups"] == [1, 2, 3]
    # each CSP fit sees exactly the two training subjects (120 trials), never all 180
    assert RecordingCSP.fit_sizes == [120, 120, 120]
    assert r["accuracy_mean"] > 0.8   # the synthetic signal is identical across "people"


# ── within-subject ────────────────────────────────────────────────────────────

def test_within_subject_aggregates_per_subject_results(three_subjects):
    out = multisubject.run_within_subject(_req("within_subject"), three_subjects, lambda m: None)

    (lda,) = out["classifiers"]
    accs = [p["accuracy"] for p in lda["per_subject"]]
    assert [p["subject"] for p in lda["per_subject"]] == [1, 2, 3]
    assert lda["accuracy_mean"] == pytest.approx(np.mean(accs))
    assert lda["accuracy_std"] == pytest.approx(np.std(accs))
    assert np.array(lda["confusion_matrix"]).sum() == 180, "summed over all subjects' trials"
    assert out["dataset_summary"]["n_epochs"] == 180
    assert out["evaluation"] == "within_subject"
    assert out["trials_per_subject"] == {"1": 60, "2": 60, "3": 60}


# ── cross-subject ─────────────────────────────────────────────────────────────

def test_cross_subject_pools_and_holds_out_each_subject(three_subjects):
    out = multisubject.run_cross_subject(_req("cross_subject"), three_subjects, lambda m: None)
    (lda,) = out["classifiers"]
    assert lda["fold_groups"] == [1, 2, 3]
    assert [p["subject"] for p in lda["per_subject"]] == [1, 2, 3]
    assert np.array(lda["confusion_matrix"]).sum() == 180
    assert out["dataset_summary"]["classes"] == ["left_hand", "right_hand"]


def test_progress_names_the_subject(three_subjects):
    seen = []
    multisubject.run_within_subject(_req("within_subject"), three_subjects, seen.append)
    assert any(m.startswith("Subject 2 (2 of 3)") for m in seen)


# ── validation ────────────────────────────────────────────────────────────────

def test_upload_sessions_are_refused(three_subjects):
    with pytest.raises(HTTPException) as exc:
        multisubject.run_within_subject(_req("within_subject"), {**three_subjects, "source": "upload"}, print)
    assert "public dataset" in exc.value.detail


def test_needs_two_subjects(three_subjects):
    with pytest.raises(HTTPException) as exc:
        multisubject.run_cross_subject(_req("cross_subject", subjects=(1,)), three_subjects, print)
    assert "at least two subjects" in exc.value.detail


# ── real data ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("evaluation", ["within_subject", "cross_subject"])
def test_bnci2014_001_three_cached_subjects(evaluation):
    """BCI IV 2a subjects 1, 4, 5 (the ones already cached). Prints per-subject numbers."""
    session = {"source": "moabb", "dataset_name": "BNCI2014_001", "subjects": [1], "is_ready": True}
    req = AnalysisRequest(session_id="t", mode="motor_imagery", evaluation=evaluation, subjects=[1, 4, 5],
                          classifiers={"selected": ["riemann_ts", "lda"], "n_folds": 5})
    run = multisubject.run_cross_subject if evaluation == "cross_subject" else multisubject.run_within_subject
    out = run(req, session, lambda m: None)
    print()
    for r in out["classifiers"]:
        per = ", ".join(f"s{p['subject']}={p['accuracy']:.3f}" for p in r["per_subject"])
        print(f"[BNCI2014_001 {evaluation}] {r['classifier']}: {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} ({per})")
    ts = next(r for r in out["classifiers"] if r["classifier"] == "riemann_ts")
    if evaluation == "within_subject":
        # subject 1's within-subject score must match the single-subject baseline exactly
        assert ts["per_subject"][0]["accuracy"] == pytest.approx(0.833, abs=0.005)
    assert out["dataset_summary"]["n_epochs"] == 3 * 576
