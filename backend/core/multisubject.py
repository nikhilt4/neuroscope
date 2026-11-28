"""
Multi-subject evaluation (NEUROSCOPE_CONTEXT.md section 8).

within_subject — train and test separately on each subject (each gets its own CSP
    filters and model), then report the mean ± std across subjects. This is how BCI
    papers report results ("72% ± 11% over 9 subjects"), and what a single-subject
    number can't express: subject 1 vs subject 5 on BNCI2014_001 is ~83% vs ~50%.

cross_subject — pool every subject's trials and leave one subject out per fold
    (LeaveOneGroupOut with subject as the group). Tests whether a decoder works on
    someone it has never seen, without calibration. Plain StratifiedKFold on pooled
    data would leak: the same person would sit in train and test.

Both reuse the single-subject pipeline unchanged (reload_data -> prepare_epochs ->
features -> run_all_classifiers); they only change which data goes in and how the
folds are drawn.
"""
from typing import Callable, Optional

import mne
import numpy as np
from fastapi import HTTPException

from backend.core.classifiers import run_all_classifiers
from backend.core.features import FREQUENCY_BANDS, extract_features
from backend.core.pipeline import apply_class_names, get_labels, prepare_epochs, reload_data
from backend.core.presets import get_preset
from backend.core.visualizer import spectral_summary

MULTI_SOURCES = {"mne", "moabb", "braindecode"}   # sources where "subject N" means something


def _validate(request, session) -> list:
    if request.evaluation not in ("within_subject", "cross_subject"):
        raise HTTPException(status_code=400, detail=f"Unknown evaluation: {request.evaluation}")
    if session.get("source") not in MULTI_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Multi-subject evaluation needs a public dataset with several subjects. "
                   "An uploaded file is a single recording.")
    subjects = sorted({int(s) for s in (request.subjects or [])})
    if len(subjects) < 2:
        raise HTTPException(status_code=400,
                            detail="Pick at least two subjects for a multi-subject evaluation.")
    return subjects


def _load_subject(request, session, subject: int):
    """One subject's epochs, prepared exactly as a single-subject run would prepare them."""
    sess = {**session, "subjects": [subject]}
    data = reload_data(sess)
    preset = get_preset(session.get("dataset_name"), request.mode, task=session.get("task"))
    overrides = request.preprocessing.model_dump(exclude_unset=True) if request.preprocessing else {}
    epochs, config = prepare_epochs(data, {**preset, **overrides}, runs=session.get("runs"))
    return data, apply_class_names(epochs, config.get("class_names")), config


def _code_to_name(epochs: mne.BaseEpochs) -> dict:
    return {int(v): str(k) for k, v in epochs.event_id.items()}


def _confusion_in_names(result: dict, epochs: mne.BaseEpochs, y: np.ndarray, classes: list):
    """
    Re-index one subject's confusion matrix onto the shared class list, so matrices from
    different subjects can be summed. Deep models report LabelEncoder indices (0..k-1, in
    sorted-code order) rather than event codes — map those back first.
    """
    if not result.get("confusion_matrix"):
        return None
    labels = result["confusion_labels"]
    if result.get("category") == "deep":
        labels = sorted(set(y.tolist()))
    names = [_code_to_name(epochs).get(int(c), str(c)) for c in labels]
    out = np.zeros((len(classes), len(classes)), dtype=int)
    idx = [classes.index(n) for n in names]
    for i, row in enumerate(result["confusion_matrix"]):
        for j, v in enumerate(row):
            out[idx[i], idx[j]] += v
    return out


def run_within_subject(request, session, progress: Callable[[str], None]) -> dict:
    subjects = _validate(request, session)
    clf = request.classifiers
    per_subject, trials, counts_total = {}, {}, {}
    classes: list = []
    first = None   # (data, epochs, config, y, names, extras) of the first subject, for visuals
    confusions: dict = {}

    for i, s in enumerate(subjects, start=1):
        tag = f"Subject {s} ({i} of {len(subjects)})"
        progress(f"{tag}: loading")
        data, epochs, config = _load_subject(request, session, s)
        y, names, counts = get_labels(epochs)
        for n in names:
            if n not in classes:
                classes.append(n)
        for n, c in counts.items():
            counts_total[n] = counts_total.get(n, 0) + c
        trials[s] = int(len(epochs))

        X, extras = (None, {})
        if request.mode != "motor_imagery":
            progress(f"{tag}: extracting features")
            X, extras = extract_features(epochs, request.mode, labels=y)

        results = run_all_classifiers(
            selected_classifiers=clf.selected, mode=request.mode, configs=dict(clf.configs),
            X=X, y=y, epochs_obj=epochs,
            csp_config={"n_components": clf.csp_components, "reg": clf.csp_reg},
            cv_folds=clf.n_folds, progress=lambda m, tag=tag: progress(f"{tag}: {m}"),
        )
        per_subject[s] = results
        for r in results:
            cm = _confusion_in_names(r, epochs, y, classes) if r.get("accuracy_mean") is not None else None
            if cm is not None:
                confusions.setdefault(r["classifier"], []).append(cm)
        if first is None:
            first = (data, epochs, config, y, names, extras)
        else:
            del data, epochs   # one subject in memory at a time

    progress("Computing spectra and scalp maps")
    data0, ep0, config0, y0, names0, extras0 = first
    psd, topomaps = spectral_summary(ep0, y0, names0, FREQUENCY_BANDS,
                                     psds=extras0.get("raw_psds"), freqs=extras0.get("freqs"))

    rows = []
    for name in clf.selected:
        accs = [(s, r) for s in subjects for r in per_subject[s] if r["classifier"] == name]
        ok = [(s, r) for s, r in accs if r.get("accuracy_mean") is not None]
        errors = {s: r.get("error") for s, r in accs if r.get("accuracy_mean") is None}
        if not ok:
            rows.append({"classifier": name, "accuracy_mean": None,
                         "error": next(iter(errors.values()), "Did not run"), "subject_errors": errors})
            continue
        vals = np.array([r["accuracy_mean"] for _, r in ok])
        bal = np.array([r.get("balanced_accuracy_mean", np.nan) for _, r in ok])
        cms = confusions.get(name, [])
        size = len(classes)
        summed = sum((np.pad(c, ((0, size - c.shape[0]), (0, size - c.shape[1]))) for c in cms),
                     np.zeros((size, size), dtype=int))
        rows.append({
            "classifier": name,
            "category": ok[0][1]["category"],
            "accuracy_mean": float(vals.mean()),
            "accuracy_std": float(vals.std()),
            "balanced_accuracy_mean": float(np.nanmean(bal)),
            "balanced_accuracy_std": float(np.nanstd(bal)),
            # one "fold" per subject, so the shared results UI shows the spread across people
            "fold_scores": [float(v) for v in vals],
            "fold_groups": [int(s) for s, _ in ok],
            "n_folds": len(ok),
            "per_subject": [{"subject": int(s), "accuracy": float(r["accuracy_mean"]),
                             "accuracy_std": float(r["accuracy_std"])} for s, r in ok],
            "subject_errors": errors,
            "confusion_matrix": summed.tolist(),
            "confusion_labels": list(range(size)),
        })
    return _assemble(request, session, subjects, rows, classes, counts_total, trials,
                     data0, ep0, config0, psd, topomaps, extras0,
                     visuals_note=f"Spectra and scalp maps show subject {subjects[0]}.")


def run_cross_subject(request, session, progress: Callable[[str], None]) -> dict:
    subjects = _validate(request, session)
    clf = request.classifiers
    epochs_list, trials = [], {}
    first = None
    for i, s in enumerate(subjects, start=1):
        progress(f"Subject {s} ({i} of {len(subjects)}): loading")
        data, epochs, config = _load_subject(request, session, s)
        epochs_list.append(epochs)
        trials[s] = int(len(epochs))
        if first is None:
            first = (data, config)

    # Give every subject the same name -> code mapping before pooling; concatenate_epochs
    # needs consistent event_id, and codes can differ between subjects even when names match.
    # https://mne.tools/1.12/generated/mne.concatenate_epochs.html
    classes = []
    for ep in epochs_list:
        for name in ep.event_id:
            if name not in classes:
                classes.append(name)
    shared = {name: i + 1 for i, name in enumerate(classes)}
    for ep in epochs_list:
        old = _code_to_name(ep)
        ep.events[:, 2] = [shared[old[int(c)]] for c in ep.events[:, 2]]
        ep.event_id = {n: shared[n] for n in ep.event_id}

    progress(f"Pooling {sum(trials.values())} trials from {len(subjects)} subjects")
    pooled = mne.concatenate_epochs(epochs_list, add_offset=True, on_mismatch="warn", verbose=False)
    groups = np.concatenate([[s] * trials[s] for s in subjects])
    del epochs_list
    y, names, counts = get_labels(pooled)

    X, extras = (None, {})
    if request.mode != "motor_imagery":
        progress("Extracting features")
        X, extras = extract_features(pooled, request.mode, labels=y)

    progress("Computing spectra and scalp maps")
    psd, topomaps = spectral_summary(pooled, y, names, FREQUENCY_BANDS,
                                     psds=extras.get("raw_psds"), freqs=extras.get("freqs"))

    results = run_all_classifiers(
        selected_classifiers=clf.selected, mode=request.mode, configs=dict(clf.configs),
        X=X, y=y, epochs_obj=pooled,
        csp_config={"n_components": clf.csp_components, "reg": clf.csp_reg},
        cv_folds=clf.n_folds, progress=progress, groups=groups,
    )
    for r in results:
        if r.get("accuracy_mean") is not None:
            r["per_subject"] = [{"subject": int(g), "accuracy": float(a)}
                                for g, a in zip(r["fold_groups"], r["fold_scores"])]
    data0, config0 = first
    return _assemble(request, session, subjects, results, names, counts, trials,
                     data0, pooled, config0, psd, topomaps, extras,
                     visuals_note=f"Spectra and scalp maps pool all {len(subjects)} subjects.")


def _assemble(request, session, subjects, results, classes, counts, trials,
              data, epochs, config, psd, topomaps, extras, visuals_note) -> dict:
    from backend.routes.analyze import describe_preprocessing   # avoid a circular import
    ok = [r for r in results if r.get("accuracy_mean") is not None]
    best = max(ok, key=lambda r: r["accuracy_mean"]) if ok else None
    return {
        "session_id": request.session_id,
        "mode": request.mode,
        "evaluation": request.evaluation,
        "subjects": subjects,
        "trials_per_subject": {str(k): v for k, v in trials.items()},
        "dataset_summary": {
            "source": session.get("source"),
            "dataset": session.get("dataset_name"),
            "subjects": subjects,
            "runs": session.get("runs"),
            "n_epochs": int(sum(trials.values())),
            "n_channels": int(len(epochs.ch_names)),
            "sampling_rate": float(epochs.info["sfreq"]),
            "epoch_duration": float(epochs.tmax - epochs.tmin),
            "classes": classes,
            "class_counts": counts,
        },
        "preprocessing_applied": describe_preprocessing(data, config, epochs, session.get("source")),
        "classifiers": results,
        "best_classifier": best["classifier"] if best else None,
        "feature_info": {k: v for k, v in extras.items() if k in ("feature_names", "band_names", "n_components")},
        "psd": psd,
        "topomaps": topomaps,
        "visuals_note": visuals_note,
    }
