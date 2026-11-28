"""
User-chosen class names (PreprocessingConfig.class_names → pipeline.apply_class_names).
Renaming happens after preprocessing, on the epochs, so it must change what results
call each class without changing which trials are which.
"""
import mne
import numpy as np
import pytest
from fastapi import HTTPException

from backend.core.pipeline import apply_class_names, get_labels

mne.set_log_level("WARNING")


@pytest.fixture
def epochs():
    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    events = np.array([[i * 100, 0, 1 + (i % 2)] for i in range(6)])
    data = np.random.default_rng(0).standard_normal((6, 2, 50)) * 1e-6
    return mne.EpochsArray(data, info, events=events, event_id={"Stimulus/S  1": 1, "Stimulus/S  2": 2})


def test_renames_classes_and_keeps_trial_labels(epochs):
    out = apply_class_names(epochs, {"Stimulus/S  1": "target", "Stimulus/S  2": "non-target"})
    y_before, _, _ = get_labels(epochs)
    y_after, names, counts = get_labels(out)
    assert names == ["target", "non-target"]
    assert counts == {"target": 3, "non-target": 3}
    assert (y_before == y_after).all()          # same trials, same codes — only names change


def test_partial_unknown_and_blank_names(epochs):
    out = apply_class_names(epochs, {"Stimulus/S  1": "target", "Stimulus/S  2": "  ", "not-a-class": "x"})
    assert set(out.event_id) == {"target", "Stimulus/S  2"}


def test_does_not_mutate_the_callers_epochs(epochs):
    apply_class_names(epochs, {"Stimulus/S  1": "target"})
    assert set(epochs.event_id) == {"Stimulus/S  1", "Stimulus/S  2"}


def test_no_mapping_returns_the_same_object(epochs):
    assert apply_class_names(epochs, None) is epochs
    assert apply_class_names(epochs, {}) is epochs


def test_two_classes_with_one_name_are_refused(epochs):
    # merging two classes under one name would silently change the task
    with pytest.raises(HTTPException) as err:
        apply_class_names(epochs, {"Stimulus/S  1": "same", "Stimulus/S  2": "same"})
    assert err.value.status_code == 400
    assert "same" in err.value.detail


def test_renaming_onto_an_existing_name_is_refused(epochs):
    with pytest.raises(HTTPException):
        apply_class_names(epochs, {"Stimulus/S  1": "Stimulus/S  2"})
