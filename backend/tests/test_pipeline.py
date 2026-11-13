"""
prepare_epochs is the single door into preprocessing. Pre-epoched sources
(MOABB, limo, kiloword, FIF epochs) must skip preprocess_raw — MOABB has
already filtered and epoched, and filtering again would be wrong.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.core import pipeline
from backend.core.pipeline import get_labels, prepare_epochs


def test_pre_epoched_data_skips_preprocess_raw(monkeypatch, synthetic_epochs):
    spy = MagicMock(side_effect=AssertionError("preprocess_raw must not run on epochs"))
    # pipeline.py does `from backend.core.preprocessor import preprocess_raw`,
    # so the name to patch is the one bound inside backend.core.pipeline.
    monkeypatch.setattr(pipeline, "preprocess_raw", spy)

    config = {"bandpass_low": 8.0, "bandpass_high": 30.0}
    epochs, returned_config = prepare_epochs(synthetic_epochs, config)

    spy.assert_not_called()
    assert epochs is synthetic_epochs
    assert returned_config == config


def test_raw_data_goes_through_preprocess_raw(monkeypatch, synthetic_raw):
    sentinel = object()
    spy = MagicMock(return_value=sentinel)
    monkeypatch.setattr(pipeline, "preprocess_raw", spy)

    config = {"bandpass_low": 8.0}
    epochs, _ = prepare_epochs(synthetic_raw, config)

    spy.assert_called_once_with(synthetic_raw, config)
    assert epochs is sentinel


def test_pre_epoched_data_still_honours_train_crop(synthetic_epochs):
    epochs, _ = prepare_epochs(synthetic_epochs.copy(), {"epoch_train_crop": (0.2, 0.8)})
    assert epochs.tmin == pytest.approx(0.2, abs=1 / epochs.info["sfreq"])
    assert epochs.tmax == pytest.approx(0.8, abs=1 / epochs.info["sfreq"])


def test_train_crop_outside_epoch_window_is_a_400(synthetic_epochs):
    with pytest.raises(HTTPException) as exc:
        prepare_epochs(synthetic_epochs, {"epoch_train_crop": (0.5, 5.0)})
    assert exc.value.status_code == 400


def test_get_labels_returns_real_class_names(synthetic_epochs):
    y, class_names, counts = get_labels(synthetic_epochs)

    assert class_names == ["left_hand", "right_hand"]
    assert counts == {"left_hand": 30, "right_hand": 30}
    assert sorted(set(y.tolist())) == [1, 2]
    # must be plain Python types for JSON serialisation
    assert all(type(n) is str for n in class_names)
    assert all(type(v) is int for v in counts.values())
