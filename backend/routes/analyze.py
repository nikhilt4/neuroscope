import mne
from fastapi import APIRouter, HTTPException

from backend.models import AnalysisRequest
from backend.utils.session_registry import get_session
from backend.core.presets import get_preset
from backend.core.pipeline import reload_data, prepare_epochs, get_labels
from backend.core.features import extract_features, FREQUENCY_BANDS
from backend.core.classifiers import run_all_classifiers
from backend.core.visualizer import spectral_summary

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Web-demo guard: 300 epochs x 5 folds is minutes of training and would hold
# a worker hostage. Power users can still raise it via the API. (backlog 17)
MAX_EEGNET_EPOCHS = 50


def describe_preprocessing(data, config: dict, epochs: mne.BaseEpochs, source: str) -> dict:
    """
    What preprocessing ACTUALLY ran, for the results page and the exported script.

    Pre-epoched sources (MOABB paradigms, limo, kiloword, FIF epochs) skip
    preprocess_raw entirely, so reporting the merged preset for them would
    describe filters that never touched the data. Instead, report the
    filter edges and window MNE recorded on the epochs themselves.
    """
    if not isinstance(data, mne.BaseEpochs):
        return {k: v for k, v in config.items() if not k.startswith("_")}

    applied = {
        "pre_epoched": True,
        "source": source,
        "note": "Filtering and epoching were done before NeuroScope received the data "
                "(by the MOABB paradigm or the file's author); NeuroScope did not re-filter.",
        # info["highpass"/"lowpass"] are the filter edges MNE recorded on the data
        # https://mne.tools/1.12/generated/mne.Info.html
        "highpass_hz": float(epochs.info["highpass"]),
        "lowpass_hz": float(epochs.info["lowpass"]),
        "epoch_window_s": [float(epochs.tmin), float(epochs.tmax)],
    }
    if config.get("epoch_train_crop"):
        applied["epoch_train_crop"] = config["epoch_train_crop"]   # this one DID run
    return applied


@router.post("")
async def analyze(request: AnalysisRequest):
    session = get_session(request.session_id)

    # Explicit `is False`: sessions written before is_ready existed have no key
    # and were always analysable, so a missing key must not block them.
    if session.get("is_ready") is False:
        raise HTTPException(
            status_code=400,
            detail="This dataset isn't ready to analyse yet. Finish the setup step "
                   "(column mapping, variable mapping, BIDS selection or event setup) first."
        )

    available = session.get("capabilities", {}).get("available_modes", [])
    if available and request.mode not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{request.mode}' unavailable for this dataset. Available: {available}"
        )

    data = reload_data(session)

    # preset supplies defaults; user overrides win
    preset = get_preset(session.get("dataset_name"), request.mode, task=session.get("task"))
    overrides = request.preprocessing.model_dump(exclude_unset=True) if request.preprocessing else {}
    config = {**preset, **overrides}

    epochs, config = prepare_epochs(data, config, runs=session.get("runs"))
    y, class_names, class_counts = get_labels(epochs)

    if len(set(y.tolist())) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least two classes to classify; found {class_names}."
        )

    # motor_imagery keeps CSP inside the CV pipeline, so no features here
    X, extras = (None, {})
    if request.mode != "motor_imagery":
        X, extras = extract_features(epochs, request.mode, labels=y)

    # Every mode gets the PSD + topomaps; spectral mode reuses the PSD it
    # already computed for its features instead of computing it twice.
    psd_data, topomaps = spectral_summary(
        epochs, y, class_names, FREQUENCY_BANDS,
        psds=extras.get("raw_psds"), freqs=extras.get("freqs"),
    )

    clf = request.classifiers
    configs = dict(clf.configs)

    if "eegnet" in clf.selected:
        eeg = configs.setdefault("eegnet", {})
        eeg["epochs"] = min(eeg.get("epochs", MAX_EEGNET_EPOCHS), MAX_EEGNET_EPOCHS)

    results = run_all_classifiers(
        selected_classifiers=clf.selected,
        mode=request.mode,
        configs=configs,
        X=X,
        y=y,
        epochs_obj=epochs,
        csp_config={"n_components": clf.csp_components, "reg": clf.csp_reg},
        cv_folds=clf.n_folds,
    )

    successful = [r for r in results if r.get("accuracy_mean") is not None]
    best = max(successful, key=lambda r: r["accuracy_mean"]) if successful else None

    return {
        "session_id": request.session_id,
        "mode": request.mode,
        "dataset_summary": {
            "source": session.get("source"),
            "dataset": session.get("dataset_name") or session.get("file_path"),
            "subjects": session.get("subjects"),
            "runs": session.get("runs"),
            "n_epochs": int(len(epochs)),
            "n_channels": int(len(epochs.ch_names)),
            "sampling_rate": float(epochs.info["sfreq"]),
            "epoch_duration": float(epochs.tmax - epochs.tmin),
            "classes": class_names,
            "class_counts": class_counts,
        },
        "preprocessing_applied": describe_preprocessing(data, config, epochs, session.get("source")),
        "classifiers": results,
        "best_classifier": best["classifier"] if best else None,
        "feature_info": {k: v for k, v in extras.items()
                         if k in ("feature_names", "band_names", "n_components")},
        "psd": psd_data,
        "topomaps": topomaps,
    }