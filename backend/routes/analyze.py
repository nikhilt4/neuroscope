from fastapi import APIRouter, HTTPException

from backend.models import AnalysisRequest
from backend.utils.session_registry import get_session
from backend.core.presets import get_preset
from backend.core.pipeline import reload_data, prepare_epochs, get_labels
from backend.core.features import extract_features
from backend.core.classifiers import run_all_classifiers

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Web-demo guard: 300 epochs x 5 folds is minutes of training and would hold
# a worker hostage. Power users can still raise it via the API. (backlog 17)
MAX_EEGNET_EPOCHS = 50


@router.post("")
async def analyze(request: AnalysisRequest):
    session = get_session(request.session_id)

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
            "n_epochs": int(len(epochs)),
            "n_channels": int(len(epochs.ch_names)),
            "sampling_rate": float(epochs.info["sfreq"]),
            "epoch_duration": float(epochs.tmax - epochs.tmin),
            "classes": class_names,
            "class_counts": class_counts,
        },
        "preprocessing_applied": {k: v for k, v in config.items() if not k.startswith("_")},
        "classifiers": results,
        "best_classifier": best["classifier"] if best else None,
        "feature_info": {k: v for k, v in extras.items()
                         if k in ("feature_names", "band_names", "n_components")},
    }