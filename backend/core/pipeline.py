import mne
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException

from backend.core.preprocessor import preprocess_raw, _crop_epochs
from backend.core.presets import get_preset, resolve_annotation_rename


def reload_data(session: dict):
    """
    Option A: reconstruct the session's data from scratch on each request.
    Stateless — survives restarts and works across multiple workers.
    Costs a re-load (and re-download check) per call; see the .fif cache
    plan for the optimisation.
    """
    source = session.get("source")

    if source == "upload":
        fmt = session.get("format")
        path = Path(session["file_path"])
        if not path.exists():
            raise HTTPException(status_code=410, detail="Session data expired — please re-upload.")

        # tabular formats need the user's column/variable mapping, stored as a
        # dict in the session file and rehydrated back into its Pydantic model
        if fmt in ("csv", "tsv"):
            from backend.models import ColumnMapping
            from backend.loaders.tabular_loader import load_csv, load_tsv
            if not session.get("column_mapping"):
                raise HTTPException(
                    status_code=400,
                    detail="This file still needs column configuration — POST /upload/configure first."
                )
            mapping = ColumnMapping(**session["column_mapping"])
            return load_csv(path, mapping) if fmt == "csv" else load_tsv(path, mapping)

        if fmt == "mat":
            from backend.models import VariableMapping
            from backend.loaders.tabular_loader import load_mat
            if not session.get("variable_mapping"):
                raise HTTPException(
                    status_code=400,
                    detail="This file still needs variable configuration — POST /upload/configure-mat first."
                )
            return load_mat(path, VariableMapping(**session["variable_mapping"]))

        from backend.loaders.file_loader import load_file
        mne_obj, _ = load_file(path, fmt)   # returns (obj, obj_type)
        return mne_obj

    if source == "bids":
        from backend.loaders.bids_loader import load_bids
        return load_bids(
            Path(session["bids_root"]),
            subject=session.get("subject"),
            task=session.get("task"),
            session=session.get("session"),
            run=session.get("run"),
        )

    if source == "mne":
        from backend.fetchers.mne_fetcher import fetch_mne_dataset
        return fetch_mne_dataset(session["dataset_name"],
                                 session["subjects"], session.get("runs"))

    if source == "moabb":
        from backend.fetchers.moabb_fetcher import fetch_moabb_dataset
        return fetch_moabb_dataset(session["dataset_name"], session["subjects"])

    if source == "braindecode":
        from backend.fetchers.braindecode_fetcher import fetch_braindecode_dataset
        return fetch_braindecode_dataset(session["dataset_name"], session["subjects"])

    raise HTTPException(status_code=400, detail=f"Cannot reload session from source: {source}")


def prepare_epochs(data, config: dict, runs: Optional[list] = None) -> Tuple[mne.BaseEpochs, dict]:
    """
    Single front door to the preprocessing stage.

    Data arriving already epoched (MOABB paradigms, limo, kiloword, FIF uploads)
    skips preprocess_raw entirely — MOABB has already applied its own
    literature-validated filtering and epoching, and re-filtering would be wrong.
    """
    if isinstance(data, mne.BaseEpochs):
        return _crop_epochs(data, config.get("epoch_train_crop")), config
    if config.get("annotation_rename"):
        mapping = resolve_annotation_rename(config, runs=runs)
        config = {**config, "annotation_rename": mapping}
        # The classes to classify are whatever the rename produced for THIS run
        # selection (eegbci: left/right hand for runs 4/8/12, hands/feet for 6/10/14).
        # Hardcoding them in the preset broke the hands/feet variant. Unrenamed
        # annotations (eegbci's T0 = rest) are dropped. An explicit include_events
        # from the user still wins.
        if not config.get("include_events"):
            config["include_events"] = list(dict.fromkeys(mapping.values()))
    return preprocess_raw(data, config), config


def get_labels(epochs: mne.Epochs) -> Tuple[np.ndarray, list]:
    """
    Returns integer labels plus human-readable class names in matching order,
    so results say 'left_hand' rather than '1'.

    Everything is cast to plain Python types — MNE hands back np.str_ keys and
    numpy integers, which break JSON serialisation in the response.
    """
    y = epochs.events[:, 2]
    id_to_name = {int(v): str(k) for k, v in epochs.event_id.items()}
    class_names = [id_to_name.get(int(c), str(c)) for c in sorted(set(y.tolist()))]
    counts = {id_to_name.get(int(c), str(c)): int((y == c).sum()) for c in sorted(set(y.tolist()))}
    return y, class_names, counts