from fastapi import HTTPException
from typing import Optional


# Curated display metadata for the frontend dropdown — a starting subset, NOT a whitelist.
# fetch_moabb_dataset works for any valid MOABB dataset class name, whether or not
# it's listed here. These entries were manually verified against MOABB's dataset
# pages (see conversation history) since dataset.metadata has known inconsistencies
# for some fields (confirmed: BNCI2014_001's lowpass_hz and study_design don't match
# the real downloaded data — MOABB's own catalog is marked "under active consolidation").
MOABB_DATASETS = {
    "BNCI2014_001": {
        "name": "BCI Competition IV 2a",
        "n_subjects": 9,
        "n_sessions": 2,
        "n_channels": 25,       # 22 EEG + 3 EOG
        "n_eeg_channels": 22,
        "sampling_rate": 250,
        "reference": "left_mastoid",
        "task": "motor_imagery",
        "classes": ["left_hand", "right_hand", "feet", "tongue"],
        "trial_duration_s": 4.0,
    },
    "BNCI2014_002": {
        "name": "BCI Competition IV 2b",
        "n_subjects": 14,
        "n_channels": 15,
        "sampling_rate": 512,
        "reference": "left_mastoid",
        "montage": "laplacian",   # NOT standard 10-20 — special Laplacian derivation layout
        "task": "motor_imagery",
        "classes": ["right_hand", "feet"],
        "trial_duration_s": 5.0,
        "n_sessions": 1,
    },
    "Zhou2016": {
        "name": "Zhou 2016 Motor Imagery",
        "n_subjects": 4,
        "n_channels": 14,
        "sampling_rate": 250,
        "reference": "left_mastoid",
        "montage": "standard_1020",
        "task": "motor_imagery",
        "classes": ["left_hand", "right_hand", "feet"],
        "trial_duration_s": 10.0,
        "n_sessions": 3,
    },
}

# Maps a dataset's live .paradigm string to the MOABB paradigm class that should process it.
# .paradigm is a confirmed-reliable attribute (verified twice against real dataset objects).
PARADIGM_CLASSES = {
    "imagery": "MotorImagery",
    "p300": "P300",
    "ssvep": "SSVEP",
    "cvep": "CVEP",
    "restingstate": "RestingStateToP300Adapter",  # unverified — flag if this errors
}


def list_moabb_datasets() -> list:
    """
    Returns curated, hand-verified datasets for the frontend dropdown.
    This is NOT the full list of what fetch_moabb_dataset can handle —
    any dataset class in moabb.datasets can be fetched by name even if
    it isn't listed here.
    """
    return [{"id": key, **value} for key, value in MOABB_DATASETS.items()]


def restore_class_names(epochs, labels) -> "mne.BaseEpochs":
    """
    Rebuilds epochs.event_id from MOABB's per-trial `labels`, so downstream code
    sees {'left_hand': 1, ...} instead of {'1': 1, ...}.

    Verified on BNCI2014_001 subject 1 (MOABB 1.5.0): event codes 1–4 pair
    one-to-one with left_hand / right_hand / feet / tongue across all 576 trials.

    The mapping is only applied if every code maps to exactly one label and vice
    versa. Anything else means the codes and labels disagree, and guessing would
    put the wrong name on a class — worse than showing the number.
    """
    codes = epochs.events[:, 2].tolist()
    names = [str(label) for label in labels]
    if len(codes) != len(names):
        return epochs

    code_to_name: dict = {}
    for code, name in zip(codes, names):
        if code_to_name.setdefault(int(code), name) != name:
            return epochs                          # one code, two names
    if len(set(code_to_name.values())) != len(code_to_name):
        return epochs                              # two codes, one name

    epochs.event_id = {name: code for code, name in sorted(code_to_name.items())}
    return epochs


def fetch_moabb_dataset(dataset_name: str, subjects: list, n_classes: Optional[int] = None):
    """
    Fetches and preprocesses a MOABB dataset using the correct paradigm class,
    determined dynamically from the dataset's own .paradigm attribute.
    Works for any valid MOABB dataset name, not just curated entries above.
    """
    try:
        import moabb.datasets as md
        import moabb.paradigms as mp

        dataset_class = getattr(md, dataset_name, None)
        if dataset_class is None:
            raise HTTPException(status_code=400, detail=f"MOABB dataset class not found: {dataset_name}")

        dataset = dataset_class()

        paradigm_type = dataset.paradigm  # confirmed real attribute: "imagery", "ssvep", etc.

        paradigm_class_name = PARADIGM_CLASSES.get(paradigm_type)
        if paradigm_class_name is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported paradigm type '{paradigm_type}' for dataset '{dataset_name}'."
            )

        paradigm_class = getattr(mp, paradigm_class_name, None)
        if paradigm_class is None:
            raise HTTPException(
                status_code=500,
                detail=f"Paradigm class '{paradigm_class_name}' not found in moabb.paradigms."
            )

        if paradigm_type == "imagery":
            # dataset.event_id is confirmed reliable — gives real class count directly
            resolved_n_classes = n_classes or len(dataset.event_id) or 2
            paradigm = paradigm_class(n_classes=resolved_n_classes)
        else:
            paradigm = paradigm_class()

        # https://moabb.neurotechx.com/docs/generated/moabb.paradigms.MotorImagery.html (MOABB 1.5)
        # With return_epochs=True the EpochsArray carries event_id {'1': 1, '2': 2, ...};
        # the real class names only survive in the separate `labels` array.
        epochs, labels, metadata = paradigm.get_data(
            dataset=dataset,
            subjects=subjects,
            return_epochs=True
        )

        return restore_class_names(epochs, labels)  # same contract as limo/kiloword/FIF loaders

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch MOABB dataset: {str(e)}")