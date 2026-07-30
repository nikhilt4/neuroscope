from fastapi import HTTPException

BRAINDECODE_DATASETS = {
    "BNCI2014_001": {
        "name": "BCI Competition IV 2a (Braindecode)",
        "n_subjects": 9,
        "task": "motor_imagery",
        "description": "Same as MOABB version, pre-formatted for PyTorch training.",
    },
    "Schirrmeister2017": {
        "name": "High-Gamma Dataset",
        "n_subjects": 14,
        "task": "motor_imagery",
        "description": "High-density motor imagery dataset, good for deep learning.",
    },
}


def list_braindecode_datasets() -> list:
    return [{"id": key, **value} for key, value in BRAINDECODE_DATASETS.items()]


def fetch_braindecode_dataset(dataset_name: str, subjects: list):
    if dataset_name not in BRAINDECODE_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown Braindecode dataset: {dataset_name}")

    try:
        from braindecode.datasets import MOABBDataset

        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subjects)

        # braindecode wraps MOABB's raw objects in its own dataset structure
        # each "dataset.datasets[i].raw" is an mne.io.Raw
        if len(dataset.datasets) == 0:
            raise HTTPException(status_code=500, detail="No data returned from Braindecode fetch.")

        raw = dataset.datasets[0].raw
        return raw

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Braindecode dataset: {str(e)}")