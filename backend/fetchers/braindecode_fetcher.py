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
        import mne
        from braindecode.datasets import MOABBDataset

        # https://braindecode.org/stable/generated/braindecode.datasets.MOABBDataset.html (braindecode 1.6)
        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subjects)

        # braindecode returns ONE sub-dataset per session x run. BNCI2014_001 subject 1
        # is 12 of them (2 sessions x 6 runs, 48 trials each). Taking only datasets[0]
        # silently used 48 of 576 trials and scored below chance (fetch audit, 13 Nov).
        if len(dataset.datasets) == 0:
            raise HTTPException(status_code=500, detail="No data returned from Braindecode fetch.")

        raws = []
        for sub in dataset.datasets:
            raw = sub.raw.load_data()
            # Drop the stim channel: the detector and preprocessor read a stim channel
            # before annotations, which turned 'left_hand' etc. into bare codes '1'..'4'.
            # The annotations carry the same events WITH their names.
            stim = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "stim"]
            if stim:
                raw = raw.drop_channels(stim)
            raws.append(raw)

        # https://mne.tools/1.12/generated/mne.concatenate_raws.html — joins add
        # "BAD boundary"/"EDGE boundary" annotations, which events_from_annotations
        # skips by default, so they never become classes.
        return mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Braindecode dataset: {str(e)}")