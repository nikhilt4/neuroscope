import mne
from mne.datasets import eegbci, sleep_physionet
from fastapi import HTTPException
from pathlib import Path


MNE_DATASETS = {
    "eegbci": {
        "name": "PhysioNet Motor Imagery",
        "n_subjects": 109,
        "task": "motor_imagery",
        "description": "Left/right hand and fist/feet motor imagery, 64 channels.",
        "runs": {
            "left_right_hand": [4, 8, 12],
            "hands_feet": [6, 10, 14],
        }
    },
    "sleep_physionet": {
        "name": "Sleep-EDF",
        "n_subjects": 78,
        "task": "spectral",
        "description": "Sleep stage polysomnography recordings.",
    },
    "ssvep": {
        "name": "SSVEP Dataset",
        "n_subjects": 1,
        "task": "ssvep",
        "description": "Steady-state visually evoked potentials, single demo subject.",
    },
    "erp_core": {
        "name": "ERP CORE",
        "n_subjects": 5,
        "task": "erp",
        "description": "6 ERP paradigms (N170, MMN, N2pc, N400, P3, LRP/ERN), subjects 015-019.",
        "sessions": ["N400", "ERN", "LRP", "MMN", "N2pc", "N170", "P3"],
        "is_bids": True,
    },
    "limo": {
        "name": "LIMO EEG",
        "n_subjects": 18,
        "task": "erp",
        "description": "Face discrimination ERP task (Face/A vs Face/B), 128 channels, pre-epoched.",
    },
    "kiloword": {
        "name": "Kiloword",
        "n_subjects": 1,
        "task": "erp",
        "description": "EEG during single-word reading, word-level metadata (concreteness, frequency, etc). Grand-average epochs, not per-subject.",
    }
}


def list_mne_datasets() -> list:
    return [
        {"id": key, **value} for key, value in MNE_DATASETS.items()
    ]


def fetch_mne_dataset(dataset_name: str, subjects: list, runs: list = None):
    if dataset_name not in MNE_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown MNE dataset: {dataset_name}")

    subject = subjects[0]

    try:
        if dataset_name == "eegbci":
            run_list = runs or MNE_DATASETS["eegbci"]["runs"]["left_right_hand"]
            raw_fnames = eegbci.load_data(subject=subject, runs=run_list, verbose=False)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
            raw = mne.concatenate_raws(raws)
            eegbci.standardize(raw)
            return raw

        elif dataset_name == "sleep_physionet":
            paths = sleep_physionet.age.fetch_data(subjects=[subject], recording=[1], verbose=False)
            psg_path, hyp_path = paths[0]
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            annotations = mne.read_annotations(hyp_path)
            raw.set_annotations(annotations, emit_warning=False)
            return raw

        elif dataset_name == "ssvep":
            from mne.datasets import ssvep
            data_path = ssvep.data_path()
            # NOTE: verify exact filename pattern against your installed mne-data version
            raw_path = data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
            raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose=False)
            return raw

        elif dataset_name == "erp_core":
            from mne.datasets import erp_core
            from backend.loaders.bids_loader import scan_bids_entities, load_bids
            
            bids_root = erp_core.data_path(verbose=False)
            
            entities = scan_bids_entities(bids_root)
            
            return {"bids_root": bids_root, "entities": entities}
            

        elif dataset_name == "limo":
            from mne.datasets import limo
            epochs = limo.load_data(subject=subject, verbose=False)
            return epochs  # this one returns Epochs directly, not Raw

        elif dataset_name == "kiloword":
            from mne.datasets import kiloword
            data_path_list = kiloword.data_path(verbose=False)
            data_dir = data_path_list[0]  # docs confirm: returned as a list of length one

            # Filename not documented anywhere I could verify — search for the .fif file
            # rather than hardcoding a guessed name
            fif_files = list(Path(data_dir).glob("*-epo.fif")) or list(Path(data_dir).glob("*.fif"))

            if not fif_files:
                raise HTTPException(
                    status_code=500,
                    detail=f"Kiloword dataset downloaded but no .fif file found in {data_dir}. "
                        f"Contents: {list(Path(data_dir).iterdir())}"
                )

            epochs = mne.read_epochs(fif_files[0], preload=True, verbose=False)
            return epochs

        else:
            raise HTTPException(status_code=400, detail=f"Fetcher not implemented for: {dataset_name}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch MNE dataset: {str(e)}")