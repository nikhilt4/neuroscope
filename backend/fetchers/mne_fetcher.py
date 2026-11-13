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
        "name": "ERP CORE (Flankers)",
        "n_subjects": 1,
        "n_channels": 33,
        "sampling_rate": 1024,
        "task": "erp",
        "description": "Single subject, Flankers task (ERN paradigm). Compatible/incompatible stimuli with left/right targets, plus response events.",
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


# Sleep-EDF stages (Rechtschaffen & Kales) -> AASM names. Stages 3 and 4 merge into N3,
# as in MNE's sleep tutorial, where both map to one event id ("Sleep stage 3/4").
# "Sleep stage ?" and "Movement time" are left out, so they never become classes.
SLEEP_STAGE_NAMES = {
    "Sleep stage W": "wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
SLEEP_WAKE_MARGIN_S = 30 * 60


def load_sleep_recording(psg_path, hyp_path) -> mne.io.BaseRaw:
    """
    One Sleep-EDF night, prepared the way MNE's sleep-staging tutorial prepares it
    (https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html, MNE 1.13 docs;
    every call below checked against the installed MNE 1.12.1 with inspect.signature).

    1. Channel types. Read with infer_types=True, so 'EEG Fpz-Cz' -> eeg, 'EOG horizontal'
       -> eog, 'EMG submental' -> emg, 'Resp oro-nasal' -> resp, 'Temp rectal' -> temperature.
       Without it all 7 channels loaded as 'eeg' and the features included body temperature.
       DIFFERENCE from the tutorial: 'Event marker' is loaded as misc, not stim. Our event
       extraction reads a stim channel before annotations, so a stim marker would override
       the hypnogram.
    2. Wake trimming. A Sleep-EDF night is ~2/3 wake (subject 1: 15.5 of 23.4 h). The
       tutorial keeps only 30 min of wake before the first sleep stage and 30 min after the
       last. Without it a classifier that always says "wake" scores ~66%.
    3. Stage names. R&K labels become AASM names, with stages 3 and 4 merged into N3.

    The 30-s windows themselves come from the preset (event_chunk_duration=30), which the
    preprocessor passes to mne.events_from_annotations(chunk_duration=30.0), as in the tutorial.
    """
    raw = mne.io.read_raw_edf(psg_path, infer_types=True, misc=["Event marker"],
                              preload=True, verbose=False)
    raw.set_annotations(prepare_sleep_annotations(mne.read_annotations(hyp_path)), emit_warning=False)
    return raw


def prepare_sleep_annotations(annotations: mne.Annotations) -> mne.Annotations:
    """Wake trimming + R&K -> AASM renaming for one hypnogram (see load_sleep_recording)."""
    sleep = [i for i, d in enumerate(annotations.description)
             if d in SLEEP_STAGE_NAMES and d != "Sleep stage W"]
    if sleep:
        first, last = sleep[0], sleep[-1]
        # Same 30-min margins as the tutorial's annot.crop(annot[1]["onset"] - 30*60,
        # annot[-2]["onset"] + 30*60), but located by stage name instead of position,
        # so it doesn't rely on the night starting and ending with exactly one wake block.
        annotations.crop(
            annotations.onset[first] - SLEEP_WAKE_MARGIN_S,
            annotations.onset[last] + annotations.duration[last] + SLEEP_WAKE_MARGIN_S,
            emit_warning=False,
        )

    keep = [i for i, d in enumerate(annotations.description) if d in SLEEP_STAGE_NAMES]
    annotations = annotations[keep]
    # Annotations.rename raises for mapping keys that aren't present (MNE 1.12), and not
    # every night has every stage: Sleep-EDF subject 4 has no "Sleep stage 4". Only pass
    # the labels this hypnogram actually contains.
    present = set(annotations.description)
    annotations.rename({k: v for k, v in SLEEP_STAGE_NAMES.items() if k in present})
    return annotations


def fetch_mne_dataset(dataset_name: str, subjects: list, runs: list = None):
    if dataset_name not in MNE_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown MNE dataset: {dataset_name}")

    subject = subjects[0]

    mne.set_log_level("INFO")
    
    try:
        if dataset_name == "eegbci":
            run_list = runs or MNE_DATASETS["eegbci"]["runs"]["left_right_hand"]
            raw_fnames = eegbci.load_data(subjects=subject, runs=run_list, update_path=True, verbose=False)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
            raw = mne.concatenate_raws(raws)
            eegbci.standardize(raw)
            return raw

        elif dataset_name == "sleep_physionet":
            paths = sleep_physionet.age.fetch_data(subjects=[subject], recording=[1], verbose=False)
            psg_path, hyp_path = paths[0]
            return load_sleep_recording(psg_path, hyp_path)

        elif dataset_name == "ssvep":
            from mne.datasets import ssvep
            data_path = ssvep.data_path()
            # NOTE: verify exact filename pattern against your installed mne-data version
            raw_path = data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
            raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose=False)
            return raw

        elif dataset_name == "erp_core":
            from mne.datasets import erp_core
            data_path = erp_core.data_path(verbose=False)
            fif_files = list(Path(data_path).glob("*.fif"))
            if not fif_files:
                raise HTTPException(
                    status_code=500,
                    detail=f"ERP CORE downloaded but no .fif found in {data_path}."
                )
            raw = mne.io.read_raw_fif(fif_files[0], preload=True, verbose=False)
            return raw
            

        elif dataset_name == "limo":
            from mne.datasets import limo
            epochs = limo.load_data(subject=subject, update_path=True, verbose=False)
            return epochs  # this one returns Epochs directly, not Raw

        elif dataset_name == "kiloword":
            from mne.datasets import kiloword
            result = kiloword.data_path(update_path=True, verbose=False)
            data_dir = Path(result[0]) if isinstance(result, (list, tuple)) else Path(result)

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
    
    finally:
        mne.set_log_level("WARNING")