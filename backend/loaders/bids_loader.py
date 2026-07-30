import mne
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import HTTPException
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals



def scan_bids_entities(root: Path) -> List[Dict]:
    """
    Scans a BIDS dataset root and returns all available subject/session/task/run
    combinations that have EEG data, so the user can pick which one to analyze.
    """
    try:
        subjects = get_entity_vals(root, "subject", ignore_datatypes=["anat", "func", "dwi"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read BIDS dataset: {str(e)}")

    if not subjects:
        raise HTTPException(status_code=400, detail="No subjects found in BIDS dataset.")

    try:
        sessions = get_entity_vals(root, "session")
    except Exception:
        sessions = []

    try:
        tasks = get_entity_vals(root, "task")
    except Exception:
        tasks = []

    try:
        runs = get_entity_vals(root, "run")
    except Exception:
        runs = []

    combinations = []
    for subject in subjects:
        entry = {"subject": subject}
        if sessions:
            entry["sessions"] = sessions
        if tasks:
            entry["tasks"] = tasks
        if runs:
            entry["runs"] = runs
        combinations.append(entry)

    return combinations


def load_bids(root: Path, subject: str, task: Optional[str] = None, session: Optional[str] = None, run: Optional[str] = None) -> mne.io.Raw:
    """
    Loads a specific subject/session/task/run from a BIDS dataset root.
    Returns an MNE Raw object with events already attached via mne-bids.
    """
    try:
        bids_path = BIDSPath(root=root, subject=subject, task=task, session=session, run=run, suffix="eeg")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
        return raw
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load BIDS dataset: {str(e)}")

