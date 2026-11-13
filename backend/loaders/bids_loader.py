import mne
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import HTTPException
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths

# Raw EEG extensions in the BIDS spec, minus EEGLAB .set (deliberately unsupported in
# this project). Matching the header/data file only (.vhdr, not .eeg/.vmrk) gives
# exactly one hit per recording.
# https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
EEG_EXTENSIONS = [".edf", ".bdf", ".vhdr"]


def scan_bids_entities(root: Path) -> List[Dict]:
    """
    Every subject/session/task/run combination that actually has an EEG recording,
    as a flat list: [{"subject": "01", "session": None, "task": "imagery", "run": "1"}, ...].

    Flat combinations, not per-entity lists, so the frontend's dropdowns can narrow
    each other: picking subject 02 only offers the tasks subject 02 really has.
    Scanning entities independently would offer combinations with no file behind them,
    which then fail at load time.
    """
    try:
        # https://mne.tools/mne-bids/stable/generated/mne_bids.find_matching_paths.html (MNE-BIDS 0.19)
        paths = find_matching_paths(root, datatypes="eeg", suffixes="eeg", extensions=EEG_EXTENSIONS)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read BIDS dataset: {e}")

    combos = {
        (p.subject, p.session, p.task, p.run)
        for p in paths
        if p.subject
    }
    if not combos:
        raise HTTPException(
            status_code=400,
            detail="No EEG recordings found. Expected files like "
                   "sub-01/eeg/sub-01_task-x_eeg.edf (EDF, BDF or BrainVision).",
        )

    keys = ("subject", "session", "task", "run")
    # sort with None treated as "" so mixed None/str values compare
    return [dict(zip(keys, c)) for c in sorted(combos, key=lambda c: tuple(v or "" for v in c))]


def load_bids(root: Path, subject: str, task: Optional[str] = None, session: Optional[str] = None, run: Optional[str] = None) -> mne.io.Raw:
    """
    Loads a specific subject/session/task/run from a BIDS dataset root.
    read_raw_bids attaches events from *_events.tsv as annotations, so the detector
    finds them the same way it does for EDF uploads.
    """
    try:
        # https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html (MNE-BIDS 0.19)
        bids_path = BIDSPath(root=root, subject=subject, task=task, session=session, run=run,
                             suffix="eeg", datatype="eeg")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
        return raw
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load BIDS recording: {e}")
