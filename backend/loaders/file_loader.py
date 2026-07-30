from pathlib import Path
from fastapi import HTTPException
import mne


def load_file(path: Path, format: str) -> tuple[mne.io.Raw | mne.Epochs, str]:
    """
    routes to the right loader below returns mne_object, object_type where type is raw or epochs
    """
    match format:
        case "edf":
            return load_edf(path), "raw"
        case "bdf":
            return load_bdf(path), "raw"
        case "gdf":
            return load_gdf(path), "raw"
        case "fif":
            return load_fif(path), "epochs"
        case "brainvision":
            return load_brainvision(path), "raw"
        case _:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {format}")



def load_edf(path: Path) -> mne.io.Raw:
    """use mne.io.read_raw_edf to load edf file"""
    
    try:
        return mne.io.read_raw_edf(path, preload=True, verbose=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load EDF file: {str(e)}")



def load_bdf(path: Path) -> mne.io.Raw:
    """use mne.io.read_raw_bdf to load bdf file"""
    
    try:
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load BDF file: {str(e)}")

def load_gdf(path: Path) -> mne.io.Raw:
    """use mne.io.read_raw_gdf to load gdf file"""
    
    try:
        return mne.io.read_raw_gdf(path, preload=True, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load GDF file: {str(e)}")


def load_fif(path: Path) -> mne.Epochs:       # note: Epochs not Raw
    """use mne.read_epochs to load fif file"""
    
    try:
        return mne.read_epochs(path, preload=True, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load FIF file: {str(e)}")


def load_brainvision(vhdr_path: Path) -> mne.io.Raw:
    """use mne.io.read_raw_brainvision to load brainvision file"""
    
    try:
        return mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load BrainVision file: {str(e)}")