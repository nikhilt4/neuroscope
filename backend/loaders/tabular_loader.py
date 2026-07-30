import mne
import csv
import pandas as pd
from pathlib import Path
import scipy.io
import numpy as np
import h5py
from fastapi import HTTPException

from backend.models import ColumnMapping, VariableMapping


def inspect_csv(path: Path) -> dict:
    
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    file = pd.read_csv(path, nrows=3)
    
    return {
        "columns": file.columns.tolist(),
        "n_rows": len(file),
        "preview": file.head(3).values.tolist(),
        "delimiter": ","
    }

def inspect_tsv(path: Path) -> dict:
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    file = pd.read_csv(path, delimiter="\t", nrows=3)
    
    return {
        "columns": file.columns.tolist(),
        "n_rows": len(file),
        "preview": file.head(3).values.tolist(),
        "delimiter": "\t"
    }




def inspect_mat(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    
    try:
        variables = scipy.io.whosmat(path)
        return {
            "variables": [{"name": var[0], "shape": var[1], "dtype": str(var[2])} for var in variables]
        }
    except (NotImplementedError, ValueError):
        if not h5py.is_hdf5(path):
            raise ValueError(f"File is not a valid MATLAB .mat file: {path}")
    
    
    with h5py.File(path, "r") as mat:
        variables = []

        for key, item in mat.items():
            if key.startswith("#"):
                continue

            if isinstance(item, h5py.Dataset):
                variables.append({
                    "name": key,
                    "shape": list(item.shape),
                    "dtype": str(item.dtype)
                })

            elif isinstance(item, h5py.Group):
                variables.append({
                    "name": key,
                    "shape": None,
                    "dtype": "group"
                })

        return {"variables": variables}

    
    

def load_csv(path: Path, mapping: ColumnMapping) -> mne.io.Raw:
    """ 
    All three build an MNE RawArray:
    1. pandas reads the file
    2. extract EEG columns as numpy array [n_channels × n_samples]
    3. build mne.Info object with channel names + sampling rate
    4. return mne.io.RawArray(data, info)

    """
    if mapping.sampling_rate is None:
        raise HTTPException(status_code=400, detail="Sampling rate must be provided for CSV files.")
    
    df = pd.read_csv(path)
    
    for col in mapping.eeg_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"EEG column '{col}' not found in CSV file.")
    
    eeg_data = df[mapping.eeg_columns].to_numpy().T  # Transpose to get shape [n_channels × n_samples]
    
    info = mne.create_info(ch_names=mapping.eeg_columns, sfreq=mapping.sampling_rate, ch_types='eeg')
    
    return mne.io.RawArray(eeg_data, info, verbose=False)
    
    
   
    
def load_tsv(path: Path, mapping: ColumnMapping) -> mne.io.Raw:
    if mapping.sampling_rate is None:
        raise HTTPException(status_code=400, detail="Sampling rate must be provided for TSV files.")
    
    df = pd.read_csv(path, delimiter="\t")
    
    for col in mapping.eeg_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"EEG column '{col}' not found in TSV file.")
    
    eeg_data = df[mapping.eeg_columns].to_numpy().T  # Transpose to get shape [n_channels × n_samples]
    
    info = mne.create_info(ch_names=mapping.eeg_columns, sfreq=mapping.sampling_rate, ch_types='eeg')
    
    return mne.io.RawArray(eeg_data, info, verbose=False)




def load_mat(path: Path, mapping: VariableMapping) -> mne.io.Raw | mne.Epochs:
    
    if mapping.sampling_rate is None:
        raise HTTPException(status_code=400, detail="Sampling rate must be provided for MAT files.")
    
    try:
        mat = scipy.io.loadmat(path)
        if mapping.eeg_variable not in mat:
            raise HTTPException(status_code=400, detail=f"EEG variable '{mapping.eeg_variable}' not found in MAT file.")
        eeg = mat[mapping.eeg_variable]
    except NotImplementedError: 
        with h5py.File(path, "r") as mat:
            if mapping.eeg_variable not in mat:
                raise HTTPException(status_code=400, detail=f"EEG variable '{mapping.eeg_variable}' not found in MAT file.")
            eeg = mat[mapping.eeg_variable][()]
    
    if mapping.data_shape == "trials_channels_samples":
        if eeg.ndim != 3:
            raise HTTPException(status_code=400, detail=f"EEG variable '{mapping.eeg_variable}' is not 3D for trials_channels_samples shape.")
        
        n_trials, n_channels, n_samples = eeg.shape
        info = mne.create_info(ch_names=[f"EEG {i+1}" for i in range(n_channels)], sfreq=mapping.sampling_rate, ch_types='eeg')
        
        return mne.EpochsArray(eeg, info, verbose=False)
    
    elif mapping.data_shape == "channels_samples":
        if eeg.ndim != 2:
            raise HTTPException(status_code=400, detail=f"EEG variable '{mapping.eeg_variable}' is not 2D for channels_samples shape.")
        
        n_channels, n_samples = eeg.shape
        info = mne.create_info(ch_names=[f"EEG {i+1}" for i in range(n_channels)], sfreq=mapping.sampling_rate, ch_types='eeg')
        
        return mne.io.RawArray(eeg, info, verbose=False)
    
    
    elif mapping.data_shape == "samples_channels":
        if eeg.ndim != 2:
            raise HTTPException(status_code=400, detail=f"EEG variable '{mapping.eeg_variable}' is not 2D for samples_channels shape.")
        
        n_samples, n_channels = eeg.shape
        info = mne.create_info(ch_names=[f"EEG {i+1}" for i in range(n_channels)], sfreq=mapping.sampling_rate, ch_types='eeg')
        
        return mne.io.RawArray(eeg.T, info, verbose=False)
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported data_shape: {mapping.data_shape}. Must be one of 'trials_channels_samples', 'channels_samples', or 'samples_channels'.")
        