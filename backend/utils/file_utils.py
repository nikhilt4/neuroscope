from fastapi import HTTPException, UploadFile
from pathlib import Path
from typing import List, Dict
import shutil




def detect_format(file_path: str) -> str:
    """
    Detects the format of the uploaded EEG file based on its extension.
    
    "file.edf"   → "edf"
    "file.bdf"   → "bdf"
    "file.gdf"   → "gdf"
    "file.fif"   → "fif"
    "file.vhdr"  → "brainvision"
    "file.mat"   → "mat"
    "file.csv"   → "csv"
    "file.tsv"   → "tsv"
    anything else → raises HTTPException(400)
    """
    extension = Path(file_path).suffix.lower()
    
    format = extension.removeprefix(".")
    
    if format == "vhdr":
        return "brainvision"
    
    if (format not in ["edf", "bdf", "gdf", "fif", "mat", "csv", "tsv"]):
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {extension}")
    
    return format



async def save_temp_file(file: UploadFile, session_id: str) -> Path:
    """
    saves to /tmp/neuroscope/{session_id}/filename
    returns full path
    
    """
    
    session_directory = Path("/tmp/neuroscope") / session_id
    session_directory.mkdir(parents=True, exist_ok=True)
    
    file_path = session_directory / file.filename
    contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)
        
    return file_path


async def save_temp_files(files: List[UploadFile], session_id: str) -> Dict[str, Path]:
    """
    saves all files to /tmp/neuroscope/session_id/
    returns filename: path dict
    
    """
    
    session_directory = Path("/tmp/neuroscope") / session_id
    session_directory.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    for file in files:
        file_path = session_directory / file.filename
        contents = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        saved[file.filename] = file_path
    
    return saved
    
    

# ADD - EXTRA ERROR INFO FOR EACH FILE TYPE
def detect_multi_format(files: List[UploadFile]) -> str:
    """
    looks at the collection of filenames
    ".vhdr" present → "brainvision"
    "_eeg.json" present → "bids"
    all .csv/.tsv → "tabular_multi"
    mixed unknown → raises HTTPException(400)
    """
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    vhdr_set = {".vhdr", ".vmrk", ".eeg"}
    tabular_set = {".csv", ".tsv"}
    file_ext_set = {Path(f.filename).suffix.lower() for f in files}
    filenames = [f.filename for f in files]

        
    if ".vhdr" in file_ext_set and ".vmrk" in file_ext_set and ".eeg" in file_ext_set:
        return "brainvision"
        
    if any(name.endswith("_eeg.json") for name in filenames):
        return "bids"
    
    if file_ext_set.issubset(tabular_set):
        return "tabular_multi"
    

    raise HTTPException(
        status_code=400,
        detail=(
            f"Could not determine collection type from uploaded files: "
            f"{', '.join(filenames)}. "
            f"Expected: BrainVision (.vhdr+.vmrk+.eeg), "
            f"BIDS folder, or all CSV/TSV files."
        )
    )
    



def cleanup_session(session_id: str) -> None:
    """
    deletes /tmp/neuroscope/{session_id}/ entirely
    called after processing or on timeout
    """
    session_dir = Path("/tmp/neuroscope") / session_id
    
    if session_dir.exists():
        shutil.rmtree(session_dir)
    
        
