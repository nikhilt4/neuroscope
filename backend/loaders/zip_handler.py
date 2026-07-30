import zipfile
from fastapi import UploadFile, HTTPException
from pathlib import Path



def zip_handler(zip_file: UploadFile, session_id: str) -> list[Path]:
    """
    Unzips the uploaded zip file and saves the contents to a temporary directory.
    
    
    Returns a list of paths to the extracted files.
    
    
    """
    session_dir = Path("/tmp/neuroscope") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file.file, 'r') as zf:
            zf.extractall(session_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")

    extracted_files = [session_dir / name for name in zf.namelist()]
    
    return extracted_files





