from pathlib import Path
from fastapi import HTTPException
import re

TEMP_BASE = Path("/tmp/neuroscope")


def validate_openneuro_id(dataset_id: str) -> None:
    if not re.match(r"^ds\d{6}$", dataset_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid OpenNeuro accession number. Expected format: ds003490"
        )


def fetch_openneuro_dataset(dataset_id: str, session_id: str) -> Path:
    """
    Downloads an OpenNeuro dataset by accession ID.
    Returns the local root path once complete.
    Runs synchronously for now — wrap in BackgroundTasks at the route level
    for large datasets later.
    """
    validate_openneuro_id(dataset_id)

    target_dir = TEMP_BASE / session_id
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openneuro
        openneuro.download(dataset=dataset_id, target_dir=str(target_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download from OpenNeuro: {str(e)}")

    if not any(target_dir.glob("sub-*")):
        raise HTTPException(
            status_code=400,
            detail=f"Downloaded dataset {dataset_id} does not appear to be a valid BIDS structure."
        )

    return target_dir