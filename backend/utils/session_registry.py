# from pathlib import Path



# _external_bids_roots: dict = {}

# def register_external_bids_root(session_id: str, root: Path):
#     _external_bids_roots[session_id] = root

# def get_external_bids_root(session_id: str):
#     return _external_bids_roots.get(session_id)




# backend/utils/session_registry.py
import json
from pathlib import Path
from typing import Optional
from fastapi import HTTPException

TEMP_BASE = Path("/tmp/neuroscope")


def _meta_path(session_id: str) -> Path:
    return TEMP_BASE / session_id / "session.json"


def register_session(session_id: str, **meta) -> dict:
    """
    Persists everything needed to reconstruct this session's data later.
    Written to disk rather than held in memory so sessions survive the
    uvicorn reloads that happen constantly during development, and so
    multiple workers can serve the same session.
    """
    path = _meta_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())

    merged = {**existing, **meta, "session_id": session_id}
    path.write_text(json.dumps(merged, indent=2, default=str))
    return merged


def get_session(session_id: str) -> dict:
    path = _meta_path(session_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please re-upload or re-fetch the dataset."
        )
    return json.loads(path.read_text())


def get_external_bids_root(session_id: str) -> Optional[str]:
    return get_session(session_id).get("bids_root")


def register_external_bids_root(session_id: str, root) -> None:
    register_session(session_id, bids_root=str(root))