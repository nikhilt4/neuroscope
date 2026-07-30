from pathlib import Path

_external_bids_roots: dict = {}

def register_external_bids_root(session_id: str, root: Path):
    _external_bids_roots[session_id] = root

def get_external_bids_root(session_id: str):
    return _external_bids_roots.get(session_id)