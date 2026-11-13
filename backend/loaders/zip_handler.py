import zipfile
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

# Uncompressed-size cap. A ZIP's declared sizes are checked BEFORE extracting, so a
# zip bomb (tiny archive, huge contents) is refused without touching the disk.
# Schirrmeister2017 — the largest dataset we handle — is a few GB in total, but one
# uploaded BIDS dataset for a single analysis shouldn't need more than this.
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


def _is_junk(name: str) -> bool:
    """macOS resource forks and hidden files. __MACOSX/ would otherwise show up as an
    extra top-level folder and hide the real dataset root."""
    parts = Path(name).parts
    return any(p == "__MACOSX" or p.startswith(".") for p in parts)


def find_bids_root(base: Path) -> Optional[Path]:
    """
    The folder that holds dataset_description.json (required at the root by the BIDS
    spec: https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html),
    or failing that, the shallowest folder that directly contains sub-* folders.
    Shallowest wins, so a derivatives/ copy deeper down is never picked by mistake.
    """
    candidates = sorted(base.rglob("dataset_description.json"), key=lambda p: len(p.parts))
    if candidates:
        return candidates[0].parent

    sub_dirs = sorted((p for p in base.rglob("sub-*") if p.is_dir()), key=lambda p: len(p.parts))
    return sub_dirs[0].parent if sub_dirs else None


def zip_handler(zip_path: Path, session_id: str, base_dir: Path = Path("/tmp/neuroscope")) -> Path:
    """
    Extracts an uploaded BIDS ZIP and returns the dataset root.

    Handles the two ways people actually zip a dataset: contents at the top level,
    or everything inside one folder (what "Compress" does on macOS and Windows).
    The ZIP itself is deleted after extraction; only the extracted tree is needed.

    Path safety: ZipFile.extract strips absolute paths, drive letters and ".."
    components, so members cannot escape the target folder.
    https://docs.python.org/3.12/library/zipfile.html#zipfile.ZipFile.extract
    """
    zip_path = Path(zip_path)
    target = base_dir / session_id / "bids"
    target.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.infolist() if not _is_junk(m.filename)]
            total = sum(m.file_size for m in members)
            if total > MAX_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"This ZIP expands to {total / 1024**3:.1f} GB; the limit is "
                           f"{MAX_UNCOMPRESSED_BYTES / 1024**3:.0f} GB. Zip just the subjects you need.",
                )
            for member in members:
                zf.extract(member, target)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="That file isn't a valid ZIP archive.")
    finally:
        zip_path.unlink(missing_ok=True)

    root = find_bids_root(target)
    if root is None:
        raise HTTPException(
            status_code=400,
            detail="No BIDS dataset found in this ZIP. It needs a dataset_description.json "
                   "file and sub-XX folders at its top level (or inside one folder).",
        )
    return root
