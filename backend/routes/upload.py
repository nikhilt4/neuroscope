from aiohttp import request
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
import base64
import json
import mne

# from backend.loaders.bids_loader import bids_loader
from backend.loaders.file_loader import load_file
from backend.loaders.tabular_loader import inspect_csv, inspect_tsv, inspect_mat, load_csv, load_tsv, load_mat
from backend.loaders.bids_loader import scan_bids_entities, load_bids
from backend.loaders.zip_handler import zip_handler


from backend.fetchers.mne_fetcher import list_mne_datasets, fetch_mne_dataset
from backend.fetchers.moabb_fetcher import list_moabb_datasets, fetch_moabb_dataset
from backend.fetchers.openneuro_fetcher import fetch_openneuro_dataset
from backend.fetchers.braindecode_fetcher import list_braindecode_datasets, fetch_braindecode_dataset


from backend.core.detector import detect_capabilities
from backend.core.presets import get_preset

from backend.utils.file_utils import detect_format, save_temp_file, save_temp_files, detect_multi_format, cleanup_session
from backend.utils.session_registry import get_external_bids_root, register_external_bids_root

from backend.models import UploadResponse, ColumnMapping, VariableMapping, DatasetFetchRequest, BidsSelection


from pydantic import BaseModel


router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

TEMP_BASE = Path("/tmp/neuroscope")

def _build_full_response(session_id: str, file_format: str, capabilities: dict, source: str = "upload") -> UploadResponse:
    return UploadResponse(
        session_id=session_id,
        format=file_format,
        n_channels=capabilities["n_channels"],
        channels=capabilities["channels"],
        sampling_rate=capabilities["sampling_rate"],
        duration_seconds=capabilities["duration_seconds"],
        events_found=capabilities["events_found"],
        event_classes=capabilities["event_classes"],
        available_modes=capabilities["available_modes"],
        recommended_mode=capabilities["recommended_mode"],
        needs_column_picker=False,
        needs_variable_picker=False,
        needs_event_setup=capabilities["needs_event_setup"],
        is_ready=not capabilities["needs_event_setup"],
        source=source,
        motor_channels=capabilities["motor_channels"],
        n_epochs=capabilities.get("n_epochs")
    )
    
    
def _build_early_response(
    session_id: str,
    file_format: str,
    needs_column_picker: bool = False,
    needs_variable_picker: bool = False,
    column_info: Optional[dict] = None,
    variable_info: Optional[dict] = None,
) -> UploadResponse:
    """
    Builds a partial UploadResponse for CSV/TSV/MAT files
    where we don't have full data yet — just inspection results.
    Frontend uses this to show the column/variable picker.
    Real values (channels, sfreq etc.) are all zero/empty until configure.
    """
    return UploadResponse(
        session_id=session_id,
        format=file_format,
        n_channels=0,
        channels=[],
        sampling_rate=0.0,
        duration_seconds=0.0,
        events_found=False,
        event_classes=[],
        available_modes=[],
        recommended_mode="basic",
        needs_column_picker=needs_column_picker,
        needs_variable_picker=needs_variable_picker,
        needs_event_setup=False,
        is_ready=False,
        source="upload",
        column_info=column_info,
        variable_info=variable_info,
    )


def _get_session_file(session_id: str) -> Path:
    """
    Finds the single saved file in a session directory.
    Used by configure endpoints to retrieve the originally uploaded file.
    Raises HTTPException if session or file not found.
    """
    session_dir = TEMP_BASE / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    files = [f for f in session_dir.iterdir() if f.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No file found for this session.")

    return files[0]



# ─── UPLOAD ROUTES ──────────────────────────────────────────

@router.post("/upload/single", response_model=UploadResponse)
async def upload_single(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    try:
        file_format = detect_format(file.filename)
    except HTTPException:
        raise

    # ── CSV / TSV → inspect only, return column info for picker
    if file_format in ["csv", "tsv"]:
        try:
            file_path = await save_temp_file(file, session_id)
            inspection = inspect_csv(file_path) if file_format == "csv" else inspect_tsv(file_path)
        except Exception as e:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to inspect file: {str(e)}")

        return UploadResponse(
            session_id=session_id,
            format=file_format,
            n_channels=0,
            channels=[],
            sampling_rate=0.0,
            duration_seconds=0.0,
            events_found=False,
            event_classes=[],
            available_modes=[],
            recommended_mode="basic",
            needs_column_picker=True,
            needs_variable_picker=False,
            needs_event_setup=False,
            is_ready=False,
            source="upload",
            column_info=inspection
        )

    # ── MAT → inspect only, return variable info for picker
    if file_format == "mat":
        try:
            file_path = await save_temp_file(file, session_id)
            inspection = inspect_mat(file_path)
        except Exception as e:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to inspect MAT file: {str(e)}")

        return UploadResponse(
            session_id=session_id,
            format=file_format,
            n_channels=0,
            channels=[],
            sampling_rate=0.0,
            duration_seconds=0.0,
            events_found=False,
            event_classes=[],
            available_modes=[],
            recommended_mode="basic",
            needs_column_picker=False,
            needs_variable_picker=True,
            needs_event_setup=False,
            is_ready=False,
            source="upload",
            variable_info=inspection
        )

    # ── EDF / BDF / GDF / FIF / BrainVision → full load path
    try:
        file_path = await save_temp_file(file, session_id)
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        mne_obj, obj_type = load_file(file_path, file_format)
    except HTTPException:
        cleanup_session(session_id)
        raise
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to load file: {str(e)}")

    try:
        capabilities = detect_capabilities(mne_obj)
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to detect capabilities: {str(e)}")

    return _build_full_response(session_id, file_format, capabilities)


@router.post("/upload/multi", response_model=UploadResponse)
async def upload_multi(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())

    try:
        multi_format = detect_multi_format(files)
    except HTTPException:
        raise

    # ── BrainVision → find the .vhdr file, load all three
    if multi_format == "brainvision":
        try:
            saved = await save_temp_files(files, session_id)
            vhdr_path = next(p for name, p in saved.items() if name.endswith(".vhdr"))
        except StopIteration:
            cleanup_session(session_id)
            raise HTTPException(status_code=400, detail="No .vhdr file found in upload.")
        except Exception as e:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")

        try:
            mne_obj, obj_type = load_file(vhdr_path, "brainvision")
        except HTTPException:
            cleanup_session(session_id)
            raise
        except Exception as e:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to load BrainVision files: {str(e)}")

        try:
            capabilities = detect_capabilities(mne_obj)
        except Exception as e:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to detect capabilities: {str(e)}")

        return _build_full_response(session_id, "brainvision", capabilities)

    # ── BIDS ZIP already handled by upload_single (zip format)
    # ── tabular_multi → not supported yet, placeholder
    raise HTTPException(
        status_code=400,
        detail=f"Multi-file format '{multi_format}' not yet supported. Use single file upload for CSV/TSV."
    )


@router.post("/upload/configure", response_model=UploadResponse)
async def configure_upload(config: ColumnMapping):
    """
    Second round trip after user submits column picker (CSV/TSV) or variable picker (MAT).
    Loads the file with the user's mapping, runs detect_capabilities, returns full response.
    Frontend goes straight to analysis after this.
    """
    session_dir = TEMP_BASE / config.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    # find the saved file
    files = [f for f in session_dir.iterdir() if f.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No file found for this session.")
    file_path = files[0]

    file_format = detect_format(file_path.name)

    # load with user mapping
    try:
        if file_format == "csv":
            mne_obj = load_csv(file_path, config)
        elif file_format == "tsv":
            mne_obj = load_tsv(file_path, config)
        elif file_format == "mat":
            # config is ColumnMapping but load_mat needs VariableMapping
            # frontend should POST to a separate /upload/configure-mat endpoint
            # for now raise a clear error
            raise HTTPException(
                status_code=400,
                detail="MAT files use /upload/configure-mat endpoint."
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unexpected format in configure: {file_format}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load file with mapping: {str(e)}")

    try:
        capabilities = detect_capabilities(mne_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect capabilities: {str(e)}")

    return _build_full_response(config.session_id, file_format, capabilities)


@router.post("/upload/configure-mat", response_model=UploadResponse)
async def configure_mat(config: VariableMapping):
    """
    Same as configure_upload but for MAT files which use VariableMapping not ColumnMapping.
    """
    session_dir = TEMP_BASE / config.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    files = [f for f in session_dir.iterdir() if f.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No file found for this session.")
    file_path = files[0]

    try:
        mne_obj = load_mat(file_path, config)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load MAT file: {str(e)}")

    try:
        capabilities = detect_capabilities(mne_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect capabilities: {str(e)}")

    return _build_full_response(config.session_id, "mat", capabilities)


@router.get("/upload/validate/{session_id}")
async def validate_session(session_id: str):
    session_dir = TEMP_BASE / session_id
    return {"valid": session_dir.exists()}



@router.post("/upload/bids", response_model=UploadResponse)
async def upload_bids(file: UploadFile = File(...)):
    """
    Accepts a ZIP file containing a BIDS dataset.
    Extracts it, scans for available subjects/sessions/tasks/runs,
    returns them for the frontend to show a selection UI.
    """
    session_id = str(uuid.uuid4())

    try:
        zip_path = await save_temp_file(file, session_id)
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to save ZIP: {str(e)}")

    try:
        bids_root = zip_handler(zip_path, session_id)
    except HTTPException:
        cleanup_session(session_id)
        raise

    try:
        entities = scan_bids_entities(bids_root)
    except HTTPException:
        cleanup_session(session_id)
        raise

    return UploadResponse(
        session_id=session_id,
        format="bids",
        n_channels=0,
        channels=[],
        sampling_rate=0.0,
        duration_seconds=0.0,
        events_found=False,
        event_classes=[],
        available_modes=[],
        recommended_mode="basic",
        needs_column_picker=False,
        needs_variable_picker=False,
        needs_event_setup=False,
        is_ready=False,
        source="upload",
        bids_entities=entities,
        needs_bids_selection=True,
    )
    
    
    
@router.post("/upload/bids-configure", response_model=UploadResponse)
async def configure_bids(selection: BidsSelection):
    """
    Second round trip after user picks subject/session/task/run
    from the BIDS selection UI.
    """
    external_root = get_external_bids_root(selection.session_id)
    
    if external_root:
        bids_root = external_root
    else:
        bids_root = TEMP_BASE / selection.session_id

        # Handle the possible nested-folder case from zip_handler
        subdirs = [p for p in bids_root.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and not any(bids_root.glob("sub-*")):
            bids_root = subdirs[0]

        if not bids_root.exists():
            raise HTTPException(status_code=404, detail="Session not found or expired.")

    try:
        raw = load_bids(
            bids_root,
            subject=selection.subject,
            task=selection.task,
            session=selection.session,
            run=selection.run,
        )
    except HTTPException:
        raise

    capabilities = detect_capabilities(raw)

    return _build_full_response(selection.session_id, "bids", capabilities)


# ── Dataset routes (stubs for later) ─────────────────────────────────────────

@router.get("/dataset/list/{source}")
async def list_datasets(source: str):
    if source == "mne":
        return {"datasets": list_mne_datasets()}
    elif source == "moabb":
        return {"datasets": list_moabb_datasets()}
    elif source == "openneuro":
        return {"datasets": [], "note": "OpenNeuro requires a specific accession ID, not a list."}
    elif source == "braindecode":
        return {"datasets": list_braindecode_datasets()}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")



@router.post("/dataset/fetch")
async def fetch_dataset(request: DatasetFetchRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())

    try:       
        if request.source == "mne" and request.dataset_name == "erp_core":
            result = fetch_mne_dataset("erp_core", request.subjects)
            entities = result["entities"]
            register_external_bids_root(session_id, result["bids_root"])

            return UploadResponse(
                session_id=session_id,
                format="bids",
                n_channels=0,
                channels=[],
                sampling_rate=0.0,
                duration_seconds=0.0,
                events_found=False,
                event_classes=[],
                available_modes=[],
                recommended_mode="basic",
                needs_column_picker=False,
                needs_variable_picker=False,
                needs_event_setup=False,
                is_ready=False,
                source="mne",
                bids_entities=entities,
                needs_bids_selection=True,
            )

        elif request.source == "mne":
            raw = fetch_mne_dataset(request.dataset_name, request.subjects, request.runs)

        elif request.source == "moabb":
            raw = fetch_moabb_dataset(request.dataset_name, request.subjects)

        elif request.source == "openneuro":
            bids_root = fetch_openneuro_dataset(request.dataset_name, session_id)
            entities = scan_bids_entities(bids_root)
            register_external_bids_root(session_id, bids_root)

            return UploadResponse(
                session_id=session_id,
                format="bids",
                n_channels=0,
                channels=[],
                sampling_rate=0.0,
                duration_seconds=0.0,
                events_found=False,
                event_classes=[],
                available_modes=[],
                recommended_mode="basic",
                needs_column_picker=False,
                needs_variable_picker=False,
                needs_event_setup=False,
                is_ready=False,
                source="openneuro",
                bids_entities=entities,
                needs_bids_selection=True,
            )

        elif request.source == "braindecode":
            raw = fetch_braindecode_dataset(request.dataset_name, request.subjects)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {request.source}")

    except HTTPException:
        cleanup_session(session_id)
        raise
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset: {str(e)}")

    try:
        capabilities = detect_capabilities(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect capabilities: {str(e)}")

    return _build_full_response(session_id, request.dataset_name, capabilities, source=request.source)

@router.get("/dataset/fetch/progress/{session_id}")
async def fetch_progress(session_id: str):
    pass



# ── Preprocessing routes ─────────────────────────────────────────

@router.get("/preprocessing/preset")
async def get_preprocessing_preset(dataset_name: Optional[str] = None, mode: str = "basic", task: Optional[str] = None):
    from backend.core.presets import get_preset
    return get_preset(dataset_name, mode, task)