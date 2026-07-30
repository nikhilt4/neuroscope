from pydantic import BaseModel
from typing import Optional, List, Dict


class ColumnMapping(BaseModel):
    session_id: str
    eeg_columns: List[str]
    time_column: Optional[str] = None
    label_column: Optional[str] = None
    sampling_rate: Optional[float] = None
    event_mapping: Optional[Dict[str, int]] = None


class VariableMapping(BaseModel):
    session_id: str
    eeg_variable: str
    label_variable: Optional[str] = None
    sampling_rate: float
    data_shape: str


class DatasetFetchRequest(BaseModel):
    source: str
    dataset_name: str
    subjects: List[int]
    runs: Optional[List[int]] = None


class UploadResponse(BaseModel):
    session_id: str
    format: str
    n_channels: int
    channels: List[str]
    sampling_rate: float
    duration_seconds: float
    events_found: bool
    event_classes: List[str]
    available_modes: List[str]
    recommended_mode: str
    needs_column_picker: bool
    needs_variable_picker: bool
    needs_event_setup: bool
    is_ready: bool
    source: str
    column_info: Optional[dict] = None
    variable_info: Optional[dict] = None
    motor_channels: Optional[dict] = None
    n_epochs: Optional[int] = None
    bids_entities: Optional[List[dict]] = None
    needs_bids_selection: bool = False
    
    
class BidsSelection(BaseModel):
    session_id: str
    subject: str
    task: Optional[str] = None
    session: Optional[str] = None
    run: Optional[str] = None