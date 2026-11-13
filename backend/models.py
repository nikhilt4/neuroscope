from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Tuple


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
    is_epochs: bool = False
    dataset_name: Optional[str] = None
    # channels typed EEG — what decoders actually see (Sleep-EDF: 7 channels, 2 EEG)
    n_eeg_channels: Optional[int] = None
    # class names the analysis will use after the preset's renames/filters
    # (eegbci events are T0/T1/T2 but the run classifies left_hand/right_hand)
    analysis_classes: Optional[List[str]] = None

    
class BidsSelection(BaseModel):
    session_id: str
    subject: str
    task: Optional[str] = None
    session: Optional[str] = None
    run: Optional[str] = None


class PreprocessingConfig(BaseModel):
    """
    All fields optional — omitted ones fall through to the dataset/mode preset.
    exclude_unset=True in the route means only what the user actually touched
    overrides the preset.
    """
    bandpass_low: Optional[float] = None
    bandpass_high: Optional[float] = None
    notch_enabled: Optional[bool] = None
    notch_freq: Optional[float] = None
    reference: Optional[Union[str, List[str]]] = None
    ica_enabled: Optional[bool] = None
    ica_components: Optional[int] = None
    epoch_tmin: Optional[float] = None
    epoch_tmax: Optional[float] = None
    baseline_enabled: Optional[bool] = None
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None
    epoch_train_crop: Optional[Tuple[float, float]] = None
    include_events: Optional[List[str]] = None
    montage: Optional[str] = None
    picks: Optional[str] = None


class ClassifierConfig(BaseModel):
    selected: List[str] = Field(default_factory=lambda: ["lda", "svm"])
    n_folds: int = Field(5, ge=2, le=20)
    csp_components: int = Field(4, ge=1, le=32)
    csp_reg: Optional[Union[str, float]] = None
    # keyed by classifier name; each builder applies its own defaults for
    # anything absent, so this can be {} for a plain run
    configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    session_id: str
    mode: str = "motor_imagery"
    preprocessing: Optional[PreprocessingConfig] = None
    classifiers: ClassifierConfig = Field(default_factory=ClassifierConfig)