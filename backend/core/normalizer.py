@dataclass
class NeuroscopeData:
    raw: mne.io.Raw | None
    epochs: mne.Epochs | None
    events: np.ndarray | None
    event_id: dict
    metadata: dict
    source: str
    session_id: str
    is_ready: bool

normalize(
    obj: mne.io.Raw | mne.Epochs,
    capabilities: dict,
    source: str,
    session_id: str
) → NeuroscopeData

    Steps:
    1. resample to 250Hz if sampling_rate != 250
    2. set standard montage where channel names match 10-20
    3. extract events if Raw, read from obj if Epochs
    4. populate NeuroscopeData
    5. save as FIF to /tmp/neuroscope/{session_id}/data.fif
    6. return NeuroscopeData

save_session(data: NeuroscopeData) → None
    serializes to disk — Raw/Epochs as .fif, metadata as .json

load_session(session_id: str) → NeuroscopeData
    loads from disk — used by Part 2 to pick up where Part 1 left off