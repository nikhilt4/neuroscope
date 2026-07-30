import mne




def preprocess_raw(raw: mne.io.Raw, config: PreprocessingConfig, events: np.ndarray = None, event_id: dict = None) -> mne.Epochs:
    """
    Main entry point. Runs the full pipeline on a Raw object:
    bandpass -> notch -> reference -> ICA -> epoch -> baseline.
    Returns Epochs ready for feature extraction.
    """
    
    

def _apply_bandpass(raw: mne.io.Raw, low: float, high: float) -> mne.io.Raw:
    """raw.filter(l_freq=low, h_freq=high, fir_design='firwin')"""

def _apply_notch(raw: mne.io.Raw, freq: float) -> mne.io.Raw:
    """raw.notch_filter(freqs=freq)"""

def _apply_reference(raw: mne.io.Raw, reference) -> mne.io.Raw:
    """
    reference can be: "average" | "none" | list of channel names (e.g. ["P9","P10"])
    raw.set_eeg_reference(...)
    """

def _apply_ica(raw: mne.io.Raw, n_components: int) -> mne.io.Raw:
    """
    mne.preprocessing.ICA — fit, auto-detect eye/muscle artifacts, exclude, apply
    """

def _epoch_raw(raw: mne.io.Raw, tmin: float, tmax: float, baseline, events=None, event_id=None) -> mne.Epochs:
    """
    mne.Epochs — uses provided events/event_id if given (from detect_capabilities),
    otherwise falls back to events_from_annotations
    """