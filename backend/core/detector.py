import mne
import numpy as np
from typing import Union
from typing import List, Union

montage = mne.channels.make_standard_montage('standard_1005')
ALL_MONTAGE_CHS = {ch.lower() for ch in montage.ch_names}

MOTOR_PRIMARY   = {"c3", "c4", "cz"} & ALL_MONTAGE_CHS
MOTOR_EXTENDED  = {"cp3", "cp4", "cpz", "fc3", "fc4", "fcz", "c1", "c2", "c5", "c6"} & ALL_MONTAGE_CHS
FRONTAL         = {"f3", "f4", "fz", "f7", "f8", "fc1", "fc2"} & ALL_MONTAGE_CHS
PARIETAL        = {"p3", "p4", "pz", "cp1", "cp2"} & ALL_MONTAGE_CHS
OCCIPITAL       = {"o1", "o2", "oz"} & ALL_MONTAGE_CHS
TEMPORAL        = {"t7", "t8", "tp7", "tp8"} & ALL_MONTAGE_CHS


def detect_capabilities(obj: mne.io.Raw | mne.Epochs) -> dict:
    
    is_epochs = isinstance(obj, mne.Epochs)
    ch_names = obj.ch_names
    sampling_freq = obj.info["sfreq"]
    
    if is_epochs:
        total_epochs_duration = len(obj) * (obj.tmax - obj.tmin)
    else:
        total_epochs_duration = obj.n_times / sampling_freq
        
    events_found, event_classes = extract_events(obj)
    motor_info = check_motor_channels(ch_names)
    has_standard_montage = check_standard_montage(ch_names)
    available_modes = get_available_modes(ch_names, events_found, motor_info["motor_imagery_possible"])

    recommended_mode = get_recommended_mode(available_modes)

    return {
    "channels": ch_names,
    "n_channels": len(ch_names),
    "sampling_rate": sampling_freq,
    "duration_seconds": total_epochs_duration,
    "n_epochs": len(obj) if is_epochs else None,
    "has_standard_montage": has_standard_montage,
    "motor_channels": motor_info,          # full breakdown for frontend display
    "events_found": events_found,
    "event_classes": event_classes,
    "n_classes": len(event_classes),
    "available_modes": available_modes,
    "recommended_mode": recommended_mode,
    "needs_event_setup": not events_found and not is_epochs,
    "is_epochs": is_epochs
}

def check_motor_channels(ch_names: List[str]) -> bool:
    """
    Checks if both C3 and C4 channels are present (case insensitive).
    
    Returns:
        bool: True if both C3 and C4 are present, False otherwise.
    """

    ch_lower = {ch.lower() for ch in ch_names}

    found_primary  = sorted(MOTOR_PRIMARY & ch_lower)
    found_extended = sorted(MOTOR_EXTENDED & ch_lower)
    found_frontal  = sorted(FRONTAL & ch_lower)
    found_parietal = sorted(PARIETAL & ch_lower)
    found_occipital = sorted(OCCIPITAL & ch_lower)
    found_temporal = sorted(TEMPORAL & ch_lower)

    return {
        "motor_imagery_possible": {"c3", "c4"}.issubset(ch_lower),
        "primary_motor": found_primary,
        "extended_motor": found_extended,
        "frontal": found_frontal,
        "parietal": found_parietal,
        "occipital": found_occipital,
        "temporal": found_temporal
    }
    
    
    
def check_standard_montage(ch_names: List[str]) -> bool:
    """
    Checks how many channel names match the standard 10-20 system.
    returns True if >50% match
    
    Returns:
        bool: True if more than 50% of channels match the standard 10-20 system, False otherwise.
    """
    
    
    data_channels = {ch.lower() for ch in ch_names}
    
    matched = data_channels.intersection(ALL_MONTAGE_CHS)
    
    return len(matched) / len(data_channels) > 0.5
    
    

def get_available_modes(ch_names: list, has_events: bool, motor_imagery_possible: bool) -> List[str]:
    """
    always includes "basic"
    adds "spectral" if 4+ standard channels
    adds "motor_imagery" if C3+C4 present AND has_events
    """

    modes = ["basic"]
    
    data_channels = {ch.lower() for ch in ch_names}
    matched = data_channels.intersection(ALL_MONTAGE_CHS)
    
    if len(matched) >= 4:
        modes.append("spectral")
    
    if motor_imagery_possible and has_events:
        modes.append("motor_imagery")
        
    return modes



def get_recommended_mode(available_modes) -> str:
    """
    returns highest capability mode available:
    "motor_imagery" > "spectral" > "basic"
    """
    
    for mode in ["motor_imagery", "spectral", "basic"]:
        if mode in available_modes:
            return mode
    return "basic"
    


def extract_events(obj: Union[mne.io.Raw, mne.Epochs]) -> tuple[bool, List[str]]:
    """
    for Raw: mne.find_events() or mne.events_from_annotations()
    for Epochs: obj.event_id.keys()
    returns (events_found: bool, class_names: List[str])
    """
    
    if isinstance(obj, mne.Epochs):
        return True, list(obj.event_id.keys())
    
    try:
        events = mne.find_events(obj, verbose=False)
        if len(events) > 0:
            unique_ids = np.unique(events[:, 2]).tolist()
            return True, [str(uid) for uid in unique_ids]
    except Exception:
        pass
    
    try:
        events, event_id = mne.events_from_annotations(obj, verbose=False)
        if len(events) > 0:
            return True, list(event_id.keys())
    except Exception:
        pass
    
    return False, []


def assign_best_montage(raw):
    candidates = ["standard_1005", "easycap-M1", "biosemi64"]
    best_montage, best_match_count = None, 0

    for name in candidates:
        montage = mne.channels.make_standard_montage(name)
        matched = set(ch.lower() for ch in raw.ch_names) & set(ch.lower() for ch in montage.ch_names)
        if len(matched) > best_match_count:
            best_montage, best_match_count = montage, len(matched)

    if best_montage:
        raw.set_montage(best_montage, on_missing="ignore", match_case=False)
    return raw

    
    