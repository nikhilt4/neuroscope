import mne
import numpy as np
from typing import List, Optional, Union

montage = mne.channels.make_standard_montage('standard_1005')
ALL_MONTAGE_CHS = {ch.lower() for ch in montage.ch_names}

MOTOR_PRIMARY   = {"c3", "c4", "cz"} & ALL_MONTAGE_CHS
MOTOR_EXTENDED  = {"cp3", "cp4", "cpz", "fc3", "fc4", "fcz", "c1", "c2", "c5", "c6"} & ALL_MONTAGE_CHS
FRONTAL         = {"f3", "f4", "fz", "f7", "f8", "fc1", "fc2"} & ALL_MONTAGE_CHS
PARIETAL        = {"p3", "p4", "pz", "cp1", "cp2"} & ALL_MONTAGE_CHS
OCCIPITAL       = {"o1", "o2", "oz"} & ALL_MONTAGE_CHS
TEMPORAL        = {"t7", "t8", "tp7", "tp8"} & ALL_MONTAGE_CHS


def detect_capabilities(obj: mne.io.Raw | mne.Epochs) -> dict:
    
    is_epochs = isinstance(obj, mne.BaseEpochs)
    ch_names = obj.ch_names
    sampling_freq = float(obj.info["sfreq"])
    
    if is_epochs:
        total_epochs_duration = len(obj) * (obj.tmax - obj.tmin)
    else:
        total_epochs_duration = obj.n_times / sampling_freq
        
    events_found, event_classes = extract_events(obj)
    motor_info = check_motor_channels(ch_names)
    has_standard_montage = check_standard_montage(ch_names)
    # https://mne.tools/1.12/generated/mne.pick_types.html — count channels typed EEG
    n_eeg = len(mne.pick_types(obj.info, eeg=True, exclude=[]))
    available_modes = get_available_modes(ch_names, events_found, motor_info["motor_imagery_possible"],
                                          n_eeg_channels=n_eeg)

    recommended_mode = get_recommended_mode(available_modes, motor_info["motor_imagery_possible"])

    return {
    "channels": ch_names,
    "n_channels": len(ch_names),
    "n_eeg_channels": n_eeg,               # Sleep-EDF: 7 channels, 2 of them EEG
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
    
    

def get_available_modes(ch_names: list, has_events: bool, motor_imagery_possible: bool,
                        n_eeg_channels: Optional[int] = None) -> List[str]:
    """
    always includes "basic"
    adds "spectral" if there is at least one EEG channel (n_eeg_channels given), or —
        when channel types are unknown — 4+ standard 10-20 names (the old rule)
    adds "motor_imagery" if C3+C4 present AND has_events

    Why spectral no longer needs standard names: band power is computed per channel and
    needs no electrode positions; only the scalp maps do, and band_topomaps already
    returns None without a montage. The old 4-standard-names rule locked Sleep-EDF
    (two bipolar EEG channels, Fpz-Cz and Pz-Oz) out of the mode sleep staging uses.
    """

    modes = ["basic"]

    if n_eeg_channels is not None:
        spectral_ok = n_eeg_channels >= 1
    else:
        data_channels = {ch.lower() for ch in ch_names}
        spectral_ok = len(data_channels.intersection(ALL_MONTAGE_CHS)) >= 4
    if spectral_ok:
        modes.append("spectral")
    
    # CSP needs events and at least 2 EEG channels — not specific channel NAMES. Requiring
    # C3+C4 by name locked out BNCI2014_002, a motor imagery dataset whose 15 channels are
    # labelled EEG1–EEG15. NEUROSCOPE_CONTEXT.md 2.3 planned this: C3+C4 becomes a
    # recommendation (get_recommended_mode), not a hard requirement.
    if has_events:
        enough = n_eeg_channels >= 2 if n_eeg_channels is not None else motor_imagery_possible
        if enough:
            modes.append("motor_imagery")
        # ERP: time-locked voltage features need events and any EEG channel
        if n_eeg_channels is None or n_eeg_channels >= 1:
            modes.append("erp")

    return modes



# A public dataset's documented task beats guessing from channel names.
#   motor_imagery: BNCI2014_002's channels are EEG1–EEG15, so names alone never suggest CSP
#   ssvep/spectral: SSVEP data happens to include C3/C4, which made CSP the default for a
#                   flicker-frequency task — band power is what separates 12 Hz from 15 Hz
#   erp: erp_core/limo/kiloword include C3/C4, which made CSP the default for waveform data
TASK_MODE = {"motor_imagery": "motor_imagery", "ssvep": "spectral", "spectral": "spectral", "erp": "erp"}


def recommend_for_task(task: Optional[str], available_modes: list, default: str) -> str:
    """The mode a dataset's known task calls for, if available; otherwise the detector's pick."""
    preferred = TASK_MODE.get(task or "")
    return preferred if preferred in available_modes else default


def get_recommended_mode(available_modes, motor_channels_present: bool = True) -> str:
    """
    Recommends motor_imagery (CSP) only when the motor-cortex sites C3 and C4 are
    present — otherwise there's no evidence the data is lateralised motor activity,
    and CSP on, say, two sleep channels is a poor default. Then spectral, then basic.
    Callers that KNOW the task (a public dataset's metadata) can override this.
    """
    # erp is last in both: without a known ERP task there's no evidence the data is
    # time-locked evoked activity, so it's offered but never the automatic default
    order = (["motor_imagery", "spectral", "basic", "erp"] if motor_channels_present
             else ["spectral", "basic", "motor_imagery", "erp"])
    for mode in order:
        if mode in available_modes:
            return mode
    return "basic"
    


def extract_events(obj: Union[mne.io.Raw, mne.Epochs]) -> tuple[bool, List[str]]:
    """
    for Raw: mne.find_events() or mne.events_from_annotations()
    for Epochs: obj.event_id.keys()
    returns (events_found: bool, class_names: List[str])
    """
    
    if isinstance(obj, mne.BaseEpochs):
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

    
    