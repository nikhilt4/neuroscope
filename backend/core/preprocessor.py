import mne
import numpy as np
from typing import Optional, Union
from fastapi import HTTPException


def preprocess_raw(
    raw: mne.io.Raw,
    config: dict,
    events: Optional[np.ndarray] = None,
    event_id: Optional[dict] = None,
) -> mne.Epochs:
    """
    Main entry point. Runs the full pipeline on a Raw object:
    bandpass -> notch -> reference -> ICA -> epoch -> baseline.
    Returns Epochs ready for feature extraction.

    config is a dict matching the shape of DATASET_PRESETS/MODE_PRESETS entries
    (bandpass_low, bandpass_high, notch_enabled, notch_freq, reference,
    ica_enabled, ica_components, epoch_tmin, epoch_tmax, baseline_enabled, baseline)
    """
    try:
        raw = raw.copy()  # never mutate the caller's original Raw object

        # 1. Bandpass filter
        bandpass_low = config.get("bandpass_low")
        bandpass_high = config.get("bandpass_high")
        if bandpass_low is not None or bandpass_high is not None:
            raw = _apply_bandpass(raw, bandpass_low, bandpass_high)

        # 2. Notch filter
        if config.get("notch_enabled"):
            notch_freq = config.get("notch_freq", 50.0)
            raw = _apply_notch(raw, notch_freq)

        # 3. Re-reference
        reference = config.get("reference")
        if reference and reference != "none":
            raw = _apply_reference(raw, reference)

        # 4. ICA artifact removal
        if config.get("ica_enabled"):
            n_components = config.get("ica_components", 20)
            raw = _apply_ica(raw, n_components)

        # 5. Epoch
        tmin = config.get("epoch_tmin", -0.2)
        tmax = config.get("epoch_tmax", 0.8)
        baseline = config.get("baseline") if config.get("baseline_enabled") else None

        epochs = _epoch_raw(raw, tmin, tmax, baseline, events=events, event_id=event_id)

        return epochs

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


def _apply_bandpass(raw: mne.io.Raw, low: Optional[float], high: Optional[float]) -> mne.io.Raw:
    """
    Applies a bandpass, highpass, or lowpass filter depending on which bounds are given.
    MNE's raw.filter() natively supports l_freq/h_freq individually being None.
    """
    raw.filter(l_freq=low, h_freq=high, fir_design="firwin", verbose=False)
    return raw


def _apply_notch(raw: mne.io.Raw, freq: float) -> mne.io.Raw:
    """
    Applies a notch filter at the given frequency (and its harmonics up to Nyquist)
    to remove electrical line noise.
    """
    nyquist = raw.info["sfreq"] / 2
    freqs = np.arange(freq, nyquist, freq)  # freq, 2*freq, 3*freq... up to Nyquist
    raw.notch_filter(freqs=freqs, verbose=False)
    return raw


def _apply_reference(raw: mne.io.Raw, reference: Union[str, list]) -> mne.io.Raw:
    """
    Re-references the EEG data.
    reference: "average" for common average reference,
               or a list of channel names (e.g. ["P9", "P10"]) for a specific reference.
    """
    if reference == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    elif isinstance(reference, list):
        raw.set_eeg_reference(ref_channels=reference, verbose=False)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported reference type: {reference}")
    return raw


def _apply_ica(raw: mne.io.Raw, n_components: int) -> mne.io.Raw:
    """
    Fits ICA and automatically excludes components correlated with EOG artifacts.
    Falls back to no exclusion if no EOG channels are present (rather than guessing).
    """
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, verbose=False)
    ica.fit(raw, verbose=False)

    eog_channels = mne.pick_types(raw.info, eog=True)
    if len(eog_channels) > 0:
        eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices
    # If no EOG channels exist, we don't guess — ica.exclude stays empty.
    # This is a conservative choice: better to keep a possibly-noisy component
    # than to remove a genuine brain signal component based on a weak heuristic.

    raw = ica.apply(raw.copy(), verbose=False)
    return raw


def _epoch_raw(
    raw: mne.io.Raw,
    tmin: float,
    tmax: float,
    baseline: Optional[tuple],
    events: Optional[np.ndarray] = None,
    event_id: Optional[dict] = None,
) -> mne.Epochs:
    """
    Epochs the continuous Raw data around events.
    Uses provided events/event_id if given; otherwise extracts them internally
    as a fallback (tries find_events first, then events_from_annotations).
    """
    if events is None or event_id is None:
        try:
            events = mne.find_events(raw, verbose=False)
            unique_ids = sorted(set(events[:, 2]))
            event_id = {str(uid): uid for uid in unique_ids}
        except Exception:
            events, event_id = mne.events_from_annotations(raw, verbose=False)

    if len(events) == 0:
        raise HTTPException(
            status_code=400,
            detail="No events found — cannot epoch this data. Check event/annotation setup."
        )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    return epochs