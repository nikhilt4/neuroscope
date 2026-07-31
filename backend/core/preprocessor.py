import mne
import numpy as np
from typing import Optional, Union
from fastapi import HTTPException


def _apply_montage(raw: mne.io.Raw, montage: Optional[str]) -> mne.io.Raw:
    """
    Attaches electrode positions. Needed for topomaps and any position-aware
    step later; harmless for classification itself.
    on_missing="warn" so a few unmatched channel names degrade gracefully
    rather than killing the whole run.
    """
    if not montage:
        return raw
    try:
        raw.set_montage(montage, on_missing="warn", verbose=False)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not apply montage '{montage}': {str(e)}"
        )
    return raw


def _rename_annotations(raw: mne.io.Raw, mapping: Optional[dict]) -> mne.io.Raw:
    """
    Renames raw annotation descriptions (e.g. eegbci's T1/T2 -> left_hand/right_hand).
    Only renames keys actually present — MNE raises on unknown descriptions,
    and a preset shouldn't be able to break a file it partially matches.
    """
    if not mapping:
        return raw
    if any(isinstance(v, dict) for v in mapping.values()):
        raise HTTPException(
            status_code=500,
            detail="annotation_rename is still nested — call resolve_annotation_rename() first."
        )
    if raw.annotations is None or len(raw.annotations) == 0:
        return raw

    present = set(raw.annotations.description)
    applicable = {k: v for k, v in mapping.items() if k in present}
    if applicable:
        raw.annotations.rename(applicable)
    return raw


def _extract_events(raw: mne.io.Raw, include_events: Optional[list] = None):
    """
    Pulls events from a stim channel if present, else from annotations.
    include_events restricts to named classes — this is what keeps 'rest'
    out of an otherwise-binary motor imagery task.
    """
    events = None
    try:
        events = mne.find_events(raw, verbose=False)
    except Exception:
        events = None

    if events is not None and len(events) > 0:
        event_id = {str(uid): int(uid) for uid in sorted(set(events[:, 2]))}
        if include_events:
            event_id = {k: v for k, v in event_id.items() if k in include_events}
            if not event_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"None of {include_events} found in stim channel. "
                           f"Available codes: {sorted(set(events[:, 2]))}"
                )
            keep = set(event_id.values())
            events = events[np.isin(events[:, 2], list(keep))]
        return events, event_id

    available = sorted(set(raw.annotations.description)) if raw.annotations is not None else []
    try:
        if include_events:
            requested = {name: i + 1 for i, name in enumerate(include_events)}
            return mne.events_from_annotations(raw, event_id=requested, verbose=False)
        return mne.events_from_annotations(raw, verbose=False)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract events: {str(e)}. Available annotations: {available}"
        )


def _crop_epochs(epochs: mne.Epochs, crop: Optional[tuple]) -> mne.Epochs:
    """
    Restricts each epoch to an analysis window inside the epoching window.
    For eegbci this is (1.0, 2.0) — excludes the cue-evoked response so CSP
    sees imagery-period variance rather than the visual response.
    """
    if not crop:
        return epochs
    tmin, tmax = crop
    if tmin < epochs.tmin or tmax > epochs.tmax:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis window ({tmin}, {tmax}) falls outside the epoch "
                   f"window ({epochs.tmin}, {epochs.tmax})."
        )
    return epochs.crop(tmin=tmin, tmax=tmax)




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
        
        raw = _apply_montage(raw, config.get("montage"))                    # NEW
        raw = _rename_annotations(raw, config.get("annotation_rename"))     # NEW

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

        epochs = _epoch_raw(
            raw, tmin, tmax, baseline,
            events=events, event_id=event_id,
            include_events=config.get("include_events"),                    # NEW
            picks=config.get("picks", "eeg")                               # NEW
        )
        
        
        epochs = _crop_epochs(epochs, config.get("epoch_train_crop"))

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
               a single channel name string (e.g. "Cz"),
               or a list of channel names (e.g. ["P9", "P10"]).
    """
    if reference == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    elif isinstance(reference, str):
        raw.set_eeg_reference(ref_channels=[reference], verbose=False)
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
    raw, tmin, tmax, baseline,
    events=None, event_id=None,
    include_events=None,
    picks="eeg",
) -> mne.Epochs:
    if events is None or event_id is None:
        events, event_id = _extract_events(raw, include_events)
    elif include_events:
        event_id = {k: v for k, v in event_id.items() if k in include_events}
        if not event_id:
            raise HTTPException(status_code=400, detail=f"None of {include_events} in supplied event_id.")
        events = events[np.isin(events[:, 2], list(event_id.values()))]

    if len(events) == 0:
        raise HTTPException(
            status_code=400,
            detail="No events found — cannot epoch this data. Check event/annotation setup."
        )

    return mne.Epochs(
        raw, events=events, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=baseline,
        picks=picks,          # keeps EOG/ECG/STI out of the feature pipeline
        preload=True, verbose=False,
    )