import mne
import numpy as np
from typing import Optional, Tuple, Union
from fastapi import HTTPException
from mne.decoding import CSP


def extract_features(
    epochs: mne.Epochs,
    mode: str,
    labels: Optional[np.ndarray] = None,
    n_csp_components: int = 4,
    csp_reg: Optional[Union[str, float]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Main entry point. Routes to the correct feature extraction method based on mode.

    Returns:
        features: np.ndarray, shape [n_epochs x n_features]
        extras: dict — mode-specific extra objects needed downstream
                (e.g. the fitted CSP object for motor_imagery, band names for spectral)
    """
    try:
        if mode == "motor_imagery":
            if labels is None:
                raise HTTPException(
                    status_code=400,
                    detail="Motor imagery feature extraction requires labels (event classes)."
                )
            return extract_csp_features(epochs, labels, n_csp_components, reg=csp_reg)

        elif mode == "spectral":
            return extract_band_power_features(epochs)

        elif mode == "basic":
            return extract_basic_features(epochs)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode for feature extraction: {mode}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


def extract_csp_features(
    epochs: mne.Epochs,
    labels: np.ndarray,
    n_components: int = 4,
    reg: Optional[Union[str, float]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Common Spatial Patterns — the standard feature extraction method for motor imagery.
    Learns spatial filters that maximize variance difference between classes.

    reg: regularization method for covariance estimation — None, "ledoit_wolf",
         "oas", or a float shrinkage value between 0 and 1. Helps when there
         are few trials relative to channels (reduces overfitting in CSP).

    Note: CSP is a SUPERVISED transform — it needs labels to fit. In a real
    train/test pipeline this should be fit only on training data (inside CV folds),
    not on the full dataset, to avoid data leakage. This function fits on
    everything passed in; the caller (classifiers.py) is responsible for
    wrapping this correctly inside cross-validation.
    """
    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    features = csp.fit_transform(epochs.get_data(copy=False), labels)

    extras = {
        "csp_object": csp,
        "n_components": n_components,
        "reg": reg,
        "feature_names": [f"csp_{i}" for i in range(n_components)]
    }

    return features, extras


# Standard EEG frequency bands used for spectral/band-power analysis
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


def extract_band_power_features(epochs: mne.Epochs) -> Tuple[np.ndarray, dict]:
    """
    Computes average power in standard EEG frequency bands, per channel, per epoch.
    Used for spectral mode — general frequency-domain analysis (not class-specific).

    Output feature vector per epoch: [n_channels x n_bands] flattened.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data(copy=False)  # shape: [n_epochs, n_channels, n_times]
    n_epochs, n_channels, n_times = data.shape

    # Compute power spectral density using Welch's method via MNE's Epochs API
    spectrum = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)  # psds shape: [n_epochs, n_channels, n_freqs]

    band_powers = []
    band_names_ordered = list(FREQUENCY_BANDS.keys())

    for band_name in band_names_ordered:
        low, high = FREQUENCY_BANDS[band_name]
        freq_mask = (freqs >= low) & (freqs <= high)
        # Average power within this band, per epoch per channel
        band_power = psds[:, :, freq_mask].mean(axis=2)  # shape: [n_epochs, n_channels]
        band_powers.append(band_power)

    # Stack into [n_epochs, n_channels, n_bands], then flatten to [n_epochs, n_channels * n_bands]
    stacked = np.stack(band_powers, axis=2)
    features = stacked.reshape(n_epochs, -1)

    feature_names = [
        f"{ch}_{band}"
        for ch in epochs.ch_names
        for band in band_names_ordered
    ]

    extras = {
        "band_names": band_names_ordered,
        "channel_names": epochs.ch_names,
        "feature_names": feature_names,
        "raw_psds": psds,
        "freqs": freqs
    }

    return features, extras


def extract_basic_features(epochs: mne.Epochs) -> Tuple[np.ndarray, dict]:
    """
    Simple statistical features per channel, per epoch — used for basic mode
    when no specific analysis paradigm applies. Not intended to be highly
    discriminative, just a reasonable default so basic mode still produces
    something classifiers can run on.

    Features per channel: mean, std, variance, min, max, peak-to-peak.
    """
    data = epochs.get_data(copy=False)  # [n_epochs, n_channels, n_times]

    means = data.mean(axis=2)
    stds = data.std(axis=2)
    variances = data.var(axis=2)
    mins = data.min(axis=2)
    maxs = data.max(axis=2)
    ptp = maxs - mins

    features = np.concatenate([means, stds, variances, mins, maxs, ptp], axis=1)

    stat_names = ["mean", "std", "var", "min", "max", "ptp"]
    feature_names = [
        f"{ch}_{stat}"
        for stat in stat_names
        for ch in epochs.ch_names
    ]

    extras = {
        "feature_names": feature_names,
        "stats_computed": stat_names
    }

    return features, extras