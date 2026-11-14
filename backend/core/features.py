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

        elif mode == "erp":
            return extract_erp_features(epochs)

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

    # log10, for two reasons:
    # 1. Raw band power is ~1e-12 V^2/Hz. sklearn's trees treat any feature whose
    #    range is <= FEATURE_THRESHOLD = 1e-7 as constant (sklearn/tree/_partitioner.pxd,
    #    sklearn 1.9.0), so RF and Extra Trees saw 110 "constant" features and sat at
    #    chance (BNCI2014_001 subject 1: rf 0.248 -> 0.467, extra_trees 0.245 -> 0.467).
    # 2. Power is heavily right-skewed; the log makes it closer to normal, the same
    #    reason MNE's CSP log-transforms its average-power output ("apply a log
    #    transform to standardize features",
    #    https://mne.tools/1.12/generated/mne.decoding.CSP.html).
    # The floor guards flat channels, where power is 0 and log10 would give -inf.
    # raw_psds below stays in linear units — the PSD chart and topomaps do their own scaling.
    features = np.log10(np.maximum(stacked, 1e-30)).reshape(n_epochs, -1)

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
    # Microvolts, not volts. MNE stores EEG in volts (SI units), so variance comes out
    # around 1e-10 V^2 — below sklearn's tree split threshold FEATURE_THRESHOLD = 1e-7
    # (sklearn/tree/_partitioner.pxd, 1.9.0), which made trees treat every variance
    # feature as constant (eegbci s1: median variance range 3.9e-10). Same failure
    # as band power in spectral mode; see NEUROSCOPE_CONTEXT.md section 6.
    data = epochs.get_data(copy=False) * 1e6  # [n_epochs, n_channels, n_times], µV

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

ERP_BIN_MS = 50.0


def extract_erp_features(epochs: mne.Epochs, bin_ms: float = ERP_BIN_MS) -> Tuple[np.ndarray, dict]:
    """
    Event-related potential features: the mean voltage in consecutive time bins, per
    channel, across the epoch.

    An ERP (P300, N400, ERN…) is a time-locked voltage WAVEFORM, so its information sits
    in the signal's shape over time — which band power and CSP (both power measures)
    throw away. Averaging within short intervals and classifying the resulting
    spatio-temporal vector with shrinkage LDA is the standard single-trial approach in
    Blankertz, Lemm, Treder, Haufe & Müller (2011), "Single-trial analysis and
    classification of ERP components — a tutorial", NeuroImage 56(2):814–825.
    The paper picks intervals by inspecting the class difference; EQUAL-WIDTH bins of
    `bin_ms` (default 50 ms) are this project's generic, data-agnostic version of that.

    Microvolts, not volts, for the same sklearn tree-threshold reason as basic mode.
    Output: [n_epochs, n_channels * n_bins].
    """
    data = epochs.get_data(copy=False) * 1e6                    # µV
    sfreq = float(epochs.info["sfreq"])
    per_bin = max(1, int(round(sfreq * bin_ms / 1000.0)))
    n_bins = data.shape[2] // per_bin
    trimmed = data[:, :, : n_bins * per_bin]
    binned = trimmed.reshape(data.shape[0], data.shape[1], n_bins, per_bin).mean(axis=3)
    features = binned.reshape(data.shape[0], -1)

    starts = epochs.times[: n_bins * per_bin : per_bin]
    feature_names = [f"{ch}_{t * 1000:+.0f}ms" for ch in epochs.ch_names for t in starts]
    return features, {
        "feature_names": feature_names,
        "n_bins": int(n_bins),
        "bin_ms": float(per_bin / sfreq * 1000.0),
    }
