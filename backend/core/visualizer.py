import base64
import io
from typing import Optional, Tuple

import matplotlib
# Agg is a headless backend. Must be set before pyplot is imported, or
# matplotlib tries to open a window and hangs in a worker thread.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import numpy as np


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def band_topomaps(
    epochs: mne.BaseEpochs,
    y: np.ndarray,
    class_names: list,
    psds: np.ndarray,
    freqs: np.ndarray,
    bands: dict,
    passband: Optional[tuple] = None,
) -> Optional[dict]:
    """
    One scalp map per class, for each frequency band, as base64 PNGs.

    psds is (n_epochs, n_channels, n_freqs) from extract_band_power_features.
    Returns None when the epochs carry no electrode positions — there's
    nothing to interpolate between, and plot_topomap would raise.

    Bands outside the data's passband are skipped: a delta map of data
    bandpassed at 8 Hz shows filter rolloff, not brain activity.
    """
    if epochs.get_montage() is None:
        return None

    codes = sorted(set(y.tolist()))
    out = {}

    for band, (lo, hi) in bands.items():
        if passband and (hi <= passband[0] or lo >= passband[1]):
            continue

        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            continue

        # mean power in this band, per class, per channel -> (n_classes, n_chans)
        per_class = np.array([
            psds[y == code][:, :, mask].mean(axis=(0, 2)) for code in codes
        ])
        # dB relative to the grand mean across classes, so the colour scale
        # shows where classes DIFFER rather than overall head-wide power
        grand = per_class.mean(axis=0, keepdims=True)
        rel = 10 * np.log10(per_class / (grand + 1e-30) + 1e-30)
        vlim = float(np.abs(rel).max()) or 1.0

        fig, axes = plt.subplots(
            1, len(codes), figsize=(2.1 * len(codes), 2.6), layout="constrained"
        )
        axes = np.atleast_1d(axes)

        im = None
        for ax, name, values in zip(axes, class_names, rel):
            im, _ = mne.viz.plot_topomap(
                values, epochs.info, axes=ax, show=False,
                cmap="RdBu_r", vlim=(-vlim, vlim),
                contours=4, extrapolate="head", sphere="auto",
            )
            ax.set_title(str(name).replace("_", " "), fontsize=9)   # "left_hand" -> "left hand"

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
            cbar.set_label("dB vs. mean", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        out[band] = {
            "image": _fig_to_base64(fig),
            "range_hz": [float(lo), float(hi)],
        }

    return out or None


def spectral_summary(
    epochs: mne.BaseEpochs,
    y: np.ndarray,
    class_names: list,
    bands: dict,
    psds: Optional[np.ndarray] = None,
    freqs: Optional[np.ndarray] = None,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Class-averaged PSD curves and band topomaps for the results page, in EVERY mode.

    The spectrum describes the data, not the feature method, so it shouldn't
    depend on whether the classifiers used CSP, band power or statistics.
    Spectral mode already computed the PSD for its features and passes it in;
    other modes compute it here with the same settings, so all modes show
    identical plots for identical data.

    Never raises. Visuals are optional; losing a five-minute EEGNet run
    because a plot failed would be far worse than a missing chart. A failed
    PSD gives (None, None); a failed topomap leaves the PSD in place.
    """
    try:
        if psds is None or freqs is None:
            # Same call as features.extract_band_power_features, so spectral and
            # non-spectral runs produce identical curves.
            # https://mne.tools/1.12/generated/mne.Epochs.html#mne.Epochs.compute_psd
            spectrum = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
            psds, freqs = spectrum.get_data(return_freqs=True)

        passband = [
            float(epochs.info.get("highpass") or 0),
            float(epochs.info.get("lowpass") or epochs.info["sfreq"] / 2),
        ]
        psd_data = {
            "freqs": [float(f) for f in freqs],
            "curves": {
                # log10 so the 1/f falloff doesn't flatten everything above alpha
                name: [float(v) for v in np.log10(psds[y == code].mean(axis=(0, 1)) + 1e-20)]
                for code, name in zip(sorted(set(y.tolist())), class_names)
            },
            "bands": {k: list(v) for k, v in bands.items()},
            "passband": passband,
        }
    except Exception:
        return None, None   # visuals are optional — never fail the analysis over a plot

    try:
        topomaps = band_topomaps(epochs, y, class_names, psds, freqs, bands,
                                 passband=tuple(passband))
    except Exception:
        topomaps = None     # e.g. odd channel positions; the PSD chart is still useful

    return psd_data, topomaps