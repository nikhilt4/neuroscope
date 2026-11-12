import base64
import io
from typing import Optional

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
            ax.set_title(name, fontsize=9)

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
            cbar.set_label("dB vs. mean", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        out[band] = {
            "image": _fig_to_base64(fig),
            "range_hz": [float(lo), float(hi)],
        }

    return out or None