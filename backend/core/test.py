# import pyriemann
# print(pyriemann.__version__)
# import pyriemann.classification as c, pyriemann.estimation as e
# print([n for n in dir(c) if not n.startswith("_")])
# print([n for n in dir(e) if not n.startswith("_")])


# import mne
# import numpy as np
# from classifiers import run_classifier

# # 40 trials, 22 channels, 500 samples, 2 classes
# epochs =    # a real Epochs object
# y = epochs.events[:, 2]

# for name in ["lda", "riemann_mdm", "riemann_ts", "riemann_fgmdm"]:
#     print(name, run_classifier(name, "motor_imagery", {}, y=y, epochs_obj=epochs, cv_folds=3))


import numpy as np
import mne
from classifiers import run_classifier, run_all_classifiers

rng = np.random.default_rng(42)
N_PER_CLASS, N_CHANS, N_TIMES, SFREQ = 30, 8, 500, 250.0


def _make_trials(strong_ch: int, n: int) -> np.ndarray:
    """Noise trials where one channel carries 3x the variance."""
    d = rng.standard_normal((n, N_CHANS, N_TIMES)) * 1e-6
    d[:, strong_ch, :] *= 3.0
    return d


# class 1 -> channel 0 dominant, class 2 -> channel 1 dominant.
# A variance difference across channels is exactly what CSP and covariance-based
# Riemannian methods key on, so both should score well above chance.
data = np.concatenate([_make_trials(0, N_PER_CLASS), _make_trials(1, N_PER_CLASS)])
labels = np.array([1] * N_PER_CLASS + [2] * N_PER_CLASS)

info = mne.create_info([f"EEG{i:03d}" for i in range(N_CHANS)], SFREQ, "eeg")
events = np.column_stack([
    np.arange(len(labels)) * N_TIMES,
    np.zeros(len(labels), dtype=int),
    labels,
])
epochs = mne.EpochsArray(
    data, info, events=events,
    event_id={"left": 1, "right": 2},
    tmin=0.0, verbose=False,
)

y = epochs.events[:, 2]
print(f"epochs: {epochs.get_data(copy=False).shape}, classes: {np.unique(y)}\n")

# ── motor imagery: CSP-based + Riemannian, both consume 3D epochs ──
results = run_all_classifiers(
    selected_classifiers=["lda", "svm", "riemann_mdm", "riemann_ts", "riemann_fgmdm"],
    mode="motor_imagery",
    configs={},
    y=y,
    epochs_obj=epochs,
    cv_folds=3,
)
for r in results:
    if r.get("error"):
        print(f"  {r['classifier']:<16} FAILED — {r['error']}")
    else:
        print(f"  {r['classifier']:<16} {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}  [{r['category']}]")

# ── spectral: 2D feature matrix, no CSP, scaler applies ──
X = np.log(np.var(epochs.get_data(copy=False), axis=2))   # (n_epochs, n_chans)
print()
for r in run_all_classifiers(["lda", "knn", "gnb", "hgb"], "spectral", {}, X=X, y=y, cv_folds=3):
    if r.get("error"):
        print(f"  {r['classifier']:<16} FAILED — {r['error']}")
    else:
        print(f"  {r['classifier']:<16} {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}  [{r['category']}]")