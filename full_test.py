# import mne
# from backend.fetchers.mne_fetcher import fetch_mne_dataset
# from backend.core.presets import get_preset, resolve_annotation_rename
# from backend.core.preprocessor import preprocess_raw
# from backend.core.classifiers import run_all_classifiers

# mne.set_log_level("WARNING")

# RUNS = [4, 8, 12]
# raw = fetch_mne_dataset("eegbci", subjects=[1], runs=RUNS)

# # T1/T2 mean different things depending on runs, so the nested mapping in the
# # preset has to be resolved against the run selection before preprocessing.
# preset = get_preset("eegbci", mode="motor_imagery")
# rename = resolve_annotation_rename(preset, runs=RUNS)
# preset = {**preset, "annotation_rename": rename}

# # everything else — rename, event filtering, epoching, crop — happens inside
# epochs = preprocess_raw(raw, preset)

# y = epochs.events[:, 2]
# print(f"epochs: {epochs.get_data(copy=False).shape}")
# print(f"event_id: {epochs.event_id}, classes: {set(y)}\n")

# for r in run_all_classifiers(
#     ["lda", "svm", "rf", "riemann_ts", "riemann_mdm"],
#     "motor_imagery", {}, y=y, epochs_obj=epochs, cv_folds=5,
# ):
#     if r.get("error"):
#         print(f"  {r['classifier']:<14} FAILED — {r['error']}")
#     else:
#         print(f"  {r['classifier']:<14} {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}")


from backend.fetchers.mne_fetcher import fetch_mne_dataset
from backend.core.detector import detect_capabilities

raw = fetch_mne_dataset("eegbci", subjects=[1], runs=[4, 8, 12])
caps = detect_capabilities(raw)
print({k: type(v).__name__ for k, v in caps.items()})