# from moabb.datasets import BNCI2014_001, Kalunga2016

# d1 = BNCI2014_001()
# d2 = Kalunga2016()

# print(dir(d1))          # look for anything like "paradigm"
# print(getattr(d1, "paradigm", "NOT FOUND"))
# print(getattr(d2, "paradigm", "NOT FOUND"))
# print(d1.interval)
# print(d1.n_sessions)
# print(d1.code)
# print(d1.event_id)


# from moabb.datasets import BNCI2014_001

# d = BNCI2014_001()

# print("=== dataset.metadata ===")
# print(d.metadata)
# if d.metadata is not None:
#     print(vars(d.metadata))

# print("\n=== dataset.get_additional_metadata (subject 1) ===")
# try:
#     # signature is (subject, session, run) per the docs — adjust ids if this errors
#     result = d.get_additional_metadata(subject="1", session="0train", run="0")
#     print(result)
# except Exception as e:
#     print(f"Error: {e}")

# print("\n=== dataset.get_data (subject 1 only, to keep download small) ===")
# data = d.get_data(subjects=[1])
# print(type(data))
# print(list(data.keys()))              # subject ids
# subj_data = data[1]
# print(list(subj_data.keys()))          # session ids
# first_session = next(iter(subj_data.values()))
# print(list(first_session.keys()))      # run ids
# first_run = next(iter(first_session.values()))
# print(type(first_run))
# print(first_run.info)                  # this is the MNE Info object — very useful



# import inspect
# from mne.datasets import eegbci, sleep_physionet, ssvep, erp_core, limo, kiloword

# for label, fn in [
#     ("eegbci.load_data", eegbci.load_data),
#     ("sleep_physionet.age.fetch_data", sleep_physionet.age.fetch_data),
#     ("ssvep.data_path", ssvep.data_path),
#     ("erp_core.data_path", erp_core.data_path),
#     ("limo.load_data", limo.load_data),
#     ("kiloword.data_path", kiloword.data_path),
# ]:
#     params = inspect.signature(fn).parameters
#     print(f"{label:<34} update_path={'update_path' in params}  path={'path' in params}")



# import inspect
# from mne.datasets import sleep_physionet
# print(inspect.signature(sleep_physionet.age.fetch_data))

# import mne.datasets.utils as u
# print([n for n in dir(u) if "fetch" in n.lower() or "download" in n.lower()])



# import inspect
# import mne.datasets.utils as u
# print(inspect.getsource(u._downloader_params))


# import inspect
# from mne.datasets import erp_core
# print(inspect.signature(erp_core.data_path))



# import mne
# from mne.datasets import erp_core

# p = erp_core.data_path(verbose=False)
# fif = list(p.glob("*.fif"))[0]
# print(fif.name)

# try:
#     obj = mne.io.read_raw_fif(fif, preload=False, verbose=False)
#     print("RAW:", obj.info["sfreq"], "Hz,", len(obj.ch_names), "channels")
#     print("annotations:", set(obj.annotations.description) if obj.annotations else "none")
# except Exception as e:
#     print("not raw:", e)
#     obj = mne.read_epochs(fif, preload=False, verbose=False)
#     print("EPOCHS:", len(obj), "epochs,", obj.event_id)



# import mne
# from braindecode_fetcher import fetch_braindecode_dataset
# raw = fetch_braindecode_dataset("Schirrmeister2017", [1])
# print(raw.ch_names[:10])
# print(set(raw.annotations.description))


# import inspect, moabb.datasets.download as dl
# print(inspect.getsource(dl.get_dataset_path))


# import inspect
# import moabb.datasets.bnci as b
# src = inspect.getsource(b)
# for i, line in enumerate(src.split("\n")):
#     if "dl." in line or "data_dl" in line or "retrieve" in line:
#         print(i, line)

# import moabb.datasets.download as dl
# print(inspect.getsource(dl.data_dl))


# from moabb.datasets.download import _sanitize_path
# from pathlib import Path
# print(_sanitize_path(Path(r"C:\Users\nikhi\mne_data\MNE-bnci-data")))



# import numpy as np
# from backend.fetchers.moabb_fetcher import fetch_moabb_dataset
# ep = fetch_moabb_dataset("BNCI2014_001", [1])
# d = ep.get_data(copy=False)
# print("raw std:", np.std(d), "max:", np.abs(d).max())
# print("x1e6 std:", np.std(d * 1e6))


# from braindecode.classifier import EEGClassifier
# from braindecode.models import EEGNet
# m = EEGNet(n_chans=22, n_outputs=4, n_times=1001, sfreq=250)
# c = EEGClassifier(module=m, lr=0.5, max_epochs=1, train_split=None)
# print(c.lr, c.get_params().get("optimizer__lr", "not set"))


# from braindecode.models import EEGNet
# from braindecode.classifier import EEGClassifier
# c = EEGClassifier(module=EEGNet, module__n_chans=22, module__n_outputs=4,
#                   module__n_times=1001, module__sfreq=250,
#                   optimizer__lr=0.5, max_epochs=1, train_split=None)
# print(c.get_params()["optimizer__lr"])
# print(c.optimizer)


# import inspect
# from braindecode.models import EEGNet
# print(inspect.signature(EEGNet.__init__))


# from braindecode.models import EEGNet
# m = EEGNet(n_chans=22, n_outputs=4, n_times=1001, sfreq=250)
# print(sum(p.numel() for p in m.parameters()))
# m2 = EEGNet(n_chans=22, n_outputs=4, n_times=501, sfreq=250)
# print(sum(p.numel() for p in m2.parameters()))



from backend.fetchers.moabb_fetcher import fetch_moabb_dataset
ep = fetch_moabb_dataset("BNCI2014_001", [1])
print("montage:", ep.get_montage())
print("ch_names:", ep.ch_names[:8])