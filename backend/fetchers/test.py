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



import inspect
from mne.datasets import eegbci, sleep_physionet, ssvep, erp_core, limo, kiloword

for label, fn in [
    ("eegbci.load_data", eegbci.load_data),
    ("sleep_physionet.age.fetch_data", sleep_physionet.age.fetch_data),
    ("ssvep.data_path", ssvep.data_path),
    ("erp_core.data_path", erp_core.data_path),
    ("limo.load_data", limo.load_data),
    ("kiloword.data_path", kiloword.data_path),
]:
    params = inspect.signature(fn).parameters
    print(f"{label:<34} update_path={'update_path' in params}  path={'path' in params}")