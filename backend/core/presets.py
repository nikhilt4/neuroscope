from typing import Optional

# Presets keyed by dataset_name (specific) or mode (generic fallback)
# Specific dataset presets take priority over mode-based generic ones

DATASET_PRESETS = {
    "eegbci": {
        "bandpass_low": 7.0,
        "bandpass_high": 30.0,
        "notch_enabled": False,
        "epoch_tmin": -1.0,
        "epoch_tmax": 4.0,
        "epoch_train_crop": (1.0, 2.0),
        "baseline_enabled": False,
        "reference": "average",
        "ica_enabled": False,
        "annotation_rename": {
            "hands_feet_runs": {"T1": "hands", "T2": "feet"},
            "left_right_runs": {"T1": "left_hand", "T2": "right_hand"},
        },
        "montage": "standard_1005",
    },

    "sleep_physionet": {
        "bandpass_low": 0.3,
        "bandpass_high": 35.0,
        "notch_enabled": False,
        "epoch_tmin": 0.0,
        "epoch_tmax": 30.0,
        "baseline_enabled": False,
        "reference": "none",
        "ica_enabled": False,
        "montage": None,
    },

    "ssvep": {
        "bandpass_low": 0.1,
        "bandpass_high": None,
        "notch_enabled": True,
        "notch_freq": 50.0,
        "epoch_tmin": -1.0,
        "epoch_tmax": 20.0,
        "baseline_enabled": False,
        "reference": "average",
        "ica_enabled": False,
        "montage": "easycap-M1",
    },

    "erp_core": {
        "N400": {
            "bandpass_low": 0.1, "bandpass_high": None, "notch_freq": 60,
            "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline_enabled": True,
            "reference": ["P9", "P10"], "ica_enabled": False,
        },
        "ERN": {
            "bandpass_low": 0.1, "bandpass_high": None, "notch_freq": 60,
            "epoch_tmin": -0.6, "epoch_tmax": 0.4, "baseline": (-0.4, -0.2),
            "reference": ["P9", "P10"], "ica_enabled": True,
        },
        "LRP": {
            "bandpass_low": 0.1, "bandpass_high": None, "notch_freq": 60,
            "epoch_tmin": -0.8, "epoch_tmax": 0.2, "baseline": (None, -0.6),
            "reference": ["P9", "P10"], "ica_enabled": True,
        },
        "MMN": {
            "bandpass_low": 0.1, "bandpass_high": None, "notch_freq": 60,
            "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline_enabled": True,
            "reference": ["P9", "P10"], "ica_enabled": True,
        },
        "N2pc": {
            "bandpass_low": 0.1, "bandpass_high": None, "notch_freq": 60,
            "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline_enabled": True,
            "reference": ["P9", "P10"], "ica_enabled": True,
        },
        "N170": {
            "bandpass_low": 1.0, "bandpass_high": 100.0, "notch_freq": 60,
            "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline_enabled": True,
            "reference": "average", "ica_enabled": True,
        },
        "P3": {
            "bandpass_low": 0.1, "bandpass_high": None, "notch_freq": 60,
            "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline_enabled": True,
            "reference": ["P9", "P10"], "ica_enabled": False,
        },
    },

    "limo": {
        "bandpass_low": None,
        "bandpass_high": None,
        "notch_enabled": False,
        "baseline_enabled": False,
        "reference": "none",
        "ica_enabled": False,
        "montage": None,
    },

    "kiloword": {
        "bandpass_low": 0.1,
        "bandpass_high": 40.0,
        "notch_enabled": False,
        "baseline_enabled": False,
        "reference": "average",
        "ica_enabled": False,
        "montage": None,
        "_confidence": "unverified",
    }
}

# Generic fallback presets by MODE, used for user uploads with no known dataset
MODE_PRESETS = {
    "motor_imagery": {
        "bandpass_low": 7.0,
        "bandpass_high": 30.0,
        "notch_enabled": True,
        "notch_freq": 50.0,
        "epoch_tmin": -0.2,
        "epoch_tmax": 0.8,
        "baseline_enabled": True,
        "reference": "average",
        "ica_enabled": False,
    },
    "spectral": {
        "bandpass_low": 1.0,
        "bandpass_high": 40.0,
        "notch_enabled": True,
        "notch_freq": 50.0,
        "epoch_tmin": -0.2,
        "epoch_tmax": 0.8,
        "baseline_enabled": True,
        "reference": "average",
        "ica_enabled": False,
    },
    "basic": {
        "bandpass_low": 1.0,
        "bandpass_high": 40.0,
        "notch_enabled": True,
        "notch_freq": 50.0,
        "reference": "average",
        "ica_enabled": False,
    },
}


def get_preset(dataset_name: Optional[str], mode: str, task: Optional[str] = None) -> dict:
    """
    Returns the best available preset:
    1. Dataset-specific + task-specific (e.g. erp_core + N400)
    2. Dataset-specific (e.g. eegbci)
    3. Mode-based generic fallback (e.g. motor_imagery)
    """
    if dataset_name and dataset_name in DATASET_PRESETS:
        preset = DATASET_PRESETS[dataset_name]
        if task and isinstance(preset, dict) and task in preset:
            return preset[task]
        if not isinstance(preset, dict) or "bandpass_low" in preset:
            return preset

    return MODE_PRESETS.get(mode, MODE_PRESETS["basic"])


