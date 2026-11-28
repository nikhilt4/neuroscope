"""
Playback: replay a continuous public recording in real time, as if it were streaming
off a headset. v1 is display-only (traces + event markers); live decoding will later
consume the same stream.

Stateless like the rest of the app (NEUROSCOPE_CONTEXT.md 2.2): each stream request
rebuilds the recording from the session record. Pause / seek / speed changes are just
a new request with a different `start`.
"""
import json
import time
from collections import OrderedDict
from typing import Iterator, List, Optional

import mne
import numpy as np
from fastapi import HTTPException

from backend.core.pipeline import reload_data
from backend.core.preprocessor import _apply_bandpass, _apply_notch, _rename_annotations
from backend.core.presets import get_preset, resolve_annotation_rename

# public datasets that arrive as continuous raw, plus the user's own continuous recordings
PLAYBACK_SOURCES = {"mne", "braindecode", "upload", "bids"}
FRAMES_PER_SECOND = 25                       # smooth scrolling without flooding the connection
DISPLAY_FILTER = (0.5, 40.0)                 # used when the dataset preset has no filter edges
MAX_DISPLAY_CHANNELS = 8
MOTOR_FIRST = ["C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4", "C1", "C2", "C5", "C6"]


# Prepared recordings for the most recently played sessions. Without this every
# pause / seek / speed change reloaded and re-filtered the whole recording (BCI IV 2a via
# braindecode is 77 min: several seconds before the first frame). It's only a cache —
# a restart or another worker simply rebuilds — so the stateless design still holds.
_CACHE: "OrderedDict[str, mne.io.BaseRaw]" = OrderedDict()
_CACHE_SIZE = 2


def load_for_playback_cached(session: dict) -> mne.io.BaseRaw:
    key = json.dumps([session.get("session_id"), session.get("source"), session.get("dataset_name"),
                      session.get("subjects"), session.get("runs")], default=str)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    raw = load_for_playback(session)
    _CACHE[key] = raw
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return raw


def load_for_playback(session: dict) -> mne.io.BaseRaw:
    """
    The session's recording as continuous EEG, display-filtered, with event names the
    rest of the app uses (eegbci T1/T2 -> left_hand/right_hand, Sleep-EDF stages already
    AASM, ERP CORE correct/error).

    Display filter: always 0.5–40 Hz (clipped below Nyquist), plus the preset's mains notch
    if it has one — enough to remove drift and line noise so traces look like EEG. NOT the
    preset's analysis band: eegbci's 7–30 Hz would hide everything but mu/beta. Zero-phase
    (offline) filtering is fine because nothing is decoded; a live-decoding mode will need
    causal filtering (MNE's phase="minimum").
    """
    if session.get("source") == "moabb":
        raise HTTPException(
            status_code=400,
            detail="MOABB delivers this dataset already cut into trials, so there's no continuous "
                   "recording to play back. Load it from braindecode instead (for BCI Competition "
                   "IV 2a, the braindecode version is continuous).")
    if session.get("source") not in PLAYBACK_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Playback supports uploaded recordings and public datasets from MNE and braindecode.")
    data = reload_data(session)
    if isinstance(data, mne.BaseEpochs):
        raise HTTPException(
            status_code=400,
            detail="This dataset arrives already cut into trials, so there's no continuous "
                   "recording to play back. For BCI Competition IV 2a, use the braindecode "
                   "version, which is continuous.")

    # EEG plus the eye and heart channels clinicians read alongside it (her reference
    # screenshot shows ECG under the EEG). EMG/respiration stay out: Sleep-EDF stores
    # some at 1 Hz, which would draw as flat lines after upsampling.
    picks = mne.pick_types(data.info, eeg=True, eog=True, ecg=True)
    if not len(picks):
        raise HTTPException(status_code=400, detail="This recording has no EEG channels to display.")
    raw = data.copy().pick(picks)
    preset = get_preset(session.get("dataset_name"), "motor_imagery")
    if preset.get("annotation_rename"):
        try:
            raw = _rename_annotations(raw, resolve_annotation_rename(preset, session.get("runs")))
        except HTTPException:
            pass   # unknown run variant: keep the raw names rather than refuse to play
    lo, hi = DISPLAY_FILTER
    nyquist = raw.info["sfreq"] / 2
    raw = _apply_bandpass(raw, lo, hi if hi < nyquist else None)
    if preset.get("notch_enabled") and preset.get("notch_freq", 50.0) < nyquist:
        raw = _apply_notch(raw, preset.get("notch_freq", 50.0))
    return raw


def default_channels(ch_names: List[str]) -> List[str]:
    """Up to 8 channels, motor-cortex sites first (what motor-imagery demos want to see)."""
    lower = {c.lower(): c for c in ch_names}
    picked = [lower[m.lower()] for m in MOTOR_FIRST if m.lower() in lower]
    for c in ch_names:
        if len(picked) >= MAX_DISPLAY_CHANNELS:
            break
        if c not in picked:
            picked.append(c)
    return picked[:MAX_DISPLAY_CHANNELS]


def event_list(raw: mne.io.BaseRaw) -> List[dict]:
    """
    Every event as {t (s from the start of the data), label}. events_from_annotations
    skips BAD/EDGE boundary markers by default, so concatenation joins don't appear.
    https://mne.tools/1.12/generated/mne.events_from_annotations.html
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        return []
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    names = {v: k for k, v in event_id.items()}
    sfreq = raw.info["sfreq"]
    return [{"t": round(float((s - raw.first_samp) / sfreq), 3), "label": names[int(code)]}
            for s, _, code in events]


def playback_info(raw: mne.io.BaseRaw) -> dict:
    events = event_list(raw)
    return {
        "channels": raw.ch_names,
        "default_channels": default_channels(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
        "duration_s": float(raw.n_times / raw.info["sfreq"]),
        "events": events,
        "event_labels": sorted({e["label"] for e in events}),
        "display_filter_hz": [raw.info.get("highpass"), raw.info.get("lowpass")],
    }


def stream_frames(raw: mne.io.BaseRaw, channels: Optional[List[str]] = None, start: float = 0.0,
                  speed: float = 1.0, pace: bool = True, prefill: float = 0.0,
                  preview: bool = False) -> Iterator[str]:
    """
    Server-Sent Events for a real-time replay. Each frame carries 1/25 s of samples
    (µV) for the requested channels; events are sent as the playhead passes them.
    `pace=False` skips the sleeps (tests).

    `prefill` (s): first send the `prefill` seconds BEFORE `start` in one unpaced frame,
    so after a seek the screen is full immediately — like scrubbing a video — instead of
    refilling from the edge. `preview=True` stops after the prefill (paused scrubbing).

    SSE wire format: "data: <json>\\n\\n".
    https://html.spec.whatwg.org/multipage/server-sent-events.html
    """
    chans = [c for c in (channels or default_channels(raw.ch_names)) if c in raw.ch_names]
    if not chans:
        raise HTTPException(status_code=400, detail="None of the requested channels are in this recording.")
    speed = float(min(max(speed, 0.25), 8.0))
    sfreq = float(raw.info["sfreq"])
    picks = [raw.ch_names.index(c) for c in chans]
    data = raw.get_data(picks=picks) * 1e6          # µV, channels x samples
    n_total = data.shape[1]
    events = event_list(raw)

    per_frame = max(1, int(round(sfreq / FRAMES_PER_SECOND)))
    pos = int(max(0.0, min(start, n_total / sfreq)) * sfreq)
    ev_idx = next((i for i, e in enumerate(events) if e["t"] >= pos / sfreq), len(events))

    def sse(obj) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    yield sse({"type": "start", "t": pos / sfreq, "channels": chans, "sfreq": sfreq, "speed": speed})

    if prefill > 0:
        pf = max(0, pos - int(prefill * sfreq))
        for e in events:
            if pf / sfreq <= e["t"] < pos / sfreq:
                yield sse({"type": "event", **e, "prefill": True})
        if pos > pf:
            yield sse({"type": "data", "t": pf / sfreq, "prefill": True,
                       "samples": np.round(data[:, pf:pos], 2).tolist()})
    if preview:
        yield sse({"type": "end", "t": pos / sfreq, "preview": True})
        return

    t_wall0, pos0 = time.perf_counter(), pos
    while pos < n_total:
        end = min(n_total, pos + per_frame)
        t_end = end / sfreq
        while ev_idx < len(events) and events[ev_idx]["t"] < t_end:
            yield sse({"type": "event", **events[ev_idx]})
            ev_idx += 1
        # 2-decimal µV keeps frames small (~2 KB for 8 channels at 250 Hz) and is far
        # finer than anything visible on screen
        yield sse({"type": "data", "t": pos / sfreq,
                   "samples": np.round(data[:, pos:end], 2).tolist()})
        pos = end
        if pace:
            # schedule against the wall clock rather than sleeping a fixed amount, so
            # serialisation time doesn't make playback drift slower than real time
            due = t_wall0 + (pos - pos0) / sfreq / speed
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    yield sse({"type": "end", "t": n_total / sfreq})


# ── Page viewer ─────────────────────────────────────────────────────────────────
# A clinical-style page: every channel stacked, one window of time at a time, fetched on
# demand instead of streamed frame by frame (a neuroscientist found the video-style replay
# slow and asked for a scrollable sheet like a clinical EEG reader).

MAX_PAGE_S = 60.0
MAX_PAGE_CHANNELS = 64
# Points per channel per page the browser needs: a page is ~1,300 px wide, so ~2 points
# per pixel is plenty. Decimation never goes below MIN_VIEW_SFREQ, which stays above twice
# the 40 Hz display low-pass, so striding can't alias what's on screen.
TARGET_POINTS = 2600
MIN_VIEW_SFREQ = 100.0

# Longitudinal bipolar ("double banana"), ordered as in the AP montage clinicians showed us:
# right parasagittal, left parasagittal, right temporal, left temporal, midline.
# https://www.acns.org/UserFiles/file/Guideline6-RecommendedMontagesforUseinClinicalEEG.pdf
# (ACNS Guideline 6; UNVERIFIED exact ordering against the PDF — matches common practice).
DOUBLE_BANANA = [
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]
# 10-20 names changed in the 10-10 system (T3 -> T7 etc.); accept either in a recording
OLD_NAMES = {"t7": "t3", "t8": "t4", "p7": "t5", "p8": "t6"}
MIN_BIPOLAR_PAIRS = 4


def _hemisphere(name: str) -> str:
    """left / right / midline from a 10-20 name: odd = left, even = right, z = midline."""
    first = name.split("-")[0].strip()
    if first[-1:].lower() == "z":
        return "midline"
    digits = "".join(ch for ch in first if ch.isdigit())
    if not digits:
        return "midline"
    return "left" if int(digits) % 2 else "right"


def _find(name: str, lower_map: dict) -> Optional[str]:
    key = name.lower()
    return lower_map.get(key) or lower_map.get(OLD_NAMES.get(key, ""))


def _bipolar_pairs(raw: mne.io.BaseRaw) -> List[tuple]:
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    lower = {c.lower(): c for c in eeg}
    pairs = []
    for a, b in DOUBLE_BANANA:
        ca, cb = _find(a, lower), _find(b, lower)
        if ca and cb:
            pairs.append((ca, cb))
    return pairs


def available_montages(raw: mne.io.BaseRaw) -> List[str]:
    out = ["referential"]
    if len(mne.pick_types(raw.info, eeg=True)) >= 3:
        out.append("average")   # average of 2 channels is meaningless
    if len(_bipolar_pairs(raw)) >= MIN_BIPOLAR_PAIRS:
        out.append("bipolar")
    return out


def channel_meta(raw: mne.io.BaseRaw) -> List[dict]:
    types = raw.get_channel_types()
    return [{"name": n, "type": t, "group": _hemisphere(n) if t == "eeg" else t}
            for n, t in zip(raw.ch_names, types)]


def viewer_info(raw: mne.io.BaseRaw) -> dict:
    """playback_info plus what the page viewer needs: channel types/groups and montages."""
    return {**playback_info(raw), "channel_meta": channel_meta(raw),
            "montages": available_montages(raw), "max_page_s": MAX_PAGE_S}


def page_window(raw: mne.io.BaseRaw, start: float, duration: float, montage: str = "referential",
                channels: Optional[List[str]] = None) -> dict:
    """
    One page of the recording, in µV, with the montage applied and the events inside it.

    Montages:
      referential — channels as recorded;
      average     — each EEG channel minus the mean of all EEG channels (non-EEG unchanged);
      bipolar     — longitudinal "double banana" differences (Fp1-F3, …) for every pair whose
                    two electrodes exist, followed by any EOG/ECG channels.
    `channels` (names in the montage's output) limits what is returned; None = all, capped.
    """
    if montage not in available_montages(raw):
        raise HTTPException(status_code=400,
                            detail=f"The {montage} montage isn't available for this recording.")
    sfreq = float(raw.info["sfreq"])
    total = raw.n_times / sfreq
    duration = float(min(max(duration, 1.0), MAX_PAGE_S))
    start = float(min(max(start, 0.0), max(0.0, total - duration)))
    i0, i1 = int(start * sfreq), min(raw.n_times, int((start + duration) * sfreq))

    data = raw.get_data(start=i0, stop=i1) * 1e6
    types = raw.get_channel_types()
    eeg_idx = [i for i, t in enumerate(types) if t == "eeg"]
    other_idx = [i for i, t in enumerate(types) if t != "eeg"]

    if montage == "bipolar":
        index = {n: i for i, n in enumerate(raw.ch_names)}
        pairs = _bipolar_pairs(raw)
        rows = [data[index[a]] - data[index[b]] for a, b in pairs]
        names = [f"{a}-{b}" for a, b in pairs]
        meta = [{"name": n, "type": "eeg", "group": _hemisphere(n)} for n in names]
        rows += [data[i] for i in other_idx]
        meta += [channel_meta(raw)[i] for i in other_idx]
        out = np.array(rows) if rows else np.empty((0, data.shape[1]))
    else:
        out = data.copy()
        if montage == "average" and eeg_idx:
            out[eeg_idx] -= out[eeg_idx].mean(axis=0, keepdims=True)
        meta = channel_meta(raw)

    if channels:
        keep = [i for i, m in enumerate(meta) if m["name"] in set(channels)]
        if not keep:
            raise HTTPException(status_code=400, detail="None of the requested channels are in this montage.")
    else:
        keep = list(range(min(len(meta), MAX_PAGE_CHANNELS)))
    out, meta = out[keep], [meta[i] for i in keep]

    # stride decimation, never below MIN_VIEW_SFREQ (see TARGET_POINTS)
    step = max(1, min(int(np.ceil(out.shape[1] / TARGET_POINTS)), int(sfreq // MIN_VIEW_SFREQ) or 1))
    out = out[:, ::step]
    events = [e for e in event_list(raw) if start <= e["t"] < start + duration]
    return {
        "start": start, "duration": duration, "sfreq": sfreq / step, "montage": montage,
        "channels": meta, "events": events,
        # 1 decimal µV: far below what a pixel shows, and keeps a 22-channel page ~300 KB
        "data": np.round(out, 1).tolist(),
    }
