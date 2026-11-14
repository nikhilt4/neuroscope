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

PLAYBACK_SOURCES = {"mne", "braindecode"}   # v1: public datasets that arrive as continuous raw
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
            detail="Playback currently supports public datasets from MNE and braindecode. "
                   "Uploaded files are coming later.")
    data = reload_data(session)
    if isinstance(data, mne.BaseEpochs):
        raise HTTPException(
            status_code=400,
            detail="This dataset arrives already cut into trials, so there's no continuous "
                   "recording to play back. For BCI Competition IV 2a, use the braindecode "
                   "version, which is continuous.")

    raw = data.copy().pick("eeg")
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
