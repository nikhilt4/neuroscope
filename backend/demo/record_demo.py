"""
Records real backend responses for the static online demo (frontend `npm run build:demo`).

The demo has no server: frontend/src/demo/demoApi.js answers every API call from the
JSON files this script writes into frontend/public/demo/. Everything in them comes from
the real backend over HTTP, through the same routes the UI calls, so the demo shows
genuine numbers, scalp maps and spectra, not mock-ups.

Start the backend first (uvicorn backend.main:app), then from the repo root
(takes ~20-40 min, most of it EEGNet and the 9-subject runs):
    python -m backend.demo.record_demo
    python -m backend.demo.record_demo --skip-multi      # reuse the multi-subject files

What gets recorded:
  * BCI Competition IV 2a (MOABB BNCI2014_001): within- and cross-subject runs over all
    9 subjects (motor imagery mode, every decoder except EEGNet), then the subject with
    the best tangent-space score is recorded in every analysis mode with all 14 decoders.
  * Sleep-EDF (MNE sleep_physionet), subject 1: every analysis mode, all 14 decoders,
    plus a 30-minute playback excerpt (int16 binary) around the most sleep-stage changes.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import requests

from backend.core.playback import event_list, load_for_playback
from backend.utils.session_registry import get_session

OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo"

ALL_CLASSIFIERS = ["lda", "svm", "rf", "logreg", "extra_trees", "hgb", "mlp", "knn", "gnb",
                   "riemann_ts", "riemann_mdm", "riemann_fgmdm", "xdawn_ts", "eegnet"]
# EEGNet is left out of the 9-subject runs: 9 subjects x 5 folds of GPU training would
# triple the recording time, and the demo's Step 4 says so when EEGNet is picked.
MULTI_CLASSIFIERS = [c for c in ALL_CLASSIFIERS if c != "eegnet"]

BNCI = {"source": "moabb", "id": "BNCI2014_001"}
SLEEP = {"source": "mne", "id": "sleep_physionet", "subject": 1}

PLAYBACK_CLIP_S = 30 * 60
# int16 µV x 10: 0.1 µV resolution, ±3276 µV range — far finer than the display needs,
# and a 30-min, 2-channel, 100 Hz clip stays under 1 MB.
PLAYBACK_SCALE = 10.0

API = "http://127.0.0.1:8000"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def call(method: str, path: str, **kw) -> dict:
    # no timeout: a 9-subject cross-subject run legitimately takes several minutes
    r = requests.request(method, API + path, timeout=None, **kw)
    if r.status_code != 200:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
    return r.json()


def save(name: str, obj) -> None:
    path = OUT / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    log(f"  wrote {name} ({path.stat().st_size / 1e3:.0f} KB)")


def analysis_payload(session_id: str, mode: str, selected: list, **extra) -> dict:
    # Mirrors what Wizard.jsx sends with untouched settings: no preprocessing overrides,
    # the default classifier config (same defaults as ClassifySection.jsx).
    return {"session_id": session_id, "mode": mode, "preprocessing": {},
            "classifiers": {"selected": selected, "n_folds": 5, "csp_components": 4,
                            "csp_reg": None, "configs": {}},
            **extra}


def fetch(ds: dict, subject: int) -> dict:
    log(f"fetch {ds['id']} subject {subject}")
    return call("POST", "/dataset/fetch",
                json={"source": ds["source"], "dataset_name": ds["id"], "subjects": [subject]})


def record_single(ds: dict, subject: int, fetched: dict, manifest: dict) -> None:
    key = ds["id"]
    save(f"fetch__{key}.json", fetched)
    for mode in fetched["available_modes"]:
        preset = call("GET", "/preprocessing/preset", params={"dataset_name": key, "mode": mode})
        save(f"preset__{key}__{mode}.json", preset)
        log(f"analyze {key} s{subject} {mode}")
        res = call("POST", "/analyze", json=analysis_payload(fetched["session_id"], mode, ALL_CLASSIFIERS))
        save(f"result__{key}__{mode}__single.json", res)
        manifest["results"].setdefault(key, {}).setdefault(mode, []).append("single")


def record_multi(fetched: dict, manifest: dict) -> dict:
    """Both multi-subject evaluations on all 9 BNCI subjects. Returns the within-subject result."""
    within = None
    for evaluation in ("within_subject", "cross_subject"):
        log(f"analyze BNCI {evaluation} subjects 1-9")
        res = call("POST", "/analyze", json=analysis_payload(
            fetched["session_id"], "motor_imagery", MULTI_CLASSIFIERS,
            evaluation=evaluation, subjects=list(range(1, 10))))
        save(f"result__{BNCI['id']}__motor_imagery__{evaluation}.json", res)
        manifest["results"].setdefault(BNCI["id"], {}).setdefault("motor_imagery", []).append(evaluation)
        within = res if evaluation == "within_subject" else within
    return within


def best_subject(within: dict) -> int:
    ts = next(c for c in within["classifiers"] if c["classifier"] == "riemann_ts")
    scores = {p["subject"]: round(p["accuracy"], 3) for p in ts["per_subject"]}
    log(f"  tangent-space per subject: {scores}")
    return int(max(scores, key=scores.get))


def record_playback(fetched: dict) -> dict:
    """30-minute Sleep-EDF excerpt around the densest run of sleep-stage changes."""
    session = get_session(fetched["session_id"])
    raw = load_for_playback(session)
    sfreq = float(raw.info["sfreq"])
    events = event_list(raw)
    duration = raw.n_times / sfreq
    # slide a 30-min window in 5-min steps, keep the one with the most stage markers
    starts = np.arange(0, max(1.0, duration - PLAYBACK_CLIP_S), 300.0)
    counts = [sum(s <= e["t"] < s + PLAYBACK_CLIP_S for e in events) for s in starts]
    t0 = float(starts[int(np.argmax(counts))])
    i0, i1 = int(t0 * sfreq), int(min(duration, t0 + PLAYBACK_CLIP_S) * sfreq)
    data = raw.get_data(start=i0, stop=i1) * 1e6
    np.clip(np.round(data * PLAYBACK_SCALE), -32768, 32767).astype("<i2").tofile(OUT / "playback__sleep.bin")
    clip_events = [{"t": round(e["t"] - t0, 3), "label": e["label"]}
                   for e in events if t0 <= e["t"] < t0 + PLAYBACK_CLIP_S]
    info = {
        "channels": raw.ch_names,
        "default_channels": raw.ch_names[:8],
        "sfreq": sfreq,
        "duration_s": (i1 - i0) / sfreq,
        "events": clip_events,
        "event_labels": sorted({e["label"] for e in clip_events}),
        "display_filter_hz": [raw.info.get("highpass"), raw.info.get("lowpass")],
        # demo-only fields
        "n_samples": i1 - i0,
        "scale": PLAYBACK_SCALE,
        "clip_start_s": t0,
        "recording_duration_s": duration,
    }
    save("playback__sleep.json", info)
    log(f"  playback clip from {t0 / 3600:.2f} h, {len(clip_events)} stage markers")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-multi", action="store_true", help="reuse existing multi-subject files")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = {"recorded": date.today().isoformat(), "results": {}, "datasets": {}}

    # ── BCI IV 2a ────────────────────────────────────────────────────────────
    if args.skip_multi:
        within = json.loads((OUT / f"result__{BNCI['id']}__motor_imagery__within_subject.json").read_text())
        manifest["results"].setdefault(BNCI["id"], {})["motor_imagery"] = ["within_subject", "cross_subject"]
    else:
        within = record_multi(fetch(BNCI, 1), manifest)
    subject = best_subject(within)
    log(f"best BNCI subject: {subject}")
    record_single(BNCI, subject, fetch(BNCI, subject), manifest)
    manifest["datasets"][BNCI["id"]] = {"source": BNCI["source"], "subject": subject}

    # ── Sleep-EDF ────────────────────────────────────────────────────────────
    sleep_fetch = fetch(SLEEP, SLEEP["subject"])
    record_single(SLEEP, SLEEP["subject"], sleep_fetch, manifest)
    manifest["datasets"][SLEEP["id"]] = {"source": SLEEP["source"], "subject": SLEEP["subject"],
                                         "playback": "sleep"}
    record_playback(sleep_fetch)

    # ── dataset catalogues, trimmed to the demo datasets ─────────────────────
    for source in ("mne", "moabb"):
        cat = call("GET", f"/dataset/list/{source}")
        keep = [d for d in cat["datasets"] if d["id"] in manifest["datasets"]]
        for d in keep:
            d["demo_subject"] = manifest["datasets"][d["id"]]["subject"]
        save(f"datasets__{source}.json", {"datasets": keep})

    save("manifest.json", manifest)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
