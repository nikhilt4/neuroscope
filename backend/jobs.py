"""
Background jobs: dataset fetches and analyses in their own killable processes.

Why processes, not threads: a stalled download sits in a C-level socket read that
ignores Ctrl+C and can't be interrupted from Python (NEUROSCOPE_CONTEXT.md section 6,
"A stalled download wedges a worker permanently"). A thread stuck there can never be
stopped; a process can always be terminated. The cost is a few seconds of start-up
per job while the child imports MNE.

State lives on disk (TEMP_BASE/jobs/<id>/state.json), not in memory, matching the
stateless session design (section 2.2): any server worker can report on or cancel
any job, and a job's status survives a uvicorn restart.

Lifecycle: queued -> running -> done | failed | cancelled | timeout
"""
import json
import multiprocessing as mp
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException

TEMP_BASE = Path("/tmp/neuroscope")
JOBS_DIR = TEMP_BASE / "jobs"
FINISHED = {"done", "failed", "cancelled", "timeout"}

# Defaults — overridable per request. Fetches can be genuinely large (Schirrmeister2017
# is ~490 MB per subject); analyses with EEGNet can take minutes.
DEFAULT_TIMEOUT_S = {"fetch": 600, "analyze": 1800, "_test_sleep": 30}

# spawn is the only start method on Windows and the safe one elsewhere: the child gets
# a fresh interpreter instead of a fork of a server holding sockets and threads.
# https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods
_CTX = mp.get_context("spawn")


def _state_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "state.json"


def _read_json_retrying(path: Path) -> dict:
    """
    Read state.json while other processes may be replacing it. On Windows a reader can
    briefly see a PermissionError or an empty file mid-replace (seen intermittently in
    test_jobs), so retry for up to ~1 s before giving up.
    """
    last_error = None
    for _ in range(30):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, PermissionError) as e:
            last_error = e
            time.sleep(0.03)
    raise last_error


def read_state(job_id: str) -> dict:
    path = _state_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found. It may have expired.")
    return _read_json_retrying(path)


def _write_state(job_id: str, **changes) -> dict:
    """
    Merge changes into state.json via write-temp-then-replace. Each writer (the job
    process, the server's watchdog thread, a cancel request) uses its OWN temp file —
    a shared "state.tmp" let two simultaneous writers corrupt each other.
    """
    import os
    path = _state_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _read_json_retrying(path) if path.exists() else {}
    state.update(changes, updated=time.time())
    tmp = path.with_name(f"state.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(state, default=str))
    # Windows refuses to replace a file another process has open for reading
    for attempt in range(30):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.03)
    return state


# ── what each job kind does (runs in the CHILD process) ───────────────────────

def _do_fetch(payload: dict, progress: Callable[[str], None]) -> dict:
    from backend.models import DatasetFetchRequest
    from backend.routes.upload import run_fetch
    return run_fetch(DatasetFetchRequest(**payload), progress=progress).model_dump()


def _do_analyze(payload: dict, progress: Callable[[str], None]) -> dict:
    from backend.models import AnalysisRequest
    from backend.routes.analyze import run_analysis
    return run_analysis(AnalysisRequest(**payload), progress=progress)


def _do_test_sleep(payload: dict, progress: Callable[[str], None]) -> dict:
    """Test-only job: sleeps, so timeout and cancel can be tested without real data."""
    for i in range(int(payload.get("seconds", 60))):
        progress(f"Sleeping {i}s")
        time.sleep(1)
    return {"slept": payload.get("seconds")}


HANDLERS = {"fetch": _do_fetch, "analyze": _do_analyze, "_test_sleep": _do_test_sleep}


def _child_main(job_id: str, kind: str, payload: dict) -> None:
    """Entry point of the job process. Writes progress, then the result or the error."""
    import os
    import mne
    mne.set_log_level("WARNING")
    # record our own pid too: the parent writes it after start(), and the two
    # read-modify-write updates can race, which could drop the parent's copy
    _write_state(job_id, status="running", stage="Starting", pid=os.getpid())
    try:
        result = HANDLERS[kind](payload, lambda msg: _write_state(job_id, stage=msg))
        (JOBS_DIR / job_id / "result.json").write_text(json.dumps(result, default=str))
        _write_state(job_id, status="done", stage="Finished", finished=time.time())
    except HTTPException as e:
        _write_state(job_id, status="failed", error=str(e.detail), finished=time.time())
    except Exception as e:  # noqa: BLE001 — any crash must end up in the job state, not vanish
        _write_state(job_id, status="failed", finished=time.time(),
                     error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])


# ── server side ───────────────────────────────────────────────────────────────

def _kill(pid: Optional[int]) -> None:
    """Terminate a job process and anything it started (e.g. torch workers)."""
    if not pid:
        return
    import psutil
    # https://psutil.readthedocs.io/en/latest/#psutil.Process.children (psutil 7.x)
    try:
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def _enforce_timeout(job_id: str, state: dict) -> dict:
    """If a running job has outlived its timeout, kill it and record that."""
    if state.get("status") in FINISHED:
        return state
    if time.time() - state.get("started", time.time()) > state.get("timeout_s", 1e9):
        _kill(state.get("pid"))
        return _write_state(job_id, status="timeout", finished=time.time(),
                            error=f"Stopped after {int(state['timeout_s'])} s without finishing. "
                                  "If this was a download, the server may be slow or unreachable — try again later.")
    return state


def _watchdog(job_id: str, process) -> None:
    """Background thread in the server: enforces the timeout even if nobody polls."""
    while process.is_alive():
        state = read_state(job_id)
        state = _enforce_timeout(job_id, state)
        if state.get("status") in FINISHED:
            return
        time.sleep(1)
    # The process exited. If it died without writing a final state (crash, kill -9),
    # say so instead of leaving the job "running" forever.
    state = read_state(job_id)
    if state.get("status") not in FINISHED:
        _write_state(job_id, status="failed", finished=time.time(),
                     error=f"The job process exited unexpectedly (exit code {process.exitcode}).")


def start_job(kind: str, payload: dict, timeout_s: Optional[float] = None) -> str:
    if kind not in HANDLERS:
        raise HTTPException(status_code=400, detail=f"Unknown job kind: {kind}")
    job_id = uuid.uuid4().hex
    timeout = float(timeout_s or DEFAULT_TIMEOUT_S[kind])
    _write_state(job_id, id=job_id, kind=kind, status="queued", stage="Queued",
                 started=time.time(), timeout_s=timeout)
    process = _CTX.Process(target=_child_main, args=(job_id, kind, payload), daemon=True)
    process.start()
    _write_state(job_id, pid=process.pid)
    threading.Thread(target=_watchdog, args=(job_id, process), daemon=True).start()
    return job_id


def job_status(job_id: str) -> dict:
    """State plus elapsed time; includes the result once done."""
    state = _enforce_timeout(job_id, read_state(job_id))
    out = {k: v for k, v in state.items() if k not in ("pid", "traceback")}
    end = state.get("finished") or time.time()
    out["elapsed_s"] = round(end - state.get("started", end), 1)
    if state.get("status") == "done":
        out["result"] = json.loads((JOBS_DIR / job_id / "result.json").read_text())
    return out


def cancel_job(job_id: str) -> dict:
    state = read_state(job_id)
    if state.get("status") in FINISHED:
        return job_status(job_id)
    _kill(state.get("pid"))
    _write_state(job_id, status="cancelled", stage="Cancelled", finished=time.time())
    return job_status(job_id)
