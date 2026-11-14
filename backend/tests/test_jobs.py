"""
Background jobs: completion, timeout, cancel, failure reporting.

Uses the test-only "_test_sleep" job so timeout/cancel need no data or network.
Each job is a real spawned process, so every test pays ~2-5 s of start-up.
"""
import shutil
import time

import psutil
import pytest

from backend import jobs


def _wait(job_id, statuses, limit=90):
    """Poll until the job reaches one of `statuses`; return its status dict."""
    deadline = time.time() + limit
    while time.time() < deadline:
        s = jobs.job_status(job_id)
        if s["status"] in statuses:
            return s
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} never reached {statuses}; last: {jobs.job_status(job_id)}")


@pytest.fixture
def cleanup():
    started = []
    yield started
    for job_id in started:
        jobs.cancel_job(job_id)
        shutil.rmtree(jobs.JOBS_DIR / job_id, ignore_errors=True)


def test_job_runs_to_completion_with_progress(cleanup):
    job_id = jobs.start_job("_test_sleep", {"seconds": 2})
    cleanup.append(job_id)
    s = _wait(job_id, jobs.FINISHED)
    assert s["status"] == "done", s
    assert s["result"] == {"slept": 2}
    assert s["elapsed_s"] > 0
    assert "pid" not in s, "internal fields stay internal"


def test_timeout_kills_the_process(cleanup):
    job_id = jobs.start_job("_test_sleep", {"seconds": 120}, timeout_s=4)
    cleanup.append(job_id)
    _wait(job_id, {"running"})
    pid = jobs.read_state(job_id)["pid"]
    s = _wait(job_id, {"timeout"}, limit=30)
    assert "Stopped after 4 s" in s["error"]
    time.sleep(1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_cancel_stops_a_running_job(cleanup):
    job_id = jobs.start_job("_test_sleep", {"seconds": 120})
    cleanup.append(job_id)
    _wait(job_id, {"running"})
    pid = jobs.read_state(job_id)["pid"]
    s = jobs.cancel_job(job_id)
    assert s["status"] == "cancelled"
    time.sleep(1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    # a finished job stays finished — cancelling again is harmless
    assert jobs.cancel_job(job_id)["status"] == "cancelled"


def test_failure_surfaces_the_error_message(cleanup):
    job_id = jobs.start_job("analyze", {"session_id": "does-not-exist", "classifiers": {"selected": ["lda"]}})
    cleanup.append(job_id)
    s = _wait(job_id, jobs.FINISHED)
    assert s["status"] == "failed"
    assert "Session not found" in s["error"]


def test_unknown_job_is_a_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        jobs.job_status("nope")
    assert exc.value.status_code == 404


@pytest.mark.slow
def test_real_fetch_then_analyze_as_jobs(cleanup):
    """eegbci subject 1, runs 4/8/12 (cached, ~8 MB), then LDA — the UI's real path."""
    fid = jobs.start_job("fetch", {"source": "mne", "dataset_name": "eegbci", "subjects": [1], "runs": [4, 8, 12]})
    cleanup.append(fid)
    f = _wait(fid, jobs.FINISHED, limit=180)
    assert f["status"] == "done", f
    assert f["result"]["analysis_classes"] == ["left_hand", "right_hand"]

    aid = jobs.start_job("analyze", {"session_id": f["result"]["session_id"], "mode": "motor_imagery",
                                     "classifiers": {"selected": ["lda"]}})
    cleanup.append(aid)
    a = _wait(aid, jobs.FINISHED, limit=300)
    assert a["status"] == "done", a
    acc = a["result"]["classifiers"][0]["accuracy_mean"]
    print(f"\n[eegbci subject 1, job] lda: {acc:.3f}")
    assert acc == pytest.approx(0.711, abs=0.03)   # matches the fetch audit's direct run
