"""
Job endpoints: start a fetch or an analysis in the background, poll it, cancel it.

The old synchronous routes (POST /dataset/fetch, POST /analyze) still work — these
run the same functions (run_fetch / run_analysis) in a separate process.
"""
from typing import Optional

from fastapi import APIRouter

from backend import jobs
from backend.models import AnalysisRequest, DatasetFetchRequest

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/fetch")
async def start_fetch(request: DatasetFetchRequest, timeout_s: Optional[float] = None):
    """timeout_s defaults to 600 s — long enough for the largest per-subject download."""
    return {"job_id": jobs.start_job("fetch", request.model_dump(), timeout_s)}


@router.post("/analyze")
async def start_analyze(request: AnalysisRequest, timeout_s: Optional[float] = None):
    """
    timeout_s defaults to 1800 s (EEGNet over several folds can take minutes), or 3600 s for
    multi-subject runs, which may also download each new subject first.
    """
    if timeout_s is None and request.evaluation != "single":
        timeout_s = 3600
    return {"job_id": jobs.start_job("analyze", request.model_dump(exclude_unset=True), timeout_s)}


@router.get("/{job_id}")
async def get_job(job_id: str):
    return jobs.job_status(job_id)


@router.post("/{job_id}/cancel")
async def cancel(job_id: str):
    return jobs.cancel_job(job_id)
