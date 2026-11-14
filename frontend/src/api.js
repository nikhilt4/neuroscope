import { DEMO } from "./demo/config";
import * as demo from "./demo/demoApi";

// VITE_API_BASE lets a deployed build point at a hosted backend; local dev uses uvicorn's default.
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

/**
 * All backend errors arrive as {"detail": "..."} from FastAPI's HTTPException.
 * Centralising the unwrap means components handle one error shape, not two.
 */
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    // fetch only rejects on network failure — server down, CORS, DNS
    throw new Error("Could not reach the server. Is the backend running?");
  }

  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Server returned ${response.status} with no readable body.`);
  }

  if (!response.ok) {
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail ?? body)
    );
  }
  return body;
}

function postJSON(path, payload) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function postForm(path, formData) {
  // no Content-Type header — the browser sets the multipart boundary itself
  return request(path, { method: "POST", body: formData });
}

// ── Upload ────────────────────────────────────────────────────────────────
function live_uploadSingle(file) {
  const fd = new FormData();
  fd.append("file", file);
  return postForm("/upload/single", fd);
}

function live_uploadMulti(files) {
  const fd = new FormData();
  Array.from(files).forEach((f) => fd.append("files", f));
  return postForm("/upload/multi", fd);
}

function live_uploadBids(zipFile) {
  const fd = new FormData();
  fd.append("file", zipFile);
  return postForm("/upload/bids", fd);
}

// ── Configuration round-trips ─────────────────────────────────────────────
const live_configureColumns = (mapping) => postJSON("/upload/configure", mapping);
const live_configureMat = (mapping) => postJSON("/upload/configure-mat", mapping);
const live_configureBids = (selection) => postJSON("/upload/bids-configure", selection);

// ── Datasets ──────────────────────────────────────────────────────────────
const live_listDatasets = (source) => request(`/dataset/list/${source}`);
const live_fetchDataset = (payload) => postJSON("/dataset/fetch", payload);

// ── Preprocessing + analysis ──────────────────────────────────────────────
function live_getPreset({ datasetName, mode = "basic", task } = {}) {
  const params = new URLSearchParams({ mode });
  if (datasetName) params.set("dataset_name", datasetName);
  if (task) params.set("task", task);
  return request(`/preprocessing/preset?${params}`);
}

const live_analyze = (payload) => postJSON("/analyze", payload);

// ── Background jobs ───────────────────────────────────────────────────────
// Fetches and analyses run as killable background processes on the server, so a
// stalled download can be cancelled or time out instead of freezing the backend.
const live_startFetchJob = (payload) => postJSON("/jobs/fetch", payload);
const live_startAnalyzeJob = (payload) => postJSON("/jobs/analyze", payload);
const live_getJob = (jobId) => request(`/jobs/${jobId}`);
const live_cancelJob = (jobId) => postJSON(`/jobs/${jobId}/cancel`, {});

// ── Playback ──────────────────────────────────────────────────────────────
const live_getPlaybackInfo = (sessionId) => request(`/playback/${sessionId}/info`);

/**
 * Opens a real-time replay stream (Server-Sent Events). onMessage receives each
 * parsed frame ({type: "start"|"data"|"event"|"end", ...}). Returns a close()
 * function; pausing, seeking or changing speed = close, then open again.
 */
function live_openPlayback(sessionId, { start = 0, speed = 1, channels = [], prefill = 0, preview = false },
                             onMessage, onError) {
  // prefill: the server first sends the `prefill` seconds before `start`, so a seek shows a
  // full screen at once; preview: stop after that (scrubbing while paused)
  const params = new URLSearchParams({ start: String(start), speed: String(speed),
                                       prefill: String(prefill), preview: String(preview) });
  if (channels.length) params.set("channels", channels.join(","));
  const es = new EventSource(`${API_BASE}/playback/${sessionId}/stream?${params}`);
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    onMessage(msg);
    if (msg.type === "end") es.close();     // otherwise EventSource auto-reconnects
  };
  es.onerror = () => { es.close(); onError?.(new Error("Playback stream disconnected.")); };
  return () => es.close();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Starts a job and polls it once a second until it finishes.
 * onUpdate(status) fires on every poll ({ job_id, status, stage, elapsed_s }).
 * Resolves with the job's result; rejects with the job's error message
 * (err.cancelled is true when the user cancelled).
 */
export async function runJob(startPromise, onUpdate) {
  const { job_id } = await startPromise;
  onUpdate?.({ job_id, status: "queued", stage: "Starting…", elapsed_s: 0 });
  for (;;) {
    await sleep(1000);
    const s = await getJob(job_id);
    onUpdate?.(s);
    if (s.status === "done") return s.result;
    if (["failed", "timeout", "cancelled"].includes(s.status)) {
      const err = new Error(s.status === "cancelled" ? "Cancelled." : s.error || `Job ${s.status}.`);
      err.cancelled = s.status === "cancelled";
      throw err;
    }
  }
}

// ── Exports ───────────────────────────────────────────────────────────────
// The online demo (DEMO, set at build time) swaps every call for the recorded stand-in
// in demo/demoApi.js. Components import from here either way and never know.
export const uploadSingle = DEMO ? demo.uploadSingle : live_uploadSingle;
export const uploadMulti = DEMO ? demo.uploadMulti : live_uploadMulti;
export const uploadBids = DEMO ? demo.uploadBids : live_uploadBids;
export const getPreset = DEMO ? demo.getPreset : live_getPreset;
export const openPlayback = DEMO ? demo.openPlayback : live_openPlayback;
export const configureColumns = DEMO ? demo.configureColumns : live_configureColumns;
export const configureMat = DEMO ? demo.configureMat : live_configureMat;
export const configureBids = DEMO ? demo.configureBids : live_configureBids;
export const listDatasets = DEMO ? demo.listDatasets : live_listDatasets;
export const fetchDataset = DEMO ? demo.fetchDataset : live_fetchDataset;
export const analyze = DEMO ? demo.analyze : live_analyze;
export const startFetchJob = DEMO ? demo.startFetchJob : live_startFetchJob;
export const startAnalyzeJob = DEMO ? demo.startAnalyzeJob : live_startAnalyzeJob;
export const getJob = DEMO ? demo.getJob : live_getJob;
export const cancelJob = DEMO ? demo.cancelJob : live_cancelJob;
export const getPlaybackInfo = DEMO ? demo.getPlaybackInfo : live_getPlaybackInfo;
