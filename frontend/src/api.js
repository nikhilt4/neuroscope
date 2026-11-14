const API_BASE = "http://127.0.0.1:8000";

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
export function uploadSingle(file) {
  const fd = new FormData();
  fd.append("file", file);
  return postForm("/upload/single", fd);
}

export function uploadMulti(files) {
  const fd = new FormData();
  Array.from(files).forEach((f) => fd.append("files", f));
  return postForm("/upload/multi", fd);
}

export function uploadBids(zipFile) {
  const fd = new FormData();
  fd.append("file", zipFile);
  return postForm("/upload/bids", fd);
}

// ── Configuration round-trips ─────────────────────────────────────────────
export const configureColumns = (mapping) => postJSON("/upload/configure", mapping);
export const configureMat = (mapping) => postJSON("/upload/configure-mat", mapping);
export const configureBids = (selection) => postJSON("/upload/bids-configure", selection);

// ── Datasets ──────────────────────────────────────────────────────────────
export const listDatasets = (source) => request(`/dataset/list/${source}`);
export const fetchDataset = (payload) => postJSON("/dataset/fetch", payload);

// ── Preprocessing + analysis ──────────────────────────────────────────────
export function getPreset({ datasetName, mode = "basic", task } = {}) {
  const params = new URLSearchParams({ mode });
  if (datasetName) params.set("dataset_name", datasetName);
  if (task) params.set("task", task);
  return request(`/preprocessing/preset?${params}`);
}

export const analyze = (payload) => postJSON("/analyze", payload);

// ── Background jobs ───────────────────────────────────────────────────────
// Fetches and analyses run as killable background processes on the server, so a
// stalled download can be cancelled or time out instead of freezing the backend.
export const startFetchJob = (payload) => postJSON("/jobs/fetch", payload);
export const startAnalyzeJob = (payload) => postJSON("/jobs/analyze", payload);
export const getJob = (jobId) => request(`/jobs/${jobId}`);
export const cancelJob = (jobId) => postJSON(`/jobs/${jobId}/cancel`, {});

// ── Playback ──────────────────────────────────────────────────────────────
export const getPlaybackInfo = (sessionId) => request(`/playback/${sessionId}/info`);

/**
 * Opens a real-time replay stream (Server-Sent Events). onMessage receives each
 * parsed frame ({type: "start"|"data"|"event"|"end", ...}). Returns a close()
 * function; pausing, seeking or changing speed = close, then open again.
 */
export function openPlayback(sessionId, { start = 0, speed = 1, channels = [], prefill = 0, preview = false },
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