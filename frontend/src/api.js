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