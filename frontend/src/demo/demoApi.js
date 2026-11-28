/**
 * Stand-in backend for the online demo. Same function signatures as api.js, answered
 * from files recorded by backend/demo/record_demo.py (real backend output, not mocks).
 *
 * What is genuinely live in the demo: the whole UI, dataset → mode → preprocessing →
 * decoders → results, and playback, which streams the recorded samples through the
 * same frame protocol as the server's SSE stream. What isn't: training. A run returns
 * the recorded result for that dataset, mode and evaluation, cut down to the decoders
 * picked. Preprocessing and hyperparameter edits don't change it, and the UI says so.
 */

const BASE = `${import.meta.env.BASE_URL}demo/`;
const cache = new Map();

function loadJSON(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(BASE + name).then((r) => {
      if (!r.ok) throw new Error(`Demo file missing: ${name}`);
      return r.json();
    }));
  }
  return cache.get(name);
}

const manifest = () => loadJSON("manifest.json");

// session_id (from a recorded fetch) → dataset id, so later calls know which recording to use
const sessions = new Map();

const fullAppOnly = (what) => () => Promise.reject(new Error(
  `${what} needs the full app. This online demo runs without a server, on two recorded datasets.`));

export const uploadSingle = fullAppOnly("Uploading your own files");
export const uploadMulti = fullAppOnly("Uploading your own files");
export const uploadBids = fullAppOnly("Uploading a BIDS dataset");
export const configureColumns = fullAppOnly("Column mapping");
export const configureMat = fullAppOnly("MATLAB import");
export const configureBids = fullAppOnly("BIDS selection");
export const fetchDataset = fullAppOnly("Synchronous fetching");
export const analyze = fullAppOnly("Synchronous analysis");

export async function listDatasets(source) {
  if (!["mne", "moabb"].includes(source)) return { datasets: [] };
  return loadJSON(`datasets__${source}.json`);
}

export async function getPreset({ datasetName, mode = "basic" } = {}) {
  try { return await loadJSON(`preset__${datasetName}__${mode}.json`); } catch { return {}; }
}

// ── Simulated background jobs ───────────────────────────────────────────────
// A job is a list of {stage, ms} steps plus the result. getJob reports the stage the
// elapsed time has reached, so the wizard's progress footer behaves as it does live.
const jobs = new Map();
let nextJob = 1;

function makeJob(steps, result) {
  const job_id = `demo-${nextJob++}`;
  jobs.set(job_id, { steps, result, t0: performance.now(), cancelled: false });
  return { job_id };
}

export async function getJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) throw new Error("Job not found.");
  const elapsed = performance.now() - job.t0;
  const base = { job_id: jobId, elapsed_s: elapsed / 1000 };
  if (job.cancelled) return { ...base, status: "cancelled" };
  let t = 0;
  for (const s of job.steps) {
    t += s.ms;
    if (elapsed < t) return { ...base, status: "running", stage: s.stage };
  }
  return { ...base, status: "done", stage: "Done", result: await job.result };
}

export async function cancelJob(jobId) {
  const job = jobs.get(jobId);
  if (job) job.cancelled = true;
  return { job_id: jobId, status: "cancelled" };
}

export async function startFetchJob({ dataset_name, subjects }) {
  const m = await manifest();
  const ds = m.datasets[dataset_name];
  if (!ds) throw new Error("This dataset isn't part of the online demo. Run NeuroScope locally to load it.");
  const subject = subjects?.[0];
  if (subject !== ds.subject) {
    throw new Error(`The online demo includes subject ${ds.subject} of this dataset. Run NeuroScope locally to load any subject.`);
  }
  const result = loadJSON(`fetch__${dataset_name}.json`).then((r) => {
    sessions.set(r.session_id, dataset_name);
    return r;
  });
  return makeJob([
    { stage: `Loading subject ${subject} (recorded ahead of time)`, ms: 900 },
    { stage: "Reading the recording", ms: 700 },
    { stage: "Detecting channels, montage and events", ms: 800 },
  ], result);
}

const EVAL_LABEL = { single: "single-subject", within_subject: "within-subject", cross_subject: "cross-subject" };

export async function startAnalyzeJob({ session_id, mode, classifiers, evaluation = "single" }) {
  const m = await manifest();
  const dataset = sessions.get(session_id);
  if (!dataset) throw new Error("Load a dataset first.");
  const recorded = m.results[dataset]?.[mode] || [];
  if (!recorded.includes(evaluation)) {
    throw new Error(evaluation === "single"
      ? "This analysis mode wasn't recorded for the demo. Pick another mode, or run NeuroScope locally."
      : `The demo has ${EVAL_LABEL[evaluation]} results for BCI IV 2a in motor imagery mode only. Switch to that mode, or pick Single subject.`);
  }
  const full = await loadJSON(`result__${dataset}__${mode}__${evaluation}.json`);
  const selected = classifiers?.selected || [];
  const kept = full.classifiers.filter((c) => selected.includes(c.classifier));
  const missing = selected.filter((id) => !kept.some((c) => c.classifier === id));
  if (!kept.length) {
    throw new Error("None of the chosen decoders are in this recorded run. EEGNet was left out of the 9-subject runs because of training time; pick another decoder.");
  }
  const ok = kept.filter((c) => c.accuracy_mean != null);
  const best = ok.length ? ok.reduce((a, b) => (b.accuracy_mean > a.accuracy_mean ? b : a)) : null;
  const result = {
    ...full, classifiers: kept, best_classifier: best?.classifier ?? null,
    demo: { recorded: m.recorded, missing },
  };
  const steps = [
    { stage: "Loading data", ms: 600 },
    { stage: "Preprocessing", ms: 700 },
    { stage: "Extracting features", ms: 500 },
    ...kept.map((c, i) => ({ stage: `Training ${c.classifier} (${i + 1} of ${kept.length})`,
                            ms: c.classifier === "eegnet" ? 1400 : 450 })),
    { stage: "Computing spectra and scalp maps", ms: 600 },
  ];
  return makeJob(steps, result);
}

// ── Playback ────────────────────────────────────────────────────────────────

async function playbackFor(sessionId) {
  const m = await manifest();
  const key = m.datasets[sessions.get(sessionId)]?.playback;
  if (!key) throw new Error("In the online demo, playback is available for Sleep-EDF.");
  return key;
}

// left / right / midline from a 10-20 name, as backend/core/playback._hemisphere does
const hemisphere = (name) => {
  const first = name.split("-")[0].trim();
  if (/z$/i.test(first)) return "midline";
  const n = parseInt(first.replace(/\D/g, ""), 10);
  return Number.isNaN(n) ? "midline" : (n % 2 ? "left" : "right");
};

export async function getPlaybackInfo(sessionId) {
  const info = await loadJSON(`playback__${await playbackFor(sessionId)}.json`);
  // the recording was saved before the page viewer existed; add what it needs
  return {
    ...info,
    channel_meta: info.channels.map((name) => ({ name, type: "eeg", group: hemisphere(name) })),
    montages: ["referential"],
  };
}

const binCache = new Map();
const MIN_VIEW_SFREQ = 100;      // same display floor as the backend's page_window
const TARGET_POINTS = 2600;
function loadSamples(key) {
  if (!binCache.has(key)) {
    binCache.set(key, fetch(`${BASE}playback__${key}.bin`)
      .then((r) => { if (!r.ok) throw new Error("Demo recording missing."); return r.arrayBuffer(); })
      .then((buf) => new Int16Array(buf)));
  }
  return binCache.get(key);
}

/** Same shape as the backend's page_window, cut from the recorded excerpt. */
export async function getPlaybackWindow(sessionId, { start = 0, duration = 10 } = {}) {
  const key = await playbackFor(sessionId);
  const [info, raw] = await Promise.all([getPlaybackInfo(sessionId), loadSamples(key)]);
  const { sfreq, n_samples: n, scale } = info;
  const dur = Math.min(Math.max(duration, 1), 60);
  const s0 = Math.min(Math.max(start, 0), Math.max(0, n / sfreq - dur));
  const i0 = Math.floor(s0 * sfreq), i1 = Math.min(n, Math.floor((s0 + dur) * sfreq));
  const step = Math.max(1, Math.min(Math.ceil((i1 - i0) / TARGET_POINTS), Math.floor(sfreq / MIN_VIEW_SFREQ) || 1));
  const data = info.channels.map((_, c) => {
    const out = [];
    for (let i = i0; i < i1; i += step) out.push(raw[c * n + i] / scale);
    return out;
  });
  return {
    start: s0, duration: dur, sfreq: sfreq / step, montage: "referential",
    channels: info.channel_meta, data,
    events: info.events.filter((e) => e.t >= s0 && e.t < s0 + dur),
  };
}
