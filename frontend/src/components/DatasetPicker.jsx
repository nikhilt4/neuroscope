import { useState, useEffect } from "react";
import * as api from "../api";
import BidsSelector from "./BidsSelector";

const SOURCES = [
  { id: "mne", label: "MNE", hint: "Reference datasets bundled with MNE-Python" },
  { id: "moabb", label: "MOABB", hint: "Standard BCI benchmark datasets" },
  { id: "braindecode", label: "Braindecode", hint: "Datasets formatted for deep learning" },
];

// Download sizes and caveats aren't in the backend metadata, and they matter a
// lot — a synchronous fetch of a multi-hundred-MB dataset looks like a hang.
const DATASET_NOTES = {
  eegbci: { size: "~20 MB", speed: "fast",
            link: "https://physionet.org/content/eegmmidb/1.0.0/" },
  sleep_physionet: { size: "~250 MB", speed: "slow",
            warn: "Downlod is currently unreliable - may hang. Use another dataset if possible.",
            link: "https://physionet.org/content/sleep-edfx/1.0.0/" },
  ssvep: { size: "~100 MB", speed: "medium", warn: "Untested fetch path." },
  erp_core: { size: "~1 GB", speed: "slow",
            warn: "BIDS dataset — needs a second selection step after download.",
            link: "https://osf.io/thsqg/" },
  limo: { size: "~700 MB", speed: "slow", warn: "Downloads per subject." },
  kiloword: { size: "~40 MB", speed: "fast", warn: "Untested fetch path." },
  BNCI2014_001: { size: "~500 MB", speed: "slow",
            link: "https://www.bbci.de/competition/iv/" },
  BNCI2014_002: { size: "~600 MB", speed: "slow" },
  Zhou2016: { size: "~200 MB", speed: "medium" },
  Schirrmeister2017: { size: "~10 GB", speed: "slow",
            warn: "Very large — expect a long download." },
};

const TASK_LABELS = {
  motor_imagery: "Motor Imagery",
  spectral: "Spectral / Sleep",
  erp: "ERP",
  ssvep: "SSVEP",
};

function DatasetPicker({ onReady }) {
  const [source, setSource] = useState("mne");
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [subject, setSubject] = useState(1);
  const [runVariant, setRunVariant] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState(null);
  const [bidsData, setBidsData] = useState(null);

  useEffect(() => {
    setIsLoading(true);
    setSelected(null);
    setBidsData(null);
    setError(null);
    api
      .listDatasets(source)
      .then((res) => setDatasets(res.datasets || []))
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [source]);

  const chooseDataset = (ds) => {
    setSelected(ds);
    setSubject(1);
    setBidsData(null);
    setError(null);
    // eegbci's run groups are different tasks: [4,8,12] left/right hand,
    // [6,10,14] hands/feet. The backend can't infer which was meant.
    setRunVariant(ds.runs ? Object.keys(ds.runs)[0] : null);
  };

  const handleFetch = async () => {
    setIsFetching(true);
    setError(null);
    try {
      const payload = {
        source,
        dataset_name: selected.id,
        subjects: [Number(subject)],
      };
      if (selected.runs && runVariant) payload.runs = selected.runs[runVariant];

      const data = await api.fetchDataset(payload);
      if (data.needs_bids_selection) {
        setBidsData(data);
        return;
      }
      onReady(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsFetching(false);
    }
  };

  const notes = selected ? DATASET_NOTES[selected.id] || {} : {};

  return (
    <div>
      <div className="source-row">
        {SOURCES.map((s) => (
          <button
            key={s.id}
            className={`source-btn ${source === s.id ? "source-btn-active" : ""}`}
            onClick={() => setSource(s.id)}
          >
            <span className="source-label">{s.label}</span>
            <span className="source-hint">{s.hint}</span>
          </button>
        ))}
      </div>

      {isLoading && <p className="placeholder-text">Loading datasets…</p>}

      <div className="dataset-grid">
        {datasets.map((ds) => {
          const n = DATASET_NOTES[ds.id] || {};
          return (
            <div
              key={ds.id}
              className={`dataset-card ${selected?.id === ds.id ? "dataset-card-active" : ""}`}
              onClick={() => chooseDataset(ds)}
            >
              <div className="dataset-card-top">
                <span className="dataset-name">{ds.name || ds.id}</span>
                {ds.task && (
                  <span className="dataset-tag">{TASK_LABELS[ds.task] || ds.task}</span>
                )}
              </div>

              {ds.description && <p className="dataset-desc">{ds.description}</p>}

              <div className="dataset-meta">
                {ds.n_subjects && <span>{ds.n_subjects} subjects</span>}
                {(ds.n_eeg_channels || ds.n_channels) && (
                  <span>{ds.n_eeg_channels || ds.n_channels} channels</span>
                )}
                {ds.sampling_rate && <span>{ds.sampling_rate} Hz</span>}
                {n.size && <span className={`size-tag size-${n.speed}`}>{n.size}</span>}
              </div>

              {ds.classes && (
                <div className="dataset-classes">
                  {ds.classes.map((c) => (
                    <span key={c} className="class-pill">{c.replace(/_/g, " ")}</span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {selected && (
        <div className="fetch-panel">
          <h4>{selected.name || selected.id}</h4>

          {notes.warn && <p className="dataset-warn">{notes.warn}</p>}

          <div className="fetch-row">
            <div className="fetch-field">
              <label>Subject</label>
              <input
                type="number"
                min="1"
                max={selected.n_subjects || 999}
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
              />
              <span className="field-hint">
                Which participant's recording to analyse
                {selected.n_subjects && ` (1–${selected.n_subjects})`}. BCI models are
                usually trained per-person, since brain signals differ between individuals.
              </span>
            </div>

            {selected.runs && (
              <div className="fetch-field">
                <label>Task</label>
                <select value={runVariant} onChange={(e) => setRunVariant(e.target.value)}>
                  {Object.keys(selected.runs).map((key) => (
                    <option key={key} value={key}>{key.replace(/_/g, " ")}</option>
                  ))}
                </select>
                <span className="field-hint">
                  This dataset recorded several tasks — pick which one to classify.
                </span>
              </div>
            )}
          </div>

          {notes.link && (
            <a className="dataset-link" href={notes.link} target="_blank" rel="noreferrer">
              Dataset documentation ↗
            </a>
          )}

          <button className="primary-btn" onClick={handleFetch} disabled={isFetching}>
            {isFetching ? "Downloading…" : "Fetch Dataset"}
          </button>

          {isFetching && (
            <p className="hint-text">
              Downloading {notes.size || "the dataset"} — this runs synchronously, so the
              page will sit here until it finishes. Subsequent fetches are cached and instant.
            </p>
          )}
        </div>
      )}

      {error && <div className="error-box">{error}</div>}
      {bidsData && <BidsSelector uploadData={bidsData} onConfigured={onReady} />}
    </div>
  );
}

export default DatasetPicker;