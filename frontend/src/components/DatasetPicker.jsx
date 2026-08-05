import { useState, useEffect } from "react";
import * as api from "../api";
import BidsSelector from "./BidsSelector";

const SOURCES = [
  { id: "mne", label: "MNE", hint: "Built-in reference datasets" },
  { id: "moabb", label: "MOABB", hint: "BCI benchmark datasets" },
  { id: "braindecode", label: "Braindecode", hint: "Deep-learning benchmarks" },
];

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
    // eegbci's run groups mean different tasks — [4,8,12] is left/right hand,
    // [6,10,14] is hands/feet. The backend can't infer which was intended.
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
      if (selected.runs && runVariant) {
        payload.runs = selected.runs[runVariant];
      }
      const data = await api.fetchDataset(payload);
      // BIDS-format datasets (erp_core, OpenNeuro) need a second round trip
      // to pick subject/task/run before they're analysable
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

  return (
    <div>
      <div className="source-row">
        {SOURCES.map((s) => (
          <button
            key={s.id}
            className={`mode-btn ${source === s.id ? "mode-btn-active" : ""}`}
            onClick={() => setSource(s.id)}
            title={s.hint}
          >
            {s.label}
          </button>
        ))}
      </div>

      {isLoading && <p className="placeholder-text">Loading datasets…</p>}

      <div className="dataset-list">
        {datasets.map((ds) => (
          <div
            key={ds.id}
            className={`dataset-card ${selected?.id === ds.id ? "dataset-card-active" : ""}`}
            onClick={() => chooseDataset(ds)}
          >
            <div className="dataset-name">{ds.name || ds.id}</div>
            {ds.description && <div className="dataset-desc">{ds.description}</div>}
            <div className="dataset-meta">
              {ds.n_subjects && <span>{ds.n_subjects} subjects</span>}
              {ds.task && <span>{ds.task.replace("_", " ")}</span>}
              {ds.n_channels && <span>{ds.n_channels} ch</span>}
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <div className="config-group" style={{ marginTop: 20 }}>
          <h4>{selected.name || selected.id}</h4>
          <div className="config-row">
            <label>Subject</label>
            <input
              type="number"
              min="1"
              max={selected.n_subjects || 999}
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
            />
          </div>
          {selected.runs && (
            <div className="config-row">
              <label>Task</label>
              <select value={runVariant} onChange={(e) => setRunVariant(e.target.value)}>
                {Object.keys(selected.runs).map((key) => (
                  <option key={key} value={key}>{key.replace(/_/g, " ")}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      <button
        className="primary-btn"
        onClick={handleFetch}
        disabled={!selected || isFetching}
        style={{ marginTop: 16 }}
      >
        {isFetching ? "Downloading…" : "Fetch Dataset"}
      </button>
      {isFetching && (
        <p className="hint-text">First fetch downloads the data — this can take a few minutes.</p>
      )}

      {error && <div className="error-box">{error}</div>}

      {bidsData && <BidsSelector uploadData={bidsData} onConfigured={onReady} />}
    </div>
  );
}

export default DatasetPicker;