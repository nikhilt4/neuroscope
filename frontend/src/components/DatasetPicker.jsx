import { useState, useEffect } from "react";
import * as api from "../api";
import BidsSelector from "./BidsSelector";
import { Icon } from "./ui/Icons";
import { DEMO } from "../demo/config";
import { prettyLabel } from "../labels";

const SOURCES = [
  { id: "mne", label: "MNE", hint: "Reference datasets bundled with MNE-Python" },
  { id: "moabb", label: "MOABB", hint: "Standard BCI benchmark datasets" },
  { id: "braindecode", label: "braindecode", hint: "Datasets formatted for deep learning" },
];

// Download sizes aren't in the backend metadata, and they matter: a synchronous
// fetch of a few hundred MB looks like a hang. `measured` sizes come from this
// machine's cache on 13 Nov 2026 (per subject where it varies); the rest are estimates.
const DATASET_NOTES = {
  eegbci: { size: "8 MB", measured: true, speed: "fast", link: "https://physionet.org/content/eegmmidb/1.0.0/" },
  sleep_physionet: { size: "51 MB", measured: true, speed: "slow",
    warn: "MNE's downloader for this dataset can hang indefinitely. If it does, download the two EDF files from PhysioNet manually.",
    link: "https://physionet.org/content/sleep-edfx/1.0.0/" },
  ssvep: { size: "58 MB", measured: true, speed: "medium", warn: "This fetch path hasn't been tested end to end yet." },
  erp_core: { size: "118 MB", measured: true, speed: "medium",
    warn: "One subject, Flankers task (ERN), 33 channels at 1024 Hz. No ERP-specific mode yet, so results are exploratory." },
  limo: { size: "335 MB", measured: true, speed: "slow", warn: "Downloads per subject." },
  kiloword: { size: "28 MB", measured: true, speed: "fast", warn: "This fetch path hasn't been tested end to end yet." },
  BNCI2014_001: { size: "86 MB", measured: true, speed: "medium", link: "https://www.bbci.de/competition/iv/" },
  BNCI2014_002: { size: "~600 MB", speed: "slow" },
  Zhou2016: { size: "~85 MB", speed: "medium" },
  Schirrmeister2017: { size: "490 MB", measured: true, speed: "slow", warn: "Large per subject. The first fetch takes a while." },
};

// the demo recorded one MNE and one MOABB dataset; braindecode would be an empty tab
const VISIBLE_SOURCES = DEMO ? SOURCES.filter((s) => s.id !== "braindecode") : SOURCES;


function DatasetPicker({ onReady }) {
  const [source, setSource] = useState("moabb");
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [subject, setSubject] = useState(1);
  const [runVariant, setRunVariant] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [job, setJob] = useState(null);          // { job_id, stage, elapsed_s } while fetching
  const [error, setError] = useState(null);
  const [bidsData, setBidsData] = useState(null);

  useEffect(() => {
    setIsLoading(true);
    setSelected(null);
    setBidsData(null);
    setError(null);
    api.listDatasets(source)
      .then((res) => setDatasets(res.datasets || []))
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [source]);

  const chooseDataset = (ds) => {
    setSelected(ds);
    setSubject(ds.demo_subject ?? 1);   // the demo has one recorded subject per dataset
    setBidsData(null);
    setError(null);
    // eegbci's run groups are different tasks: [4,8,12] left/right hand,
    // [6,10,14] hands/feet. The backend can't infer which was meant.
    setRunVariant(ds.runs ? Object.keys(ds.runs)[0] : null);
  };

  const maxSubject = selected?.n_subjects || 999;
  const clampSubject = (v) => Math.min(maxSubject, Math.max(1, Number(v) || 1));

  const handleFetch = async () => {
    setIsFetching(true);
    setError(null);
    setJob(null);
    try {
      const payload = { source, dataset_name: selected.id, subjects: [clampSubject(subject)] };
      if (selected.runs && runVariant) payload.runs = selected.runs[runVariant];
      // background job: live stage + cancel, and a stalled download times out
      // server-side instead of freezing the backend
      const data = await api.runJob(api.startFetchJob(payload), setJob);
      // the backend response doesn't echo the subject; keep it for the summary panel
      const withContext = {
        ...data, subjects: payload.subjects, runs: payload.runs,
        display_name: selected.name || selected.id,   // "BCI Competition IV 2a", not "BNCI2014_001"
        task: selected.task,                           // "erp", "motor_imagery"… — used for decoder suitability notes
        n_subjects_total: selected.n_subjects,         // enables multi-subject evaluation in Step 4
      };
      if (data.needs_bids_selection) { setBidsData(withContext); return; }
      onReady(withContext);
    } catch (err) {
      if (!err.cancelled) setError(err.message);
    } finally {
      setIsFetching(false);
      setJob(null);
    }
  };

  const cancelFetch = () => job?.job_id && api.cancelJob(job.job_id).catch(() => {});

  const notes = selected ? DATASET_NOTES[selected.id] || {} : {};

  return (
    <div className="dp">
      <div className="dp-sources" role="tablist" aria-label="Dataset library">
        {VISIBLE_SOURCES.map((s) => (
          <button key={s.id} role="tab" aria-selected={source === s.id}
                  className={`dp-source ${source === s.id ? "is-active" : ""}`} onClick={() => setSource(s.id)}>
            <b>{s.label}</b><small>{s.hint}</small>
          </button>
        ))}
      </div>

      {isLoading && <div className="dp-grid">{[0, 1, 2].map((i) => <div key={i} className="dp-card dp-skeleton" />)}</div>}

      {!isLoading && (
        <div className="dp-grid">
          {datasets.map((ds) => {
            const n = DATASET_NOTES[ds.id] || {};
            const active = selected?.id === ds.id;
            return (
              <button key={ds.id} className={`dp-card ${active ? "is-active" : ""}`} onClick={() => chooseDataset(ds)}>
                <div className="dp-card-top">
                  <b>{ds.name || ds.id}</b>
                  {active && <span className="dp-check"><Icon name="check" /></span>}
                </div>
                <div className="dp-card-tags">
                  {n.size && (
                    <span className={`dp-size dp-${n.speed}`} title={n.measured ? "Measured download size" : "Estimated size"}>
                      {/* colour still shows download speed; the word was one label too many */}
                      {n.size}{!n.measured && " est."}
                    </span>
                  )}
                </div>
                {ds.description && <p>{ds.description}</p>}
                <div className="dp-meta">
                  {ds.n_subjects && <span><Icon name="user" />{ds.n_subjects}</span>}
                  {(ds.n_eeg_channels || ds.n_channels) && <span><Icon name="wave" />{ds.n_eeg_channels || ds.n_channels} ch</span>}
                  {ds.sampling_rate && <span><Icon name="bolt" />{ds.sampling_rate} Hz</span>}
                </div>
                {ds.classes && (
                  <ul className="dp-classes">{ds.classes.map((c) => <li key={c}>{prettyLabel(c)}</li>)}</ul>
                )}
              </button>
            );
          })}
        </div>
      )}

      {selected && (
        <div className="dp-fetch">
          <div className="dp-fetch-head">
            <div>
              <small>Selected</small>
              <h3>{selected.name || selected.id}</h3>
            </div>
            {notes.link && (
              <a className="up-link" href={notes.link} target="_blank" rel="noreferrer">
                Documentation<Icon name="external" />
              </a>
            )}
          </div>

          {notes.warn && <p className="up-warn"><Icon name="alert" />{notes.warn}</p>}

          <div className="dp-fields">
            <div className="dp-field">
              <label htmlFor="dp-subject">Subject</label>
              <div className="dp-stepper">
                <button type="button" aria-label="Previous subject" disabled={isFetching || DEMO}
                        onClick={() => setSubject((s) => clampSubject(Number(s) - 1))}>−</button>
                <input id="dp-subject" type="number" min="1" max={maxSubject} value={subject}
                       disabled={isFetching || DEMO}
                       onChange={(e) => setSubject(e.target.value)} onBlur={() => setSubject(clampSubject(subject))} />
                <button type="button" aria-label="Next subject" disabled={isFetching || DEMO}
                        onClick={() => setSubject((s) => clampSubject(Number(s) + 1))}>+</button>
                {selected.n_subjects && <span className="dp-of">of {selected.n_subjects}</span>}
              </div>
              <small>
                {DEMO && <><b>The online demo includes subject {selected.demo_subject}.</b> </>}
                Which participant to analyse. BCI decoders are usually trained per person, and
                accuracy varies a lot between people. On BCI IV 2a, subject 1 reaches ~83%; subject 5 only ~50%.
              </small>
            </div>

            {selected.runs && (
              <div className="dp-field">
                <label htmlFor="dp-task">Task</label>
                <select id="dp-task" value={runVariant} onChange={(e) => setRunVariant(e.target.value)} disabled={isFetching}>
                  {Object.keys(selected.runs).map((key) => <option key={key} value={key}>{key.replace(/_/g, " ")}</option>)}
                </select>
                <small>This dataset recorded several tasks. Pick which one to classify.</small>
              </div>
            )}
          </div>

          <div className="up-actions">
            <button className="wz-btn" onClick={handleFetch} disabled={isFetching}>
              {isFetching
                ? <><span className="wz-spinner" />Fetching… {Math.round(job?.elapsed_s ?? 0)}s</>
                : <>Fetch subject {clampSubject(subject)}<Icon name="arrowRight" /></>}
            </button>
            {isFetching && (
              <button className="wz-btn wz-btn-ghost" onClick={cancelFetch} disabled={!job?.job_id}>
                <Icon name="x" />Cancel
              </button>
            )}
            <span className="up-hint">
              {isFetching
                ? <><b>{job?.stage || "Starting…"}</b>. The first fetch downloads {notes.size || "the dataset"}; after that it&apos;s cached. Stops automatically after 10 minutes.</>
                : DEMO ? "Recorded ahead of time, so it loads in a couple of seconds."
                : `First fetch downloads ${notes.size || "the data"}; repeats come from the local cache.`}
            </span>
          </div>
        </div>
      )}

      {error && <div className="wz-alert"><Icon name="alert" /><span>{error}</span></div>}
      {bidsData && <BidsSelector uploadData={bidsData} onConfigured={onReady} />}
    </div>
  );
}

export default DatasetPicker;
