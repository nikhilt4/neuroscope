import { Icon } from "../ui/Icons";
import { prettyLabel } from "../../labels";

// Familiar task name + the method it runs (NEUROSCOPE_CONTEXT.md 2.3).
// The internal strings stay as they are — they're stored in presets and sessions.
export const MODE_LABELS = {
  motor_imagery: "Motor imagery (CSP)",
  spectral: "Spectral (band power)",
  basic: "Basic (statistics)",
  erp: "ERP (time-domain)",
};

const SOURCE_LABELS = { mne: "MNE", moabb: "MOABB", braindecode: "braindecode", upload: "Upload", bids: "BIDS" };

function Row({ label, value, empty = "Not chosen yet" }) {
  const missing = value === null || value === undefined || value === "";
  return (
    <div className={`wz-sum-row ${missing ? "is-empty" : ""}`}>
      <dt>{label}</dt>
      <dd>{missing ? empty : value}</dd>
    </div>
  );
}

export function fmtDuration(s) {
  if (!s && s !== 0) return null;
  if (s < 90) return `${s.toFixed(0)} s`;
  if (s < 2 * 3600) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;   // a Sleep-EDF night is ~23 h, not "1401 min"
}

/**
 * Sticky "what you've set up so far" panel. Fills in step by step, so at any
 * point you can see which subject, mode, filters and models a run will use —
 * the thing that was missing when subject-1 and subject-5 runs got compared.
 */
const EVAL_LABELS = { within_subject: "within-subject", cross_subject: "leave-one-subject-out" };

function SummaryPanel({ data, mode, preset, overrides, classifiers, evaluation }) {
  const multi = evaluation && evaluation.type !== "single";
  const filter = (() => {
    if (!data) return null;
    if (data.is_epochs) return "Applied by the source";
    const lo = overrides.bandpass_low ?? preset.bandpass_low;
    const hi = overrides.bandpass_high ?? preset.bandpass_high;
    if (lo != null && hi != null) return `${lo}–${hi} Hz bandpass`;
    if (lo != null) return `${lo} Hz high-pass`;     // e.g. ERP CORE: 0.1 Hz, no upper edge
    if (hi != null) return `${hi} Hz low-pass`;
    return preset && Object.keys(preset).length ? "None" : null;
  })();

  const datasetName = data
    ? (data.display_name || data.dataset_name || data.format?.toUpperCase() || "Uploaded file")
    : null;

  return (
    <aside className="wz-summary" aria-label="Run summary">
      <h2><Icon name="layers" />Run summary</h2>

      <section>
        <h3>Data</h3>
        <dl>
          <Row label="Dataset" value={datasetName} empty="Nothing loaded" />
          <Row label="Source" value={data && (SOURCE_LABELS[data.source] || data.source)} empty="—" />
          <Row label={multi ? "Subjects" : "Subject"} value={
            multi && evaluation.subjects.length ? evaluation.subjects.join(", ")
              : data?.subjects?.length ? data.subjects.join(", ")
              : data && (data.source === "upload" || data.source === "bids") ? "Your recording" : null} empty="—" />
          <Row label="Channels" value={data && `${data.n_channels} @ ${data.sampling_rate} Hz`} empty="—" />
          <Row label={data?.is_epochs ? "Trials" : "Duration"}
               value={data && (data.is_epochs ? data.n_epochs : fmtDuration(data.duration_seconds))} empty="—" />
          {/* analysis_classes = names after the preset's renames (eegbci T1/T2 -> left/right hand) */}
          <Row label="Classes" value={(() => {
            const cls = data?.analysis_classes ?? data?.event_classes;
            return cls?.length ? cls.map(prettyLabel).join(" · ") : null;
          })()} empty="—" />
        </dl>
      </section>

      <section>
        <h3>Analysis</h3>
        <dl>
          <Row label="Mode" value={mode && (MODE_LABELS[mode] || mode)} />
          <Row label="Filtering" value={filter} />
          <Row label="Models" value={classifiers?.selected?.length ? `${classifiers.selected.length} selected` : null} />
          <Row label="Validation" value={classifiers
            ? multi
              ? `${evaluation.subjects.length} subjects, ${EVAL_LABELS[evaluation.type]}`
              : `${classifiers.n_folds}-fold stratified`
            : null} />
        </dl>
        {classifiers?.selected?.length > 0 && (
          <ul className="wz-sum-chips">{classifiers.selected.map((c) => <li key={c}>{c}</li>)}</ul>
        )}
      </section>

      {data?.subjects?.length > 0 && (
        <p className="wz-sum-note">
          <Icon name="info" />Accuracy can differ hugely between subjects. Compare runs on the same subject.
        </p>
      )}
    </aside>
  );
}

export default SummaryPanel;
