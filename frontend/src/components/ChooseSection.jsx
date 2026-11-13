import { Icon } from "./ui/Icons";
import "./choose/Choose.css";

/**
 * Step 2 — pick the analysis mode.
 *
 * Cards are titled by the familiar task name, with the feature method it
 * actually runs underneath (NEUROSCOPE_CONTEXT.md 2.3: "motor_imagery" really
 * means "CSP"). Availability mirrors backend/core/detector.py get_available_modes();
 * a locked card says why instead of disappearing.
 */
const MODES = [
  {
    id: "motor_imagery",
    icon: "classify",
    name: "Motor imagery",
    method: "CSP spatial filtering",
    line: "Finds where on the scalp your classes differ.",
    locked: (d) => !d.motor_channels?.motor_imagery_possible
      ? "Needs both C3 and C4"
      : !d.events_found && !d.is_epochs ? "Needs event markers" : null,
  },
  {
    id: "spectral",
    icon: "results",
    name: "Spectral",
    method: "Band power",
    line: "Strength of each brain rhythm, per channel.",
    locked: () => "Needs at least one EEG channel",
  },
  {
    id: "basic",
    icon: "wave",
    name: "Basic",
    method: "Signal statistics",
    line: "Simple per-channel stats. Works on any data.",
    locked: () => null,
  },
];

function ChooseSection({ data, onModeSelected, selectedMode }) {
  if (!data) {
    return <p className="ch-empty">Load some data first and the available analyses will appear here.</p>;
  }

  const available = new Set(data.available_modes || []);

  return (
    <div className="ch">
      <div className="ch-grid" role="radiogroup" aria-label="Analysis mode">
        {MODES.map((m) => {
          const ok = available.has(m.id);
          const reason = ok ? null : m.locked(data) || "Not available for this data";
          const active = selectedMode === m.id;
          return (
            <button
              key={m.id}
              role="radio"
              aria-checked={active}
              disabled={!ok}
              className={`ch-card ${active ? "is-active" : ""} ${ok ? "" : "is-locked"}`}
              onClick={() => ok && onModeSelected(m.id)}
            >
              <div className="ch-top">
                <span className="ch-icon"><Icon name={m.icon} /></span>
                {m.id === data.recommended_mode && ok && <span className="ch-rec">Recommended</span>}
                <span className="ch-radio" aria-hidden="true">{active && <Icon name="check" />}</span>
              </div>
              <h3>{m.name}</h3>
              <span className="ch-method">{m.method}</span>
              <p className="ch-line">{m.line}</p>
              {reason && <p className="ch-reason">{reason}</p>}
            </button>
          );
        })}
      </div>

      <p className="ch-note">
        <Icon name="info" />
        <span>Only affects the classical decoders. Riemannian and deep decoders read the raw trials in every mode.</span>
      </p>
    </div>
  );
}

export default ChooseSection;
