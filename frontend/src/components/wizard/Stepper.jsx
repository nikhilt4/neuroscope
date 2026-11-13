import { Icon } from "../ui/Icons";

/**
 * Clickable progress stepper. A step is clickable when it's at or before
 * `reachable` (the furthest step the current data/config allows), so you can
 * jump back to fix something and forward again without re-running anything.
 */
function Stepper({ steps, current, reachable, onGo }) {
  const pct = (current / (steps.length - 1)) * 100;
  return (
    <nav className="wz-stepper" aria-label="Analysis steps">
      <div className="wz-stepper-track" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </div>
      <ol>
        {steps.map((s, i) => {
          const state = i < current ? "done" : i === current ? "active" : "todo";
          const clickable = i !== current && i <= reachable;
          return (
            <li key={s.name} className={`wz-step wz-step-${state}`}>
              <button
                type="button"
                onClick={() => clickable && onGo(i)}
                disabled={!clickable && i !== current}
                aria-current={i === current ? "step" : undefined}
              >
                <span className="wz-step-dot">
                  <Icon name={state === "done" ? "check" : s.icon} />
                </span>
                <span className="wz-step-text">
                  <small>Step {i + 1}</small>
                  {s.name}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export default Stepper;
