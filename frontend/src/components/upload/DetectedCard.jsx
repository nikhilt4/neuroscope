import ChannelMap from "../wizard/ChannelMap";
import { Icon } from "../ui/Icons";
import { MODE_LABELS, fmtDuration } from "../wizard/SummaryPanel";

function Stat({ icon, tone, label, value }) {
  return (
    <div className="up-stat">
      <span className={`up-stat-icon tone-${tone}`}><Icon name={icon} /></span>
      <div>
        <div className="up-stat-label">{label}</div>
        <div className="up-stat-value">{value}</div>
      </div>
    </div>
  );
}

/**
 * The payoff for loading data: what NeuroScope found in it, before any
 * analysis runs. Everything here comes from detect_capabilities on the backend.
 */
function DetectedCard({ data, onChange }) {
  const motor = data.motor_channels || {};
  // names the analysis will use, after the preset's renames (eegbci: T1/T2 -> left/right hand)
  const classes = data.analysis_classes ?? data.event_classes ?? [];
  const renamed = data.event_classes && classes.join() !== data.event_classes.join();
  const duration = data.is_epochs ? data.n_epochs : fmtDuration(data.duration_seconds);

  return (
    <section className="up-detected">
      <header>
        <span className="up-detected-badge"><Icon name="check" />Data loaded</span>
        <h2>{data.display_name || data.dataset_name || "Your recording"}</h2>
        <p>
          {data.display_name && data.dataset_name ? <span className="up-id">{data.dataset_name}</span> : null}
          {data.subjects?.length ? `Subject ${data.subjects.join(", ")} · ` : ""}
          {data.is_epochs ? "Already split into trials by the source" : "Continuous recording"}
        </p>
        <button className="up-link" onClick={onChange}>Use different data</button>
      </header>

      <div className="up-detected-body">
        <ChannelMap channels={data.channels} />

        <div className="up-detected-info">
          <div className="up-stats">
            <Stat icon="wave" tone="blue" label="Channels" value={data.n_channels} />
            <Stat icon="bolt" tone="violet" label="Sampling" value={`${data.sampling_rate} Hz`} />
            <Stat icon="clock" tone="green" label={data.is_epochs ? "Trials" : "Length"} value={duration} />
            <Stat icon="layers" tone="pink" label="Classes" value={classes.length} />
          </div>

          <div className="up-facts">
            <div>
              <h4>Classes to compare</h4>
              {classes.length ? (
                <>
                  <ul className="up-chips">{classes.map((c) => <li key={c}>{String(c).replace(/_/g, " ")}</li>)}</ul>
                  {renamed && (
                    <p className="up-muted"><Icon name="info" />From the recording&apos;s event markers {data.event_classes.join(", ")}, renamed by the dataset preset.</p>
                  )}
                </>
              ) : (
                <p className="up-warn"><Icon name="alert" />No event markers, so there&apos;s nothing to classify yet.</p>
              )}
            </div>
            <div>
              <h4>Motor channels</h4>
              <p className={motor.motor_imagery_possible ? "up-ok" : "up-muted"}>
                <Icon name={motor.motor_imagery_possible ? "check" : "info"} />
                {motor.motor_imagery_possible
                  ? `C3 and C4 present${motor.extended_motor?.length ? `, plus ${motor.extended_motor.length} nearby` : ""}`
                  : "C3 and C4 not both present, so CSP mode is unavailable"}
              </p>
            </div>
            <div>
              <h4>Analyses this data supports</h4>
              <ul className="up-chips up-chips-modes">
                {(data.available_modes || []).map((m) => (
                  <li key={m} className={m === data.recommended_mode ? "is-rec" : ""}>
                    {MODE_LABELS[m] || m}{m === data.recommended_mode && <b> · recommended</b>}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default DetectedCard;
