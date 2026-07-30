function ResultsSummary({ data }) {
  return (
    <div className="app">
      <h1>Analysis Ready</h1>
      <p className="subtitle">Dataset capabilities detected</p>

      <div className="summary-grid">
        <div className="summary-card">
          <span className="label">Channels</span>
          <span className="value">{data.n_channels}</span>
        </div>
        <div className="summary-card">
          <span className="label">Sampling Rate</span>
          <span className="value">{data.sampling_rate} Hz</span>
        </div>
        <div className="summary-card">
          <span className="label">Duration</span>
          <span className="value">{data.duration_seconds.toFixed(1)}s</span>
        </div>
        <div className="summary-card">
          <span className="label">Events Found</span>
          <span className="value">{data.events_found ? "Yes" : "No"}</span>
        </div>
      </div>

      {data.event_classes?.length > 0 && (
        <div className="section">
          <h3>Event Classes</h3>
          <div className="chip-row">
            {data.event_classes.map((cls) => (
              <span key={cls} className="chip">{cls}</span>
            ))}
          </div>
        </div>
      )}

      <div className="section">
        <h3>Available Analysis Modes</h3>
        <div className="chip-row">
          {data.available_modes.map((mode) => (
            <span
              key={mode}
              className={`chip mode-chip ${mode === data.recommended_mode ? "recommended" : ""}`}
            >
              {mode.replace("_", " ")}
              {mode === data.recommended_mode && " ★"}
            </span>
          ))}
        </div>
      </div>

      {data.motor_channels && (
        <div className="section">
          <h3>Channel Coverage</h3>
          <ul className="channel-list">
            <li><strong>Motor:</strong> {data.motor_channels.primary_motor.join(", ") || "none"}</li>
            <li><strong>Frontal:</strong> {data.motor_channels.frontal.join(", ") || "none"}</li>
            <li><strong>Parietal:</strong> {data.motor_channels.parietal.join(", ") || "none"}</li>
            <li><strong>Occipital:</strong> {data.motor_channels.occipital.join(", ") || "none"}</li>
          </ul>
        </div>
      )}
    </div>
  );
}

export default ResultsSummary;