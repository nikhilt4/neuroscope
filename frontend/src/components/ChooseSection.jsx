function ChooseSection({ data, onModeSelected, selectedMode }) {
  if (!data) {
    return <p className="placeholder-text">Upload a file to see available modes.</p>;
  }

  return (
    <div>
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
          <span className="value">{data.duration_seconds?.toFixed(1)}s</span>
        </div>
        <div className="summary-card">
          <span className="label">Events Found</span>
          <span className="value">{data.events_found ? "Yes" : "No"}</span>
        </div>
      </div>

      {data.motor_channels && (
        <p className="hint-text">
          Motor channels detected:{" "}
          {Object.entries(data.motor_channels)
            .filter(([, present]) => present)
            .map(([ch]) => ch)
            .join(", ") || "none"}
          {!data.available_modes.includes("motor_imagery") &&
            " — motor imagery needs at least two of C3/C4/Cz."}
        </p>
          )}

      <h3 className="subheading">Choose Analysis Mode</h3>
      <div className="mode-row">
        {data.available_modes.map((mode) => (
          <button
            key={mode}
            className={`mode-btn ${selectedMode === mode ? "mode-btn-active" : ""}`}
            onClick={() => onModeSelected(mode)}
          >
            {mode.replace("_", " ")}
            {mode === data.recommended_mode && " ★"}
          </button>
        ))}
      </div>
    </div>
  );
}

export default ChooseSection;