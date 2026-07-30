function ProgressBar({ steps, current }) {
  return (
    <div className="progress-bar">
      {steps.map((label, i) => (
        <div key={label} className="progress-step">
          <div
            className={`progress-dot ${
              i < current ? "done" : i === current ? "active" : ""
            }`}
          >
            {i < current ? "✓" : i + 1}
          </div>
          <span className={`progress-label ${i === current ? "active" : ""}`}>
            {label}
          </span>
          {i < steps.length - 1 && (
            <div className={`progress-line ${i < current ? "done" : ""}`} />
          )}
        </div>
      ))}
    </div>
  );
}

export default ProgressBar;