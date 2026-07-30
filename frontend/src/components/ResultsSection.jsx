function ResultsSection({ results }) {
  if (!results) {
    return <p className="placeholder-text">Run the analysis above to see results.</p>;
  }

  const best = results.classifiers.reduce((a, b) => (a.accuracy > b.accuracy ? a : b));

  return (
    <div>
      <div className="summary-grid">
        <div className="summary-card">
          <span className="label">File</span>
          <span className="value" style={{ fontSize: "1rem" }}>{results.dataset_summary.file}</span>
        </div>
        <div className="summary-card">
          <span className="label">Duration</span>
          <span className="value">{results.dataset_summary.duration}</span>
        </div>
        <div className="summary-card">
          <span className="label">Channels</span>
          <span className="value">{results.dataset_summary.channels}</span>
        </div>
        <div className="summary-card">
          <span className="label">Epochs</span>
          <span className="value">{results.dataset_summary.epochs}</span>
        </div>
      </div>

      <h3 className="subheading">Classifier Comparison</h3>
      <table className="results-table">
        <thead>
          <tr>
            <th>Classifier</th>
            <th>Accuracy</th>
            <th>Std Dev</th>
            <th>Inference Time</th>
          </tr>
        </thead>
        <tbody>
          {results.classifiers.map((c) => (
            <tr key={c.name} className={c.name === best.name ? "best-row" : ""}>
              <td>{c.name} {c.name === best.name && "★"}</td>
              <td>{c.accuracy.toFixed(1)}%</td>
              <td>± {c.std.toFixed(1)}%</td>
              <td>{c.time_ms}ms</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="results-actions">
        <button className="primary-btn">Download PDF Report</button>
      </div>
    </div>
  );
}

export default ResultsSection;