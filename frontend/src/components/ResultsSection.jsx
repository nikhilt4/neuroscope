// function ResultsSection({ results }) {
//   if (!results) {
//     return <p className="placeholder-text">Run the analysis above to see results.</p>;
//   }

//   const best = results.classifiers.reduce((a, b) => (a.accuracy > b.accuracy ? a : b));

//   return (
//     <div>
//       <div className="summary-grid">
//         <div className="summary-card">
//           <span className="label">File</span>
//           <span className="value" style={{ fontSize: "1rem" }}>{results.dataset_summary.file}</span>
//         </div>
//         <div className="summary-card">
//           <span className="label">Duration</span>
//           <span className="value">{results.dataset_summary.duration}</span>
//         </div>
//         <div className="summary-card">
//           <span className="label">Channels</span>
//           <span className="value">{results.dataset_summary.channels}</span>
//         </div>
//         <div className="summary-card">
//           <span className="label">Epochs</span>
//           <span className="value">{results.dataset_summary.epochs}</span>
//         </div>
//       </div>

//       <h3 className="subheading">Classifier Comparison</h3>
//       <table className="results-table">
//         <thead>
//           <tr>
//             <th>Classifier</th>
//             <th>Accuracy</th>
//             <th>Std Dev</th>
//             <th>Inference Time</th>
//           </tr>
//         </thead>
//         <tbody>
//           {results.classifiers.map((c) => (
//             <tr key={c.name} className={c.name === best.name ? "best-row" : ""}>
//               <td>{c.name} {c.name === best.name && "★"}</td>
//               <td>{c.accuracy.toFixed(1)}%</td>
//               <td>± {c.std.toFixed(1)}%</td>
//               <td>{c.time_ms}ms</td>
//             </tr>
//           ))}
//         </tbody>
//       </table>

//       <div className="results-actions">
//         <button className="primary-btn">Download PDF Report</button>
//       </div>
//     </div>
//   );
// }

// export default ResultsSection;






import ParamsExport from "./ParamsExport";

const CATEGORY_LABELS = {
  feature: "Classical",
  epoch: "Riemannian",
  deep: "Deep Learning",
};

function ResultsSection({ results }) {

  if (!results) {
    return <p className="placeholder-text">Run the analysis above to see results.</p>;
  }

  const summary = results.dataset_summary;
  const succeeded = results.classifiers.filter((c) => c.accuracy_mean !== null);
  const failed = results.classifiers.filter((c) => c.accuracy_mean === null);

  // Chance level depends on class count, not a fixed 50% — a 4-class motor
  // imagery task is at chance around 25%.
  const chance = 1 / summary.classes.length;
  const pct = (v) => `${(v * 100).toFixed(1)}%`;

  return (
    <div>
      <div className="summary-grid">
        <div className="summary-card">
          <span className="label">Dataset</span>
          <span className="value" style={{ fontSize: "1rem" }}>{summary.dataset}</span>
        </div>
        <div className="summary-card">
          <span className="label">Epochs</span>
          <span className="value">{summary.n_epochs}</span>
        </div>
        <div className="summary-card">
          <span className="label">Channels</span>
          <span className="value">{summary.n_channels}</span>
        </div>
        <div className="summary-card">
          <span className="label">Sampling rate</span>
          <span className="value">{summary.sampling_rate} Hz</span>
        </div>
      </div>

      <div className="class-summary">
        {summary.classes.map((name) => (
          <span key={name} className="class-chip">
            {name}: {summary.class_counts[name]} trials
          </span>
        ))}
        <span className="chance-note">Chance level ≈ {pct(chance)}</span>
      </div>

      <h3 className="subheading">Classifier Comparison</h3>
      <table className="results-table">
        <thead>
          <tr>
            <th>Classifier</th>
            <th>Type</th>
            <th>Accuracy</th>
            <th>Fold spread</th>
          </tr>
        </thead>
        <tbody>
          {succeeded
            .slice()
            .sort((a, b) => b.accuracy_mean - a.accuracy_mean)
            .map((c) => {
              const isBest = c.classifier === results.best_classifier;
              const lo = Math.min(...c.fold_scores);
              const hi = Math.max(...c.fold_scores);
              return (
                <tr key={c.classifier} className={isBest ? "best-row" : ""}>
                  <td>{c.classifier} {isBest && "★"}</td>
                  <td className="muted">{CATEGORY_LABELS[c.category] ?? c.category}</td>
                  <td>
                    {pct(c.accuracy_mean)} <span className="muted">± {pct(c.accuracy_std)}</span>
                  </td>
                  <td className="muted">{pct(lo)} – {pct(hi)}</td>
                </tr>
              );
            })}
        </tbody>
      </table>

      {/* run_all_classifiers isolates failures so one broken classifier doesn't
          kill the batch — surface them rather than silently dropping rows */}
      {failed.length > 0 && (
        <div className="failed-list">
          <h4>Did not run</h4>
          {failed.map((c) => (
            <p key={c.classifier} className="error-line">
              <strong>{c.classifier}</strong>: {c.error}
            </p>
          ))}
        </div>
      )}

      {succeeded.length > 1 && (
        <p className="interpretation">
          Differences smaller than the ± spread aren't meaningful at{" "}
          {succeeded[0].n_folds} folds over {summary.n_epochs} epochs — roughly{" "}
          {Math.round(summary.n_epochs / succeeded[0].n_folds)} trials per fold, so a
          single trial changing sides moves accuracy by about{" "}
          {pct(1 / (summary.n_epochs / succeeded[0].n_folds))}.
        </p>
      )}

      <ParamsExport results={results} />
    </div>
  );
}

export default ResultsSection;