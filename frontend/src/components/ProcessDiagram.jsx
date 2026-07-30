const STEPS = [
  { label: "Upload", desc: "EDF, BDF, CSV, MAT, BIDS..." },
  { label: "Preprocess", desc: "Filter, ICA, epoch" },
  { label: "Extract Features", desc: "CSP, PSD, bands" },
  { label: "Classify", desc: "LDA, SVM, RF, EEGNet" },
  { label: "Results", desc: "Compare & export" },
];

function ProcessDiagram() {
  return (
    <div className="diagram">
      {STEPS.map((step, i) => (
        <div key={step.label} className="diagram-item">
          <div className="diagram-box">
            <span className="diagram-label">{step.label}</span>
            <span className="diagram-desc">{step.desc}</span>
          </div>
          {i < STEPS.length - 1 && <span className="diagram-arrow">→</span>}
        </div>
      ))}
    </div>
  );
}

export default ProcessDiagram;