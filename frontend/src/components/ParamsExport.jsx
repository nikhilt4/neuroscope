import { useState } from "react";

function ParamsExport({ results }) {
  const [copied, setCopied] = useState(false);
  if (!results) return null;

  const toPy = (v) => {
    if (v === null || v === undefined) return "None";
    if (typeof v === "boolean") return v ? "True" : "False";
    if (Array.isArray(v)) return `(${v.map(toPy).join(", ")})`;
    if (typeof v === "object") {
      const inner = Object.entries(v).map(([k, x]) => `"${k}": ${toPy(x)}`);
      return `{${inner.join(", ")}}`;
    }
    return typeof v === "string" ? `"${v}"` : String(v);
  };

  const config = results.preprocessing_applied;
  const best = results.classifiers.find((c) => c.classifier === results.best_classifier);

  const code = `# NeuroScope — ${results.dataset_summary.dataset}
# ${results.dataset_summary.n_epochs} epochs, ${results.dataset_summary.n_channels} channels @ ${results.dataset_summary.sampling_rate} Hz
# Classes: ${results.dataset_summary.classes.join(", ")}
# Best: ${best?.classifier} at ${(best?.accuracy_mean * 100).toFixed(1)}%

preprocessing = {
${Object.entries(config).map(([k, v]) => `    "${k}": ${toPy(v)},`).join("\n")}
}

results = {
${results.classifiers
  .filter((c) => c.accuracy_mean !== null)
  .map((c) => `    "${c.classifier}": ${c.accuracy_mean.toFixed(4)},`)
  .join("\n")}
}
`;

  const copy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const download = () => {
    const url = URL.createObjectURL(new Blob([code], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "neuroscope_params.py";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ marginTop: 24 }}>
      <h3 className="subheading">Reproduce This Run</h3>
      <pre className="config-dump">{code}</pre>
      <div className="results-actions">
        <button className="secondary-btn" onClick={copy}>
          {copied ? "Copied ✓" : "Copy"}
        </button>
        <button className="secondary-btn" onClick={download}>Download .py</button>
      </div>
    </div>
  );
}

export default ParamsExport;