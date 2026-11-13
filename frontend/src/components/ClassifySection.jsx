import { useEffect, useState } from "react";
import { Icon } from "./ui/Icons";

const CLASSIFIER_GROUPS = [
  {
    label: "Classical",
    hint: "Trained on extracted features. In motor imagery mode, CSP is fit inside each CV fold.",
    items: [
      { id: "lda", name: "LDA", note: "Fast baseline, strong with CSP" },
      { id: "svm", name: "SVM", note: "Good on small datasets" },
      { id: "rf", name: "Random Forest", note: "Robust to noise" },
      { id: "logreg", name: "Logistic Regression", note: "Interpretable, calibrated" },
      { id: "extra_trees", name: "Extra Trees" },
      { id: "hgb", name: "Gradient Boosting" },
      { id: "mlp", name: "MLP", note: "Shallow net on features" },
      { id: "knn", name: "k-Nearest Neighbours" },
      { id: "gnb", name: "Naive Bayes", note: "Weak on EEG — useful as a floor" },
    ],
  },
  {
    label: "Riemannian",
    hint: "Covariance-based, reading the raw trials in every mode. Replaces CSP rather than following it.",
    items: [
      { id: "riemann_ts", name: "Tangent Space + LR", note: "Standard MOABB benchmark" },
      { id: "riemann_mdm", name: "MDM", note: "No training step, robust with few trials" },
      { id: "riemann_fgmdm", name: "FgMDM", note: "MDM with geodesic filtering" },
      { id: "xdawn_ts", name: "Xdawn + Tangent Space", note: "For ERP/P300, not motor imagery" },
    ],
  },
  {
    label: "Deep Learning",
    hint: "Trains on raw trials in every mode, learning its own filters. Slowest by a wide margin.",
    items: [
      { id: "eegnet", name: "EEGNet (CNN)", note: "Published architecture via braindecode" },
    ],
  },
];

const CLASSICAL_IDS = CLASSIFIER_GROUPS[0].items.map((c) => c.id);
const RIEMANNIAN_IDS = new Set(["riemann_ts", "riemann_mdm", "riemann_fgmdm", "xdawn_ts"]);

// Starting sets, from this project's measured runs (NEUROSCOPE_CONTEXT.md section 4):
//   motor imagery — BNCI2014_001 s1: riemann_ts 0.833, riemann_mdm 0.741, svm 0.741, lda 0.734
//   spectral      — Sleep-EDF s1/s4: svm, logreg, rf, hgb, knn, lda all 0.77–0.87, within fold spread
//   basic         — no benchmark yet; robust general-purpose models
const RECOMMENDED = {
  motor_imagery: ["riemann_ts", "riemann_mdm", "riemann_fgmdm", "lda", "svm"],
  spectral: ["svm", "logreg", "rf", "hgb", "knn", "lda"],
  basic: ["lda", "svm", "rf", "extra_trees"],
};

/**
 * Does this decoder suit the loaded data? Returns null, { level: "warn", text }, or
 * { level: "block", text }. Only "block" disables a decoder; warnings inform.
 * Each rule comes from a run that went wrong (Sleep-EDF subject 4, 13 Nov):
 *   2 EEG channels -> riemann_ts 0.63, riemann_mdm 0.50 (chance 0.50) vs classical ~0.79
 *   FgMDM: tangent space has c(c+1)/2 features but its LDA needs n_classes-1 (pyriemann 0.12)
 *   xdawn_ts on sleep 0.66; EEGNet on 30-s windows 0.68
 */
function suitability(id, { nEeg, nClasses, epochSeconds, task }) {
  if (id === "riemann_fgmdm" && nEeg) {
    const nTs = (nEeg * (nEeg + 1)) / 2;
    if (nTs < nClasses - 1) {
      return { level: "block", text: `Can't run: ${nEeg} EEG channels give ${nTs} covariance features, and ${nClasses} classes need at least ${nClasses - 1}.` };
    }
  }
  if (RIEMANNIAN_IDS.has(id) && nEeg && nEeg <= 3) {
    return { level: "warn", text: `Only ${nEeg} EEG channel${nEeg > 1 ? "s" : ""}, so covariance-based decoders have very little to work with.` };
  }
  if (id === "xdawn_ts" && task && task !== "erp") {
    return { level: "warn", text: "Built for time-locked ERP responses (e.g. P300). Usually weak on this kind of data." };
  }
  if (id === "eegnet" && epochSeconds > 8) {
    return { level: "warn", text: `Built for short trials of a few seconds. These are ${Math.round(epochSeconds)} s long, so expect slow training and weak results.` };
  }
  return null;
}

// Frontend `def` values MUST match the backend builder's config.get() fallbacks —
// the UI only sends values that differ from `def` (CLAUDE.md section 7).
const PARAM_FIELDS = {
  lda: [
    { key: "solver", label: "Solver", type: "select", options: ["svd", "lsqr", "eigen"], def: "svd" },
    { key: "shrinkage", label: "Shrinkage", type: "select", options: ["auto", "none"], def: "auto",
      hint: "Only applies with lsqr/eigen solvers" },
  ],
  svm: [
    { key: "kernel", label: "Kernel", type: "select", options: ["rbf", "linear", "poly"], def: "rbf" },
    { key: "C", label: "C", type: "number", step: 0.1, def: 1.0 },
    { key: "gamma", label: "Gamma", type: "select", options: ["scale", "auto"], def: "scale" },
  ],
  rf: [
    { key: "n_estimators", label: "Trees", type: "number", def: 100 },
    { key: "max_depth", label: "Max depth", type: "number", def: "" },
    { key: "max_features", label: "Max features", type: "select", options: ["sqrt", "log2"], def: "sqrt" },
  ],
  logreg: [
    { key: "C", label: "C", type: "number", step: 0.1, def: 1.0 },
    { key: "max_iter", label: "Max iterations", type: "number", def: 1000 },
  ],
  extra_trees: [{ key: "n_estimators", label: "Trees", type: "number", def: 100 }],
  hgb: [
    { key: "max_iter", label: "Iterations", type: "number", def: 100 },
    { key: "learning_rate", label: "Learning rate", type: "number", step: 0.01, def: 0.1 },
  ],
  mlp: [
    { key: "alpha", label: "L2 penalty", type: "number", step: 0.0001, def: 0.0001 },
    { key: "max_iter", label: "Max iterations", type: "number", def: 500 },
  ],
  knn: [{ key: "n_neighbors", label: "Neighbours", type: "number", def: 5 }],
  gnb: [],
  riemann_ts: [
    { key: "cov_estimator", label: "Covariance", type: "select", options: ["oas", "lwf", "scm"], def: "oas",
      hint: "Shrinkage estimators avoid singular matrices on short epochs" },
    { key: "C", label: "C", type: "number", step: 0.1, def: 1.0 },
  ],
  riemann_mdm: [
    { key: "cov_estimator", label: "Covariance", type: "select", options: ["oas", "lwf", "scm"], def: "oas" },
  ],
  riemann_fgmdm: [
    { key: "cov_estimator", label: "Covariance", type: "select", options: ["oas", "lwf", "scm"], def: "oas" },
  ],
  xdawn_ts: [
    { key: "nfilter", label: "Xdawn filters", type: "number", def: 4 },
    { key: "cov_estimator", label: "Covariance", type: "select", options: ["oas", "lwf", "scm"], def: "oas" },
  ],
  eegnet: [
    { key: "epochs", label: "Training epochs", type: "number", def: 50,
      hint: "Capped server-side to keep the demo responsive" },
    { key: "learning_rate", label: "Learning rate", type: "number", step: 0.0001, def: 0.01 },
    { key: "batch_size", label: "Batch size", type: "number", def: 32 },
    { key: "dropout_rate", label: "Dropout", type: "number", step: 0.05, def: 0.25 },
  ],
};

/**
 * Step 4. The analysis itself is started from the wizard footer's "Run analysis".
 */
function ClassifySection({ capabilities, mode, config, setConfig, epochSeconds }) {
  const [expanded, setExpanded] = useState(null);

  const ctx = {
    nEeg: capabilities?.n_eeg_channels ?? capabilities?.n_channels,
    nClasses: (capabilities?.analysis_classes ?? capabilities?.event_classes ?? []).length,
    epochSeconds,
    task: capabilities?.task,
  };
  const checks = Object.fromEntries(
    CLASSIFIER_GROUPS.flatMap((g) => g.items).map((c) => [c.id, suitability(c.id, ctx)])
  );
  const blocked = Object.keys(checks).filter((id) => checks[id]?.level === "block");
  const blockedKey = blocked.join();

  // A decoder that can't run on this data shouldn't stay selected from an earlier choice.
  // Hooks run before any early return (rules of hooks).
  useEffect(() => {
    if (!blockedKey) return;
    setConfig((prev) => {
      const kept = prev.selected.filter((id) => !blockedKey.split(",").includes(id));
      return kept.length === prev.selected.length ? prev : { ...prev, selected: kept };
    });
  }, [blockedKey, setConfig]);

  if (!mode) {
    return <p className="placeholder-text">Select a mode first.</p>;
  }

  const select = (ids) => setConfig((prev) => ({ ...prev, selected: ids.filter((id) => !checks[id]) }));
  const recommended = (RECOMMENDED[mode] || RECOMMENDED.basic).filter((id) => !checks[id]);

  const toggle = (id) =>
    setConfig((prev) => ({
      ...prev,
      selected: prev.selected.includes(id)
        ? prev.selected.filter((c) => c !== id)
        : [...prev.selected, id],
    }));

  const setParam = (clfId, key, value, def) => {
    setConfig((prev) => {
      const next = { ...prev.configs };
      const own = { ...(next[clfId] || {}) };
      if (value === "" || value === def) delete own[key];
      else own[key] = value;
      if (Object.keys(own).length) next[clfId] = own;
      else delete next[clfId];
      return { ...prev, configs: next };
    });
  };

  const paramValue = (clfId, key, def) => config.configs[clfId]?.[key] ?? def;

  return (
    <div>
      <div className="cl-quick">
        <span className="cl-quick-label">Quick pick</span>
        <button className="cl-chip" onClick={() => select(recommended)} title={recommended.join(", ")}>
          <Icon name="sparkle" />Recommended for this data
        </button>
        <button className="cl-chip" onClick={() => select(CLASSICAL_IDS)}>All classical</button>
        <button className="cl-chip" onClick={() => select([])}>Clear</button>
        <span className="cl-count">{config.selected.length} selected</span>
      </div>

      {CLASSIFIER_GROUPS.map((group) => (
        <div key={group.label} className="classifier-group">
          <h4>{group.label}</h4>
          <p className="group-hint">{group.hint}</p>
          <div className="classifier-list">
            {group.items.map((clf) => {
              const check = checks[clf.id];
              const isBlocked = check?.level === "block";
              const isSelected = config.selected.includes(clf.id) && !isBlocked;
              const fields = PARAM_FIELDS[clf.id] || [];
              return (
                <div key={clf.id} className={`classifier-item ${isBlocked ? "disabled" : ""}`}>
                  <div className="classifier-header">
                    <label className="checkbox-label">
                      <input type="checkbox" disabled={isBlocked} checked={isSelected}
                             onChange={() => toggle(clf.id)} />
                      <span>
                        {clf.name}
                        {clf.note && <em className="clf-note"> — {clf.note}</em>}
                      </span>
                    </label>
                    {fields.length > 0 && !isBlocked && (
                      <button className="link-btn" onClick={() => setExpanded(expanded === clf.id ? null : clf.id)}>
                        {expanded === clf.id ? "Hide" : "Options"}
                      </button>
                    )}
                  </div>

                  {check && (
                    <p className={`cl-fit cl-fit-${check.level}`}>
                      <Icon name={isBlocked ? "x" : "alert"} />{check.text}
                    </p>
                  )}

                  {expanded === clf.id && isSelected && (
                    <div className="classifier-config">
                      {fields.map((f) => (
                        <div key={f.key} className="config-row">
                          <label title={f.hint}>{f.label}</label>
                          {f.type === "select" ? (
                            <select value={paramValue(clf.id, f.key, f.def)}
                                    onChange={(e) => setParam(clf.id, f.key, e.target.value, f.def)}>
                              {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
                            </select>
                          ) : (
                            <input type="number" step={f.step} value={paramValue(clf.id, f.key, f.def)}
                                   onChange={(e) => setParam(clf.id, f.key,
                                     e.target.value === "" ? "" : Number(e.target.value), f.def)} />
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <div className="config-group" style={{ marginTop: 20 }}>
        <h4>Cross-Validation</h4>
        <div className="config-row">
          <label>Folds</label>
          <select value={config.n_folds}
                  onChange={(e) => setConfig((p) => ({ ...p, n_folds: Number(e.target.value) }))}>
            {[3, 5, 10].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        {mode === "motor_imagery" && (
          <div className="config-row">
            <label title="Spatial filters learned per class pair">CSP components</label>
            <input type="number" min="1" max="32" value={config.csp_components}
                   onChange={(e) => setConfig((p) => ({ ...p, csp_components: Number(e.target.value) }))} />
          </div>
        )}
      </div>

      {config.selected.includes("eegnet") && (
        <p className="hint-text">EEGNet trains a neural network per fold, so this may take a few minutes.</p>
      )}
    </div>
  );
}

export default ClassifySection;
