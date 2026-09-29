// import { useState } from "react";
// import { MOCK_RESULTS } from "../mockData";

// const CLASSIFIERS = ["LDA", "SVM", "Random Forest", "EEGNet"];

// function ClassifySection({ enabled, onRun }) {
//   const [selected, setSelected] = useState(["LDA", "SVM", "Random Forest", "EEGNet"]);
//   const [folds, setFolds] = useState(5);
//   const [expanded, setExpanded] = useState(null);

//   const [ldaConfig, setLdaConfig] = useState({ solver: "svd", shrinkage: "auto" });
//   const [svmConfig, setSvmConfig] = useState({ kernel: "rbf", C: 1.0, gamma: "scale" });
//   const [rfConfig, setRfConfig] = useState({ n_estimators: 100, max_depth: "", max_features: "sqrt" });
//   const [eegnetConfig, setEegnetConfig] = useState({ epochs: 300, learning_rate: 0.001, batch_size: 32, dropout_rate: 0.5 });

//   if (!enabled) {
//     return <p className="placeholder-text">Configure preprocessing above to select classifiers.</p>;
//   }

//   const toggle = (clf) => {
//     setSelected((prev) => (prev.includes(clf) ? prev.filter((c) => c !== clf) : [...prev, clf]));
//   };

//   return (
//     <div>
//       <div className="classifier-list">
//         {CLASSIFIERS.map((clf) => (
//           <div key={clf} className="classifier-item">
//             <div className="classifier-header">
//               <label className="checkbox-label">
//                 <input type="checkbox" checked={selected.includes(clf)} onChange={() => toggle(clf)} />
//                 {clf}
//               </label>
//               <button
//                 className="link-btn"
//                 onClick={() => setExpanded(expanded === clf ? null : clf)}
//               >
//                 {expanded === clf ? "Hide options" : "Advanced options"}
//               </button>
//             </div>

//             {expanded === clf && selected.includes(clf) && (
//               <div className="classifier-config">
//                 {clf === "LDA" && (
//                   <>
//                     <div className="config-row">
//                       <label>Solver</label>
//                       <select value={ldaConfig.solver} onChange={(e) => setLdaConfig({ ...ldaConfig, solver: e.target.value })}>
//                         <option value="svd">SVD</option>
//                         <option value="lsqr">LSQR</option>
//                         <option value="eigen">Eigen</option>
//                       </select>
//                     </div>
//                     <div className="config-row">
//                       <label>Shrinkage</label>
//                       <select value={ldaConfig.shrinkage} onChange={(e) => setLdaConfig({ ...ldaConfig, shrinkage: e.target.value })}>
//                         <option value="auto">Auto</option>
//                         <option value="">None</option>
//                       </select>
//                     </div>
//                   </>
//                 )}

//                 {clf === "SVM" && (
//                   <>
//                     <div className="config-row">
//                       <label>Kernel</label>
//                       <select value={svmConfig.kernel} onChange={(e) => setSvmConfig({ ...svmConfig, kernel: e.target.value })}>
//                         <option value="rbf">RBF</option>
//                         <option value="linear">Linear</option>
//                         <option value="poly">Polynomial</option>
//                       </select>
//                     </div>
//                     <div className="config-row">
//                       <label>C</label>
//                       <input type="number" step="0.1" value={svmConfig.C} onChange={(e) => setSvmConfig({ ...svmConfig, C: e.target.value })} />
//                     </div>
//                     <div className="config-row">
//                       <label>Gamma</label>
//                       <select value={svmConfig.gamma} onChange={(e) => setSvmConfig({ ...svmConfig, gamma: e.target.value })}>
//                         <option value="scale">Scale</option>
//                         <option value="auto">Auto</option>
//                       </select>
//                     </div>
//                   </>
//                 )}

//                 {clf === "Random Forest" && (
//                   <>
//                     <div className="config-row">
//                       <label>N Estimators</label>
//                       <input type="number" value={rfConfig.n_estimators} onChange={(e) => setRfConfig({ ...rfConfig, n_estimators: e.target.value })} />
//                     </div>
//                     <div className="config-row">
//                       <label>Max Depth</label>
//                       <input type="number" placeholder="None" value={rfConfig.max_depth} onChange={(e) => setRfConfig({ ...rfConfig, max_depth: e.target.value })} />
//                     </div>
//                     <div className="config-row">
//                       <label>Max Features</label>
//                       <select value={rfConfig.max_features} onChange={(e) => setRfConfig({ ...rfConfig, max_features: e.target.value })}>
//                         <option value="sqrt">sqrt</option>
//                         <option value="log2">log2</option>
//                       </select>
//                     </div>
//                   </>
//                 )}

//                 {clf === "EEGNet" && (
//                   <>
//                     <div className="config-row">
//                       <label>Epochs</label>
//                       <input type="number" value={eegnetConfig.epochs} onChange={(e) => setEegnetConfig({ ...eegnetConfig, epochs: e.target.value })} />
//                     </div>
//                     <div className="config-row">
//                       <label>Learning Rate</label>
//                       <input type="number" step="0.0001" value={eegnetConfig.learning_rate} onChange={(e) => setEegnetConfig({ ...eegnetConfig, learning_rate: e.target.value })} />
//                     </div>
//                     <div className="config-row">
//                       <label>Batch Size</label>
//                       <input type="number" value={eegnetConfig.batch_size} onChange={(e) => setEegnetConfig({ ...eegnetConfig, batch_size: e.target.value })} />
//                     </div>
//                     <div className="config-row">
//                       <label>Dropout Rate</label>
//                       <input type="number" step="0.05" value={eegnetConfig.dropout_rate} onChange={(e) => setEegnetConfig({ ...eegnetConfig, dropout_rate: e.target.value })} />
//                     </div>
//                   </>
//                 )}
//               </div>
//             )}
//           </div>
//         ))}
//       </div>

//       <div className="config-row" style={{ marginTop: 20 }}>
//         <label>Cross-Validation Folds</label>
//         <select value={folds} onChange={(e) => setFolds(e.target.value)}>
//           <option value="3">3</option>
//           <option value="5">5</option>
//           <option value="10">10</option>
//         </select>
//       </div>

//       <button className="primary-btn" onClick={() => onRun(MOCK_RESULTS)} disabled={selected.length === 0}>
//   Run Analysis
// </button>
//     </div>
//   );
// }

// export default ClassifySection;






import { useState } from "react";

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
    hint: "Covariance-based. Replaces CSP rather than following it. Often the strongest option.",
    needsEpochs: true,
    items: [
      { id: "riemann_ts", name: "Tangent Space + LR", note: "Standard MOABB benchmark" },
      { id: "riemann_mdm", name: "MDM", note: "No training step, robust with few trials" },
      { id: "riemann_fgmdm", name: "FgMDM", note: "MDM with geodesic filtering" },
      { id: "xdawn_ts", name: "Xdawn + Tangent Space", note: "For ERP/P300, not motor imagery" },
    ],
  },
  {
    label: "Deep Learning",
    hint: "Trains on raw epochs, learning its own filters. Slowest by a wide margin.",
    needsEpochs: true,
    items: [
      { id: "eegnet", name: "EEGNet (CNN)", note: "Published architecture via braindecode" },
    ],
  },
];

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
    { key: "dropout_rate", label: "Dropout", type: "number", step: 0.05, def: 0.5 },
  ],
};

function ClassifySection({ capabilities, mode, config, setConfig, onRun, isRunning }) {
  const [expanded, setExpanded] = useState(null);

  if (!mode) {
    return <p className="placeholder-text">Select a mode first.</p>;
  }

  // Riemannian and deep models need raw 3D epochs. Basic mode produces summary
  // statistics per epoch, so covariance/CNN methods have nothing to work with.
  const hasEpochs = mode !== "basic";

  const toggle = (id) =>
    setConfig((prev) => ({
      ...prev,
      selected: prev.selected.includes(id)
        ? prev.selected.filter((c) => c !== id)
        : [...prev.selected, id],
    }));

  // Only non-default values go into configs — the backend builders apply their
  // own defaults for anything absent, so an empty object is a valid run.
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
      {CLASSIFIER_GROUPS.map((group) => {
        const blocked = group.needsEpochs && !hasEpochs;
        return (
          <div key={group.label} className="classifier-group">
            <h4>{group.label}</h4>
            <p className="group-hint">{group.hint}</p>
            {blocked && (
              <p className="group-warning">
                Unavailable in {mode} mode — these need epoched signal data.
              </p>
            )}

            <div className="classifier-list">
              {group.items.map((clf) => {
                const isSelected = config.selected.includes(clf.id);
                const fields = PARAM_FIELDS[clf.id] || [];
                return (
                  <div key={clf.id} className={`classifier-item ${blocked ? "disabled" : ""}`}>
                    <div className="classifier-header">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          disabled={blocked}
                          checked={isSelected && !blocked}
                          onChange={() => toggle(clf.id)}
                        />
                        <span>
                          {clf.name}
                          {clf.note && <em className="clf-note"> — {clf.note}</em>}
                        </span>
                      </label>
                      {fields.length > 0 && !blocked && (
                        <button
                          className="link-btn"
                          onClick={() => setExpanded(expanded === clf.id ? null : clf.id)}
                        >
                          {expanded === clf.id ? "Hide" : "Options"}
                        </button>
                      )}
                    </div>

                    {expanded === clf.id && isSelected && (
                      <div className="classifier-config">
                        {fields.map((f) => (
                          <div key={f.key} className="config-row">
                            <label title={f.hint}>{f.label}</label>
                            {f.type === "select" ? (
                              <select
                                value={paramValue(clf.id, f.key, f.def)}
                                onChange={(e) => setParam(clf.id, f.key, e.target.value, f.def)}
                              >
                                {f.options.map((o) => (
                                  <option key={o} value={o}>{o}</option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="number"
                                step={f.step}
                                value={paramValue(clf.id, f.key, f.def)}
                                onChange={(e) =>
                                  setParam(clf.id, f.key,
                                    e.target.value === "" ? "" : Number(e.target.value), f.def)
                                }
                              />
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
        );
      })}

      <div className="config-group" style={{ marginTop: 20 }}>
        <h4>Cross-Validation</h4>
        <div className="config-row">
          <label>Folds</label>
          <select
            value={config.n_folds}
            onChange={(e) => setConfig((p) => ({ ...p, n_folds: Number(e.target.value) }))}
          >
            {[3, 5, 10].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        {mode === "motor_imagery" && (
          <div className="config-row">
            <label title="Spatial filters learned per class pair">CSP components</label>
            <input
              type="number"
              min="1"
              max="32"
              value={config.csp_components}
              onChange={(e) => setConfig((p) => ({ ...p, csp_components: Number(e.target.value) }))}
            />
          </div>
        )}
      </div>

      <button
        className="primary-btn"
        onClick={onRun}
        disabled={config.selected.length === 0 || isRunning}
      >
        {isRunning ? "Running analysis..." : "Run Analysis"}
      </button>
      {config.selected.includes("eegnet") && (
        <p className="hint-text">EEGNet trains a neural network per fold — this may take a few minutes.</p>
      )}
    </div>
  );
}

export default ClassifySection;