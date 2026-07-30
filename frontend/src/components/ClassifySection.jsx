import { useState } from "react";
import { MOCK_RESULTS } from "../mockData";

const CLASSIFIERS = ["LDA", "SVM", "Random Forest", "EEGNet"];

function ClassifySection({ enabled, onRun }) {
  const [selected, setSelected] = useState(["LDA", "SVM", "Random Forest", "EEGNet"]);
  const [folds, setFolds] = useState(5);
  const [expanded, setExpanded] = useState(null);

  const [ldaConfig, setLdaConfig] = useState({ solver: "svd", shrinkage: "auto" });
  const [svmConfig, setSvmConfig] = useState({ kernel: "rbf", C: 1.0, gamma: "scale" });
  const [rfConfig, setRfConfig] = useState({ n_estimators: 100, max_depth: "", max_features: "sqrt" });
  const [eegnetConfig, setEegnetConfig] = useState({ epochs: 300, learning_rate: 0.001, batch_size: 32, dropout_rate: 0.5 });

  if (!enabled) {
    return <p className="placeholder-text">Configure preprocessing above to select classifiers.</p>;
  }

  const toggle = (clf) => {
    setSelected((prev) => (prev.includes(clf) ? prev.filter((c) => c !== clf) : [...prev, clf]));
  };

  return (
    <div>
      <div className="classifier-list">
        {CLASSIFIERS.map((clf) => (
          <div key={clf} className="classifier-item">
            <div className="classifier-header">
              <label className="checkbox-label">
                <input type="checkbox" checked={selected.includes(clf)} onChange={() => toggle(clf)} />
                {clf}
              </label>
              <button
                className="link-btn"
                onClick={() => setExpanded(expanded === clf ? null : clf)}
              >
                {expanded === clf ? "Hide options" : "Advanced options"}
              </button>
            </div>

            {expanded === clf && selected.includes(clf) && (
              <div className="classifier-config">
                {clf === "LDA" && (
                  <>
                    <div className="config-row">
                      <label>Solver</label>
                      <select value={ldaConfig.solver} onChange={(e) => setLdaConfig({ ...ldaConfig, solver: e.target.value })}>
                        <option value="svd">SVD</option>
                        <option value="lsqr">LSQR</option>
                        <option value="eigen">Eigen</option>
                      </select>
                    </div>
                    <div className="config-row">
                      <label>Shrinkage</label>
                      <select value={ldaConfig.shrinkage} onChange={(e) => setLdaConfig({ ...ldaConfig, shrinkage: e.target.value })}>
                        <option value="auto">Auto</option>
                        <option value="">None</option>
                      </select>
                    </div>
                  </>
                )}

                {clf === "SVM" && (
                  <>
                    <div className="config-row">
                      <label>Kernel</label>
                      <select value={svmConfig.kernel} onChange={(e) => setSvmConfig({ ...svmConfig, kernel: e.target.value })}>
                        <option value="rbf">RBF</option>
                        <option value="linear">Linear</option>
                        <option value="poly">Polynomial</option>
                      </select>
                    </div>
                    <div className="config-row">
                      <label>C</label>
                      <input type="number" step="0.1" value={svmConfig.C} onChange={(e) => setSvmConfig({ ...svmConfig, C: e.target.value })} />
                    </div>
                    <div className="config-row">
                      <label>Gamma</label>
                      <select value={svmConfig.gamma} onChange={(e) => setSvmConfig({ ...svmConfig, gamma: e.target.value })}>
                        <option value="scale">Scale</option>
                        <option value="auto">Auto</option>
                      </select>
                    </div>
                  </>
                )}

                {clf === "Random Forest" && (
                  <>
                    <div className="config-row">
                      <label>N Estimators</label>
                      <input type="number" value={rfConfig.n_estimators} onChange={(e) => setRfConfig({ ...rfConfig, n_estimators: e.target.value })} />
                    </div>
                    <div className="config-row">
                      <label>Max Depth</label>
                      <input type="number" placeholder="None" value={rfConfig.max_depth} onChange={(e) => setRfConfig({ ...rfConfig, max_depth: e.target.value })} />
                    </div>
                    <div className="config-row">
                      <label>Max Features</label>
                      <select value={rfConfig.max_features} onChange={(e) => setRfConfig({ ...rfConfig, max_features: e.target.value })}>
                        <option value="sqrt">sqrt</option>
                        <option value="log2">log2</option>
                      </select>
                    </div>
                  </>
                )}

                {clf === "EEGNet" && (
                  <>
                    <div className="config-row">
                      <label>Epochs</label>
                      <input type="number" value={eegnetConfig.epochs} onChange={(e) => setEegnetConfig({ ...eegnetConfig, epochs: e.target.value })} />
                    </div>
                    <div className="config-row">
                      <label>Learning Rate</label>
                      <input type="number" step="0.0001" value={eegnetConfig.learning_rate} onChange={(e) => setEegnetConfig({ ...eegnetConfig, learning_rate: e.target.value })} />
                    </div>
                    <div className="config-row">
                      <label>Batch Size</label>
                      <input type="number" value={eegnetConfig.batch_size} onChange={(e) => setEegnetConfig({ ...eegnetConfig, batch_size: e.target.value })} />
                    </div>
                    <div className="config-row">
                      <label>Dropout Rate</label>
                      <input type="number" step="0.05" value={eegnetConfig.dropout_rate} onChange={(e) => setEegnetConfig({ ...eegnetConfig, dropout_rate: e.target.value })} />
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="config-row" style={{ marginTop: 20 }}>
        <label>Cross-Validation Folds</label>
        <select value={folds} onChange={(e) => setFolds(e.target.value)}>
          <option value="3">3</option>
          <option value="5">5</option>
          <option value="10">10</option>
        </select>
      </div>

      <button className="primary-btn" onClick={() => onRun(MOCK_RESULTS)} disabled={selected.length === 0}>
  Run Analysis
</button>
    </div>
  );
}

export default ClassifySection;