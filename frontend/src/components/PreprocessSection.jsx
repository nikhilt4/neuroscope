import { useState } from "react";

function PreprocessSection({ mode, onConfigured, config, setConfig }) {
  if (!mode) {
    return <p className="placeholder-text">Select a mode above to configure preprocessing.</p>;
  }

  const update = (key, value) => setConfig((prev) => ({ ...prev, [key]: value }));

  return (
    <div>
      <div className="config-grid">
        <div className="config-group">
          <h4>Bandpass Filter</h4>
          <div className="config-row">
            <label>Low (Hz)</label>
            <input type="number" value={config.bandpass_low} onChange={(e) => update("bandpass_low", e.target.value)} />
          </div>
          <div className="config-row">
            <label>High (Hz)</label>
            <input type="number" value={config.bandpass_high} onChange={(e) => update("bandpass_high", e.target.value)} />
          </div>
        </div>

        <div className="config-group">
          <h4>Notch Filter</h4>
          <div className="config-row">
            <label>Enabled</label>
            <input type="checkbox" checked={config.notch_enabled} onChange={(e) => update("notch_enabled", e.target.checked)} />
          </div>
          <div className="config-row">
            <label>Frequency</label>
            <select value={config.notch_freq} onChange={(e) => update("notch_freq", e.target.value)}>
              <option value="50">50 Hz (EU/UAE/Asia)</option>
              <option value="60">60 Hz (US)</option>
            </select>
          </div>
        </div>

        <div className="config-group">
          <h4>Epoching</h4>
          <div className="config-row">
            <label>tmin (s)</label>
            <input type="number" step="0.1" value={config.epoch_tmin} onChange={(e) => update("epoch_tmin", e.target.value)} />
          </div>
          <div className="config-row">
            <label>tmax (s)</label>
            <input type="number" step="0.1" value={config.epoch_tmax} onChange={(e) => update("epoch_tmax", e.target.value)} />
          </div>
        </div>

        <div className="config-group">
          <h4>Baseline Correction</h4>
          <div className="config-row">
            <label>Enabled</label>
            <input type="checkbox" checked={config.baseline_enabled} onChange={(e) => update("baseline_enabled", e.target.checked)} />
          </div>
        </div>

        <div className="config-group">
          <h4>ICA Artifact Removal</h4>
          <div className="config-row">
            <label>Enabled (slower)</label>
            <input type="checkbox" checked={config.ica_enabled} onChange={(e) => update("ica_enabled", e.target.checked)} />
          </div>
          {config.ica_enabled && (
            <div className="config-row">
              <label>Components</label>
              <input type="number" value={config.ica_components} onChange={(e) => update("ica_components", e.target.value)} />
            </div>
          )}
        </div>

        <div className="config-group">
          <h4>Re-referencing</h4>
          <div className="config-row">
            <label>Method</label>
            <select value={config.reference} onChange={(e) => update("reference", e.target.value)}>
              <option value="average">Common Average</option>
              <option value="Cz">Cz Reference</option>
              <option value="none">None</option>
            </select>
          </div>
        </div>
      </div>

      <button className="primary-btn" onClick={onConfigured}>
        Continue to Classifiers
      </button>
    </div>
  );
}

export default PreprocessSection;