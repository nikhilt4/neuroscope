// import { useState } from "react";

// function PreprocessSection({ mode, onConfigured, config, setConfig }) {
//   if (!mode) {
//     return <p className="placeholder-text">Select a mode above to configure preprocessing.</p>;
//   }

//   const update = (key, value) => setConfig((prev) => ({ ...prev, [key]: value }));

//   return (
//     <div>
//       <div className="config-grid">
//         <div className="config-group">
//           <h4>Bandpass Filter</h4>
//           <div className="config-row">
//             <label>Low (Hz)</label>
//             <input type="number" value={config.bandpass_low} onChange={(e) => update("bandpass_low", e.target.value)} />
//           </div>
//           <div className="config-row">
//             <label>High (Hz)</label>
//             <input type="number" value={config.bandpass_high} onChange={(e) => update("bandpass_high", e.target.value)} />
//           </div>
//         </div>

//         <div className="config-group">
//           <h4>Notch Filter</h4>
//           <div className="config-row">
//             <label>Enabled</label>
//             <input type="checkbox" checked={config.notch_enabled} onChange={(e) => update("notch_enabled", e.target.checked)} />
//           </div>
//           <div className="config-row">
//             <label>Frequency</label>
//             <select value={config.notch_freq} onChange={(e) => update("notch_freq", e.target.value)}>
//               <option value="50">50 Hz (EU/UAE/Asia)</option>
//               <option value="60">60 Hz (US)</option>
//             </select>
//           </div>
//         </div>

//         <div className="config-group">
//           <h4>Epoching</h4>
//           <div className="config-row">
//             <label>tmin (s)</label>
//             <input type="number" step="0.1" value={config.epoch_tmin} onChange={(e) => update("epoch_tmin", e.target.value)} />
//           </div>
//           <div className="config-row">
//             <label>tmax (s)</label>
//             <input type="number" step="0.1" value={config.epoch_tmax} onChange={(e) => update("epoch_tmax", e.target.value)} />
//           </div>
//         </div>

//         <div className="config-group">
//           <h4>Baseline Correction</h4>
//           <div className="config-row">
//             <label>Enabled</label>
//             <input type="checkbox" checked={config.baseline_enabled} onChange={(e) => update("baseline_enabled", e.target.checked)} />
//           </div>
//         </div>

//         <div className="config-group">
//           <h4>ICA Artifact Removal</h4>
//           <div className="config-row">
//             <label>Enabled (slower)</label>
//             <input type="checkbox" checked={config.ica_enabled} onChange={(e) => update("ica_enabled", e.target.checked)} />
//           </div>
//           {config.ica_enabled && (
//             <div className="config-row">
//               <label>Components</label>
//               <input type="number" value={config.ica_components} onChange={(e) => update("ica_components", e.target.value)} />
//             </div>
//           )}
//         </div>

//         <div className="config-group">
//           <h4>Re-referencing</h4>
//           <div className="config-row">
//             <label>Method</label>
//             <select value={config.reference} onChange={(e) => update("reference", e.target.value)}>
//               <option value="average">Common Average</option>
//               <option value="Cz">Cz Reference</option>
//               <option value="none">None</option>
//             </select>
//           </div>
//         </div>
//       </div>

//       <button className="primary-btn" onClick={onConfigured}>
//         Continue to Classifiers
//       </button>
//     </div>
//   );
// }

// export default PreprocessSection;





function PreprocessSection({ mode, capabilities, preset, overrides, setOverrides, onConfigured }) {
  if (!mode) {
    return <p className="placeholder-text">Select a mode first.</p>;
  }

  if (capabilities?.is_epochs) {
    return (
      <div>
        <div className="preset-banner">
          This dataset arrives already filtered and epoched by its source, so
          preprocessing options don't apply.
        </div>
        <button className="primary-btn" onClick={onConfigured}>
          Continue to Classifiers
        </button>
      </div>
    );
  }


  const valueOf = (key) => overrides[key] ?? preset[key] ?? "";
  const isOverridden = (key) => key in overrides;

  const update = (key, value) => {
    setOverrides((prev) => {
      const next = { ...prev };
      if (value === "" || value === null) {
        delete next[key];          // cleared → fall back to the preset
      } else {
        next[key] = value;
      }
      return next;
    });
  };

  const reset = () => setOverrides({});

  return (
    <div>
    
      <div className="preset-banner">
        Defaults come from this dataset's recommended preprocessing.
        {Object.keys(overrides).length > 0 && (
          <button className="link-btn" onClick={reset}>
            reset {Object.keys(overrides).length} change(s)
          </button>
        )}
      </div>

      <div className="config-grid">
        <div className="config-group">
          <h4>Bandpass Filter</h4>
          <div className="config-row">
            <label title="Removes slow drift below this frequency">Low (Hz)</label>
            <input type="number" step="0.1"
              className={isOverridden("bandpass_low") ? "overridden" : ""}
              value={valueOf("bandpass_low")}
              onChange={(e) => update("bandpass_low", e.target.value === "" ? "" : Number(e.target.value))} />
          </div>
          <div className="config-row">
            <label title="Removes high-frequency noise and muscle artifact">High (Hz)</label>
            <input type="number" step="0.1" placeholder="none"
              className={isOverridden("bandpass_high") ? "overridden" : ""}
              value={valueOf("bandpass_high")}
              onChange={(e) => update("bandpass_high", e.target.value === "" ? "" : Number(e.target.value))} />
          </div>
        </div>

        <div className="config-group">
          <h4>Notch Filter</h4>
          <div className="config-row">
            <label>Enabled</label>
            <input type="checkbox"
              checked={Boolean(valueOf("notch_enabled"))}
              onChange={(e) => update("notch_enabled", e.target.checked)} />
          </div>
          {valueOf("notch_enabled") && (
            <div className="config-row">
              <label title="Mains hum — 50 Hz in most of the world, 60 Hz in the Americas">Frequency</label>
              <select value={valueOf("notch_freq") || 50}
                onChange={(e) => update("notch_freq", Number(e.target.value))}>
                <option value="50">50 Hz (EU / UAE / Asia)</option>
                <option value="60">60 Hz (US)</option>
              </select>
            </div>
          )}
        </div>

        <div className="config-group">
          <h4>Epoching</h4>
          <div className="config-row">
            <label title="Window start, relative to each event">tmin (s)</label>
            <input type="number" step="0.1"
              className={isOverridden("epoch_tmin") ? "overridden" : ""}
              value={valueOf("epoch_tmin")}
              onChange={(e) => update("epoch_tmin", e.target.value === "" ? "" : Number(e.target.value))} />
          </div>
          <div className="config-row">
            <label title="Window end, relative to each event">tmax (s)</label>
            <input type="number" step="0.1"
              className={isOverridden("epoch_tmax") ? "overridden" : ""}
              value={valueOf("epoch_tmax")}
              onChange={(e) => update("epoch_tmax", e.target.value === "" ? "" : Number(e.target.value))} />
          </div>
        </div>

        <div className="config-group">
          <h4>Baseline Correction</h4>
          <div className="config-row">
            <label title="Subtracts the pre-event mean from each epoch">Enabled</label>
            <input type="checkbox"
              checked={Boolean(valueOf("baseline_enabled"))}
              onChange={(e) => update("baseline_enabled", e.target.checked)} />
          </div>
        </div>

        <div className="config-group">
          <h4>ICA Artifact Removal</h4>
          <div className="config-row">
            <label title="Removes eye-blink components — slow, needs an EOG channel">Enabled</label>
            <input type="checkbox"
              checked={Boolean(valueOf("ica_enabled"))}
              onChange={(e) => update("ica_enabled", e.target.checked)} />
          </div>
          {valueOf("ica_enabled") && (
            <div className="config-row">
              <label>Components</label>
              <input type="number" value={valueOf("ica_components") || 20}
                onChange={(e) => update("ica_components", Number(e.target.value))} />
            </div>
          )}
        </div>

        <div className="config-group">
          <h4>Re-referencing</h4>
          <div className="config-row">
            <label>Method</label>
            <select value={Array.isArray(valueOf("reference")) ? "custom" : (valueOf("reference") || "average")}
              onChange={(e) => update("reference", e.target.value)}>
              <option value="average">Common Average</option>
              <option value="Cz">Cz</option>
              <option value="none">None</option>
              {Array.isArray(valueOf("reference")) && (
                <option value="custom">{valueOf("reference").join(" + ")} (preset)</option>
              )}
            </select>
          </div>
        </div>
      </div>

      {/* Epoching group is meaningless for pre-epoched sources (MOABB, limo,
          kiloword, FIF) — hide rather than show controls that do nothing */}

      <button className="primary-btn" onClick={onConfigured}>
        Continue to Classifiers
      </button>
    </div>
  );
}

export default PreprocessSection;