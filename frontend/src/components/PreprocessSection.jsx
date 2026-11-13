/**
 * Step 3. Every field shows the dataset preset's value until the user changes it;
 * only changed values are sent (see the override pattern in CLAUDE.md section 7).
 * Moving on is done with the wizard footer's Continue button.
 */
function PreprocessSection({ mode, capabilities, preset, overrides, setOverrides }) {
  if (!mode) {
    return <p className="placeholder-text">Select a mode first.</p>;
  }

  if (capabilities?.is_epochs) {
    return (
      <div className="preset-banner">
        This dataset arrives already filtered and split into trials by its source, so
        preprocessing options don&apos;t apply. Continue to choose your decoders.
      </div>
    );
  }

  const valueOf = (key) => (key in overrides ? overrides[key] : preset[key]) ?? "";
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
  const num = (e) => (e.target.value === "" ? "" : Number(e.target.value));

  // ── analysis window (epoch_train_crop) ────────────────────────────────────
  // Some presets epoch wide and then analyse a narrower window: eegbci epochs
  // -1…4 s but classifies 1…2 s, to skip the visual response to the cue. That
  // crop used to be invisible, so editing tmin/tmax looked like it changed what
  // was analysed when it didn't.
  const tmin = Number(valueOf("epoch_tmin"));
  const tmax = Number(valueOf("epoch_tmax"));
  const crop = valueOf("epoch_train_crop");          // [start, end] | null | ""
  const hasCrop = Array.isArray(crop) && crop.length === 2;
  const setCropPart = (i, v) => {
    if (v === "") return;
    const pair = hasCrop ? [...crop] : [tmin, tmax];
    pair[i] = v;
    setOverrides((prev) => ({ ...prev, epoch_train_crop: pair }));
  };
  const useWholeEpoch = () =>
    // explicit null (not deletion) so it overrides a preset crop
    setOverrides((prev) => ({ ...prev, epoch_train_crop: null }));
  const cropOutside = hasCrop && (crop[0] < tmin || crop[1] > tmax || crop[0] >= crop[1]);
  const windowLen = hasCrop ? crop[1] - crop[0] : tmax - tmin;

  return (
    <div>
      <div className="preset-banner">
        <span>Defaults come from this dataset&apos;s recommended preprocessing. Changed values are highlighted.</span>
        {Object.keys(overrides).length > 0 && (
          <button className="link-btn" onClick={reset}>
            Reset {Object.keys(overrides).length} change{Object.keys(overrides).length > 1 ? "s" : ""}
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
              onChange={(e) => update("bandpass_low", num(e))} />
          </div>
          <div className="config-row">
            <label title="Removes high-frequency noise and muscle artifact">High (Hz)</label>
            <input type="number" step="0.1" placeholder="none"
              className={isOverridden("bandpass_high") ? "overridden" : ""}
              value={valueOf("bandpass_high")}
              onChange={(e) => update("bandpass_high", num(e))} />
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
              onChange={(e) => update("epoch_tmin", num(e))} />
          </div>
          <div className="config-row">
            <label title="Window end, relative to each event">tmax (s)</label>
            <input type="number" step="0.1"
              className={isOverridden("epoch_tmax") ? "overridden" : ""}
              value={valueOf("epoch_tmax")}
              onChange={(e) => update("epoch_tmax", num(e))} />
          </div>
          {valueOf("event_chunk_duration") !== "" && (
            <p className="config-hint">
              Long annotations are split into {valueOf("event_chunk_duration")}-s windows, one trial each.
            </p>
          )}
        </div>

        <div className="config-group">
          <h4>Analysis Window</h4>
          {hasCrop ? (
            <>
              <div className="config-row">
                <label title="Start of the part of each epoch the decoders see">Start (s)</label>
                <input type="number" step="0.1"
                  className={isOverridden("epoch_train_crop") ? "overridden" : ""}
                  value={crop[0]}
                  onChange={(e) => setCropPart(0, num(e))} />
              </div>
              <div className="config-row">
                <label title="End of the part of each epoch the decoders see">End (s)</label>
                <input type="number" step="0.1"
                  className={isOverridden("epoch_train_crop") ? "overridden" : ""}
                  value={crop[1]}
                  onChange={(e) => setCropPart(1, num(e))} />
              </div>
              <p className="config-hint">
                Decoders only see {crop[0]}–{crop[1]} s of each {tmin}–{tmax} s epoch.{" "}
                <button className="link-btn" onClick={useWholeEpoch}>Use the whole epoch</button>
              </p>
              {cropOutside && (
                <p className="config-warn">The window must sit inside the epoch ({tmin}–{tmax} s) and start before it ends.</p>
              )}
            </>
          ) : (
            <>
              <p className="config-hint" style={{ marginTop: 0 }}>
                Decoders see the whole {tmin}–{tmax} s epoch ({windowLen.toFixed(1)} s).
              </p>
              <button className="link-btn" onClick={() => setCropPart(0, tmin)}>Analyse a narrower window</button>
            </>
          )}
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
              className={isOverridden("reference") ? "overridden" : ""}
              onChange={(e) => update("reference", e.target.value)}>
              <option value="average">Common average</option>
              <option value="Cz">Cz</option>
              <option value="none">None</option>
              {Array.isArray(valueOf("reference")) && (
                <option value="custom">{valueOf("reference").join(" + ")} (preset)</option>
              )}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PreprocessSection;
