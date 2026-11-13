import { useState } from "react";
import * as api from "../api";

const ROLES = ["Ignore", "EEG Channel", "Time", "Label"];

function ColumnPicker({ uploadData, onConfigured }) {
  const columns = uploadData.column_info?.columns || [];

  const initialRoles = {};
  columns.forEach((col) => {
    const lower = col.toLowerCase();
    if (lower.includes("time")) initialRoles[col] = "Time";
    else if (lower.includes("label") || lower.includes("class")) initialRoles[col] = "Label";
    else if (["eog", "ecg", "emg", "sti"].some((k) => lower.includes(k))) initialRoles[col] = "Ignore";
    else initialRoles[col] = "EEG Channel";
  });

  const [roles, setRoles] = useState(initialRoles);
  const [samplingRate, setSamplingRate] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleRoleChange = (col, role) => {
    setRoles((prev) => ({ ...prev, [col]: role }));
  };

  const handleSubmit = async () => {
    setError(null);
    const eeg_columns = columns.filter((col) => roles[col] === "EEG Channel");
    const time_column = columns.find((col) => roles[col] === "Time") || null;
    const label_column = columns.find((col) => roles[col] === "Label") || null;

    if (eeg_columns.length === 0) {
      setError("Select at least one EEG channel.");
      return;
    }
    if (!samplingRate || Number(samplingRate) <= 0) {
      setError("Enter a valid sampling rate.");
      return;
    }

    setIsSubmitting(true);
    try {
      const data = await api.configureColumns({
        session_id: uploadData.session_id,
        eeg_columns,
        time_column,
        label_column,
        sampling_rate: Number(samplingRate),
      });
      onConfigured(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div>
      <h3 className="subheading">Configure Columns</h3>
      <table className="column-table">
        <thead>
          <tr><th>Column</th><th>Role</th></tr>
        </thead>
        <tbody>
          {columns.map((col) => (
            <tr key={col}>
              <td>{col}</td>
              <td>
                <select value={roles[col]} onChange={(e) => handleRoleChange(col, e.target.value)}>
                  {ROLES.map((role) => <option key={role} value={role}>{role}</option>)}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="sampling-rate-input">
        <label>Sampling Rate (Hz)</label>
        <input type="number" value={samplingRate} onChange={(e) => setSamplingRate(e.target.value)} placeholder="e.g. 250" />
      </div>

      {error && <div className="error-box">{error}</div>}

      <button className="primary-btn" onClick={handleSubmit} disabled={isSubmitting}>
        {isSubmitting ? "Loading…" : "Load data"}
      </button>
    </div>
  );
}

export default ColumnPicker;