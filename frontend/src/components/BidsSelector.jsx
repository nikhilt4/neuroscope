import { useState, useMemo } from "react";
import * as api from "../api";

function BidsSelector({ uploadData, onConfigured }) {
  const entities = uploadData.bids_entities || [];
  const [subject, setSubject] = useState("");
  const [task, setTask] = useState("");
  const [session, setSession] = useState("");
  const [run, setRun] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Entities come back as a flat list of available combinations. Each dropdown
  // narrows the options for the next — picking a subject shouldn't offer tasks
  // that subject doesn't have.
  const subjects = useMemo(
    () => [...new Set(entities.map((e) => e.subject).filter(Boolean))].sort(),
    [entities]
  );

  const matching = useMemo(
    () => entities.filter((e) => !subject || e.subject === subject),
    [entities, subject]
  );

  const optionsFor = (key) =>
    [...new Set(matching.map((e) => e[key]).filter(Boolean))].sort();

  const tasks = optionsFor("task");
  const sessions = optionsFor("session");
  const runs = optionsFor("run");

  const submit = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.configureBids({
        session_id: uploadData.session_id,
        subject,
        task: task || null,
        session: session || null,
        run: run || null,
      });
      onConfigured(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  if (entities.length === 0) {
    return <div className="error-box">No BIDS entities found in this dataset.</div>;
  }

  const Picker = ({ label, value, setValue, options, required }) =>
    options.length > 0 && (
      <div className="config-row">
        <label>{label}</label>
        <select value={value} onChange={(e) => setValue(e.target.value)}>
          <option value="">{required ? "select…" : "any"}</option>
          {options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      </div>
    );

  return (
    <div className="config-group" style={{ marginTop: 20 }}>
      <h4>Select Recording</h4>
      <Picker label="Subject" value={subject} setValue={setSubject}
              options={subjects} required />
      <Picker label="Task" value={task} setValue={setTask} options={tasks} />
      <Picker label="Session" value={session} setValue={setSession} options={sessions} />
      <Picker label="Run" value={run} setValue={setRun} options={runs} />

      {error && <div className="error-box">{error}</div>}

      <button className="primary-btn" onClick={submit}
              disabled={!subject || isLoading} style={{ marginTop: 12 }}>
        {isLoading ? "Loading…" : "Load Recording"}
      </button>
    </div>
  );
}

export default BidsSelector;