import { useState } from "react";
import ColumnPicker from "./ColumnPicker";
import DatasetPicker from "./DatasetPicker";
import BidsSelector from "./BidsSelector";
import Dropzone, { classifySelection } from "./upload/Dropzone";
import DetectedCard from "./upload/DetectedCard";
import { Icon } from "./ui/Icons";
import * as api from "../api";
import "./upload/Upload.css";

const UPLOADERS = { single: (f) => api.uploadSingle(f[0]), multi: api.uploadMulti, bids: (f) => api.uploadBids(f[0]) };

/**
 * Step 1. Two routes in (public dataset or your own files), one way out:
 * onReady(data) with the backend's UploadResponse. Once data is loaded the
 * step shows what was detected instead of the pickers.
 */
function UploadSection({ loaded, onReady }) {
  const [route, setRoute] = useState("public");
  const [changing, setChanging] = useState(false);
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(null);   // upload that needs a second step

  const finish = (data) => {
    setPending(null);
    setFiles([]);
    setChanging(false);
    onReady(data);
  };

  const handleUpload = async () => {
    const { kind } = classifySelection(files);
    if (!kind) return;
    setIsUploading(true);
    setError(null);
    setPending(null);
    try {
      const data = await UPLOADERS[kind](files);
      // CSV, MAT and BIDS need a second round trip (column / variable / recording picker)
      if (data.needs_column_picker || data.needs_variable_picker || data.needs_bids_selection) setPending(data);
      else finish(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  if (loaded && !changing) {
    return <DetectedCard data={loaded} onChange={() => setChanging(true)} />;
  }

  const selection = classifySelection(files);

  return (
    <div className="up">
      {loaded && (
        <button className="up-link up-back" onClick={() => setChanging(false)}>
          <Icon name="arrowLeft" />Keep using {loaded.dataset_name || "the current data"}
        </button>
      )}

      <div className="up-routes" role="tablist" aria-label="Where your data comes from">
        <button role="tab" aria-selected={route === "public"}
                className={`up-route ${route === "public" ? "is-active" : ""}`} onClick={() => setRoute("public")}>
          <span className="up-route-icon"><Icon name="database" /></span>
          <span>
            <b>Use a public dataset</b>
            <small>Benchmark EEG from MNE, MOABB and braindecode. Nothing to download yourself.</small>
          </span>
          <em className="up-route-badge">Quickest start</em>
        </button>
        <button role="tab" aria-selected={route === "file"}
                className={`up-route ${route === "file" ? "is-active" : ""}`} onClick={() => setRoute("file")}>
          <span className="up-route-icon"><Icon name="upload" /></span>
          <span>
            <b>Upload your own data</b>
            <small>EDF, BDF, GDF, FIF, BrainVision, CSV or a BIDS dataset as a .zip.</small>
          </span>
        </button>
      </div>

      {route === "public" ? (
        <DatasetPicker onReady={finish} />
      ) : (
        <div className="up-panel">
          <Dropzone files={files} setFiles={(f) => { setFiles(f); setError(null); setPending(null); }} disabled={isUploading} />

          <div className="up-actions">
            <button className="wz-btn" onClick={handleUpload} disabled={!selection.kind || isUploading}>
              {isUploading ? <><span className="wz-spinner" />Uploading…</> : <>Upload and inspect<Icon name="arrowRight" /></>}
            </button>
            <span className="up-hint">Files are sent to the NeuroScope backend and used only for this analysis.</span>
          </div>

          {error && <div className="wz-alert"><Icon name="alert" /><span>{error}</span></div>}

          {pending?.needs_column_picker && <ColumnPicker uploadData={pending} onConfigured={finish} />}
          {pending?.needs_variable_picker && (
            <div className="wz-alert wz-alert-info">
              <Icon name="info" />
              <span>MATLAB files need a variable picker to say which array holds the EEG. That&apos;s not built yet. Export to EDF or CSV for now.</span>
            </div>
          )}
          {pending?.needs_bids_selection && <BidsSelector uploadData={pending} onConfigured={finish} />}
        </div>
      )}
    </div>
  );
}

export default UploadSection;
