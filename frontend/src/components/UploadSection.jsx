import { useState } from "react";
import ColumnPicker from "./ColumnPicker";
import DatasetPicker from "./DatasetPicker";

// const API_BASE = "http://127.0.0.1:8000";
import * as api from "../api";

const UPLOAD_TYPES = ["Single", "Multiple", "Public Dataset", "BIDS"];

function UploadSection({ onReady }) {
  const [uploadType, setUploadType] = useState("Single");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadData, setUploadData] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
    setError(null);
    setUploadData(null);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    setIsUploading(true);
    setError(null);

    try {
      const data = uploadType === "Single"
        ? await api.uploadSingle(selectedFiles[0])
        : await api.uploadMulti(selectedFiles);

      setUploadData(data);
      // MAT files need the variable picker, same round-trip shape as CSV
      if (!data.needs_column_picker && !data.needs_variable_picker) {
        onReady(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      <div className="tab-bar">
        {UPLOAD_TYPES.map((type) => (
          <button
            key={type}
            className={`tab ${uploadType === type ? "tab-active" : ""}`}
            disabled={type === "BIDS"}
            onClick={() => {
              setUploadType(type);
              setSelectedFiles([]);
              setUploadData(null);
              setError(null);
            }}
          >
            {type}
            {type === "BIDS" && <span className="soon-badge">soon</span>}
          </button>
        ))}
      </div>

      {uploadType === "Public Dataset" ? (
        <DatasetPicker onReady={onReady} />
      ) : (
        <>
          <div className="upload-box">
            <input
              type="file"
              multiple={uploadType === "Multiple"}
              onChange={handleFileChange}
              accept=".edf,.bdf,.gdf,.fif,.vhdr,.vmrk,.eeg,.mat,.csv,.tsv"
            />
            {selectedFiles.length > 0 && (
              <p className="file-list">{selectedFiles.map((f) => f.name).join(", ")}</p>
            )}
          </div>

          <button
            className="primary-btn"
            onClick={handleUpload}
            disabled={selectedFiles.length === 0 || isUploading}
          >
            {isUploading ? "Uploading..." : "Upload"}
          </button>

          {error && <div className="error-box">{error}</div>}

          {uploadData?.needs_column_picker && (
            <ColumnPicker uploadData={uploadData} onConfigured={onReady} />
          )}
          {uploadData?.needs_variable_picker && (
            <p className="error-box">
              MAT files need a variable picker, which isn't built yet.
            </p>
          )}
        </>
      )}
    </div>
  );
}

export default UploadSection;