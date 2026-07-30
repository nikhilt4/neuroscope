import { useState } from "react";
import ColumnPicker from "./ColumnPicker";

const API_BASE = "http://127.0.0.1:8000";
const UPLOAD_TYPES = ["Single", "Multiple", "BIDS"];

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

    const formData = new FormData();
    const endpoint = uploadType === "Single" ? "/upload/single" : "/upload/multi";

    if (uploadType === "Single") {
      formData.append("file", selectedFiles[0]);
    } else {
      selectedFiles.forEach((f) => formData.append("files", f));
    }

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");

      setUploadData(data);
      if (!data.needs_column_picker) {
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

      <div className="upload-box">
        <input
          type="file"
          multiple={uploadType === "Multiple"}
          onChange={handleFileChange}
          accept=".edf,.bdf,.gdf,.fif,.vhdr,.vmrk,.eeg,.mat,.csv,.tsv"
        />
        {selectedFiles.length > 0 && (
          <p className="file-list">
            {selectedFiles.map((f) => f.name).join(", ")}
          </p>
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
    </div>
  );
}

export default UploadSection;