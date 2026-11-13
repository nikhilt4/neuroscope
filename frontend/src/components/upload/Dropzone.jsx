import { useRef, useState } from "react";
import { Icon } from "../ui/Icons";

const ACCEPT = ".edf,.bdf,.gdf,.fif,.vhdr,.vmrk,.eeg,.mat,.csv,.tsv,.zip";

const FORMATS = [
  { name: "EDF · BDF · GDF", status: "ready" },
  { name: "FIF", status: "ready" },
  { name: "BIDS .zip", status: "ready" },
  { name: "BrainVision", status: "beta" },
  { name: "CSV / TSV", status: "beta" },
  { name: "MATLAB .mat", status: "soon" },
];

function fmtSize(bytes) {
  if (bytes > 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes > 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

const ext = (f) => f.name.split(".").pop().toLowerCase();

/**
 * Works out which backend route a selection needs, so the user never has to
 * choose between "Single", "Multiple" and "BIDS" tabs themselves.
 * Returns { kind: "single" | "multi" | "bids", error?: string }.
 */
export function classifySelection(files) {
  if (files.length === 0) return { kind: null };
  const exts = files.map(ext);
  if (exts.includes("zip")) {
    return files.length === 1
      ? { kind: "bids" }
      : { kind: null, error: "A BIDS dataset goes in as a single .zip. Remove the other files." };
  }
  if (files.length === 1) {
    if (exts[0] === "vhdr" || exts[0] === "vmrk" || exts[0] === "eeg") {
      return { kind: null, error: "BrainVision recordings are three files (.vhdr, .vmrk and .eeg). Select all three together." };
    }
    return { kind: "single" };
  }
  if (!exts.includes("vhdr")) {
    return { kind: null, error: "Several files only work as a BrainVision set (.vhdr + .vmrk + .eeg). For anything else, upload one file at a time." };
  }
  return { kind: "multi" };
}

const KIND_LABEL = { single: "Single recording", multi: "BrainVision set", bids: "BIDS dataset" };

function Dropzone({ files, setFiles, disabled }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const selection = classifySelection(files);

  const add = (list) => setFiles(Array.from(list));

  return (
    <div>
      <div
        className={`up-drop ${over ? "is-over" : ""} ${files.length ? "has-files" : ""}`}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (!disabled) add(e.dataTransfer.files); }}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        aria-label="Choose EEG files to upload"
      >
        <input ref={inputRef} type="file" multiple accept={ACCEPT} hidden
               onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
        <span className="up-drop-icon"><Icon name="cloud" /></span>
        <p className="up-drop-title"><b>Drop your EEG file here</b> or <u>browse</u></p>
        <p className="up-drop-sub">One recording, a BrainVision set, or a BIDS dataset as a .zip</p>
        <ul className="up-formats" onClick={(e) => e.stopPropagation()}>
          {FORMATS.map((f) => (
            <li key={f.name} className={`fmt-${f.status}`}>
              {f.name}{f.status !== "ready" && <em>{f.status === "beta" ? "beta" : "soon"}</em>}
            </li>
          ))}
        </ul>
      </div>

      {files.length > 0 && (
        <div className="up-files">
          <div className="up-files-head">
            <span>{selection.kind ? KIND_LABEL[selection.kind] : "Selection"}</span>
            <button className="up-link" onClick={() => setFiles([])} disabled={disabled}>Clear</button>
          </div>
          <ul>
            {files.map((f) => (
              <li key={f.name}>
                <Icon name={ext(f) === "zip" ? "zip" : "file"} />
                <span className="up-file-name">{f.name}</span>
                <span className="up-file-size">{fmtSize(f.size)}</span>
                <button aria-label={`Remove ${f.name}`} disabled={disabled}
                        onClick={() => setFiles(files.filter((x) => x !== f))}>
                  <Icon name="x" />
                </button>
              </li>
            ))}
          </ul>
          {selection.error && <p className="up-warn"><Icon name="alert" />{selection.error}</p>}
        </div>
      )}
    </div>
  );
}

export default Dropzone;
