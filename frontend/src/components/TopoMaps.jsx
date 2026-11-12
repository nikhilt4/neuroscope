import { useState } from "react";

function TopoMaps({ topomaps }) {
  const bands = topomaps ? Object.keys(topomaps) : [];
  const [band, setBand] = useState(bands[0] ?? null);

  if (!topomaps || bands.length === 0) return null;
  const active = topomaps[band] || topomaps[bands[0]];

  return (
    <div className="chart-block">
      <div className="chart-head">
        <h3 className="subheading">Scalp Topography</h3>
        <select value={band} onChange={(e) => setBand(e.target.value)}>
          {bands.map((b) => (
            <option key={b} value={b}>
              {b} ({topomaps[b].range_hz[0]}–{topomaps[b].range_hz[1]} Hz)
            </option>
          ))}
        </select>
      </div>
      <p className="group-hint">
        Power in the selected band, per class, relative to the average across
        all classes. Red means more power than average at that location, blue
        means less — so the pattern shows where the classes differ.
      </p>
      <img className="topomap-img"
           src={`data:image/png;base64,${active.image}`}
           alt={`${band} band topography`} />
    </div>
  );
}

export default TopoMaps;