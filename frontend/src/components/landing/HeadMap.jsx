import { useEffect, useState } from "react";

/**
 * Top-down head with electrodes at their international 10-20 positions
 * (Jasper 1958; layout as in MNE's "standard_1020" montage, flattened to 2D).
 * Neighbouring electrodes are joined to suggest a sensor network.
 *
 * A slow "light show" lights one random electrode per second: its glow, dot,
 * label and links to its neighbours fade in, then fade out while the next one
 * fades in. The overlap is what makes it read as gradual rather than blinking.
 *
 * Coordinates: unit circle, nose at the top (y = -1).
 */
const ELECTRODES = {
  Fp1: [-0.3, -0.86], Fpz: [0, -0.92], Fp2: [0.3, -0.86],
  F7: [-0.72, -0.52], F3: [-0.38, -0.46], Fz: [0, -0.44], F4: [0.38, -0.46], F8: [0.72, -0.52],
  FC3: [-0.42, -0.22], FCz: [0, -0.22], FC4: [0.42, -0.22],
  T7: [-0.92, 0], C3: [-0.46, 0], Cz: [0, 0], C4: [0.46, 0], T8: [0.92, 0],
  CP3: [-0.42, 0.22], CPz: [0, 0.22], CP4: [0.42, 0.22],
  P7: [-0.72, 0.52], P3: [-0.38, 0.46], Pz: [0, 0.44], P4: [0.38, 0.46], P8: [0.72, 0.52],
  O1: [-0.3, 0.86], Oz: [0, 0.92], O2: [0.3, 0.86],
};
const R = 120;          // head radius in SVG units
const C = 150;          // centre
const STEP_MS = 1000;   // one electrode per second

const pts = Object.entries(ELECTRODES).map(([name, [x, y]]) => ({
  name, x: C + x * R * 0.92, y: C + y * R * 0.92,
}));
const NAMES = pts.map((p) => p.name);

// join electrodes closer than ~0.48 head radii: gives a tidy mesh without clutter
const edges = [];
pts.forEach((a, i) => pts.slice(i + 1).forEach((b) => {
  if (Math.hypot(a.x - b.x, a.y - b.y) < R * 0.48) edges.push([a, b]);
}));

/** A random electrode that isn't one of the last few, so it never stutters in place. */
function pickNext(recent) {
  const pool = NAMES.filter((n) => !recent.includes(n));
  return pool[Math.floor(Math.random() * pool.length)];
}

function HeadMap({ className = "" }) {
  const [recent, setRecent] = useState(["Cz"]);   // recent[0] is the lit electrode
  const lit = recent[0];

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return undefined;
    const id = setInterval(
      () => setRecent((prev) => [pickNext(prev), ...prev].slice(0, 4)),
      STEP_MS
    );
    return () => clearInterval(id);
  }, []);

  return (
    <svg className={className} viewBox="0 0 300 300" role="img"
         aria-label="Top-down view of a head with EEG electrodes in the international 10-20 layout.">
      <defs>
        <radialGradient id="hm-fill" cx="50%" cy="45%" r="60%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#dfe6ff" stopOpacity="0.6" />
        </radialGradient>
        <radialGradient id="hm-glow">
          <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#4f46e5" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* nose + ears */}
      <path d={`M${C - 14} ${C - R + 2} L${C} ${C - R - 18} L${C + 14} ${C - R + 2}`} className="hm-outline" />
      <path d={`M${C - R + 1} ${C - 18} q-14 18 0 36`} className="hm-outline" />
      <path d={`M${C + R - 1} ${C - 18} q14 18 0 36`} className="hm-outline" />
      <circle cx={C} cy={C} r={R} fill="url(#hm-fill)" className="hm-outline" />

      {edges.map(([a, b]) => (
        <line key={a.name + b.name} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              className={a.name === lit || b.name === lit ? "hm-edge hm-edge-lit" : "hm-edge"} />
      ))}

      {pts.map((p, i) => (
        <g key={p.name} className={p.name === lit ? "hm-node hm-lit" : "hm-node"} style={{ "--i": i }}>
          {/* every node owns a glow that just transitions opacity — no remounting,
              so fades in and out are smooth and can overlap between steps */}
          <circle cx={p.x} cy={p.y} r="20" fill="url(#hm-glow)" className="hm-glow" />
          <circle cx={p.x} cy={p.y} r="4" className="hm-dot" />
          <text x={p.x} y={p.y + 17} textAnchor="middle" className="hm-label">{p.name}</text>
        </g>
      ))}
    </svg>
  );
}

export default HeadMap;
