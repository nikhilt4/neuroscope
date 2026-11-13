import montage from "../../data/montage_1020.json";

/**
 * Head map of the channels a dataset ACTUALLY has, placed at their real
 * standard_1020 positions (exported from MNE 1.12 — see data/montage_1020.json).
 *
 * Present channels are filled; the remaining standard sites are drawn faintly so
 * the head reads as a head. Channels with non-standard names (e.g. "EEG 1" from
 * a MAT file, or bipolar pairs) can't be placed and are counted instead.
 */
const POS = montage.positions;
const CANON = Object.fromEntries(Object.keys(POS).map((k) => [k.toLowerCase(), k]));

// Head geometry from the montage itself: centre between T7/T8 and Fpz/Oz.
const CX = (POS.T7[0] + POS.T8[0]) / 2;
const CY = (POS.Fpz[1] + POS.Oz[1]) / 2;
const RADIUS = Math.max(...Object.values(POS).map(([x, y]) => Math.hypot(x - CX, y - CY))) * 1.08;

const SIZE = 240;
const PAD = 22;
const SCALE = (SIZE / 2 - PAD) / RADIUS;
const toSvg = ([x, y]) => [SIZE / 2 + (x - CX) * SCALE, SIZE / 2 - (y - CY) * SCALE];  // layout y is up

// Recording systems sometimes append dots or prefixes: "C3..", "EEG C3", "C3-REF"
function canonical(name) {
  const cleaned = name.replace(/^EEG[\s_-]*/i, "").replace(/[-_\s]?(REF|LE|AVG)$/i, "").replace(/\.+$/, "");
  return CANON[cleaned.toLowerCase()];
}

function ChannelMap({ channels = [], highlight = ["C3", "Cz", "C4"] }) {
  const present = new Set();
  let unplaced = 0;
  channels.forEach((ch) => {
    const c = canonical(ch);
    if (c) present.add(c); else unplaced += 1;
  });
  const hl = new Set(highlight);
  const r = RADIUS * SCALE;
  const c = SIZE / 2;

  return (
    <figure className="wz-chmap">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img"
           aria-label={`Head map: ${present.size} channels at standard 10-20 positions${unplaced ? `, ${unplaced} not placeable` : ""}.`}>
        <path d={`M${c - 11} ${c - r + 2} L${c} ${c - r - 13} L${c + 11} ${c - r + 2}`} className="wz-chmap-outline" />
        <path d={`M${c - r + 1} ${c - 14} q-11 14 0 28`} className="wz-chmap-outline" />
        <path d={`M${c + r - 1} ${c - 14} q11 14 0 28`} className="wz-chmap-outline" />
        <circle cx={c} cy={c} r={r} className="wz-chmap-head" />

        {Object.entries(POS).map(([name, p]) => {
          if (["T3", "T4", "T5", "T6"].includes(name)) return null;   // aliases, drawn as T7/T8/P7/P8
          const [x, y] = toSvg(p);
          const on = present.has(name) ||
            (name === "T7" && present.has("T3")) || (name === "T8" && present.has("T4")) ||
            (name === "P7" && present.has("T5")) || (name === "P8" && present.has("T6"));
          const key = on && hl.has(name);
          return (
            <g key={name} className={key ? "wz-ch wz-ch-key" : on ? "wz-ch wz-ch-on" : "wz-ch"}>
              <title>{name}{on ? "" : " (not in this recording)"}</title>
              <circle cx={x} cy={y} r={key ? 5 : on ? 3.4 : 1.6} />
              {key && <text x={x} y={y + 14} textAnchor="middle">{name}</text>}
            </g>
          );
        })}
      </svg>
      <figcaption>
        <span><i className="wz-ch-swatch wz-ch-on" />{present.size} placed</span>
        {unplaced > 0 && <span><i className="wz-ch-swatch" />{unplaced} non-standard name{unplaced > 1 ? "s" : ""}</span>}
      </figcaption>
    </figure>
  );
}

export default ChannelMap;
