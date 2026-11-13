import { f1Scores, prettyClass } from "./metrics";
import "../../styles/results.css";

/** Coloured-circle stat tile, as in the approved results design. */
export function StatTile({ icon, label, value, sub, tone }) {
  return (
    <div className="rp-stat">
      <span className={`rp-stat-icon rp-tone-${tone}`} aria-hidden="true">{icon}</span>
      <div>
        <div className="rp-stat-label">{label}</div>
        <div className="rp-stat-value">{value}</div>
        {sub && <div className="rp-stat-sub">{sub}</div>}
      </div>
    </div>
  );
}

/**
 * Horizontal accuracy bars, coloured by family, with a dashed chance marker drawn
 * per row (CSS ::after at --chance) so it lines up exactly on every row.
 * rows: [{ name, label?, family, acc (0–100), sd? (0–100) }]
 * With onSelect, each row is a button (used to pick the confusion matrix).
 */
export function AccuracyBars({ rows, chance, selected, onSelect, best, showSd = false }) {
  return (
    <>
      <div className={`rp-bars ${onSelect ? "rp-bars-live" : ""}`} style={{ "--chance": `${chance}%` }}>
        {rows.map((r, i) => {
          const inner = (
            <>
              <span className="rp-bar-name" title={r.label || r.name}>
                {r.name}{r.name === best && <span className="rp-star" aria-label="best">★</span>}
              </span>
              <span className="rp-bar-track">
                <span className={`rp-bar rp-${r.family}`} style={{ "--w": `${r.acc}%`, "--d": `${i * 60}ms` }} />
              </span>
              <span className="rp-bar-val">
                {r.acc.toFixed(1)}%{showSd && r.sd != null && <small>±{r.sd.toFixed(1)}</small>}
              </span>
            </>
          );
          return onSelect ? (
            <button key={r.name} type="button" className={`rp-bar-row ${selected === r.name ? "is-selected" : ""}`}
                    onClick={() => onSelect(r.name)} aria-pressed={selected === r.name}>
              {inner}
            </button>
          ) : (
            <div className="rp-bar-row" key={r.name}>{inner}</div>
          );
        })}
      </div>
      <div className="rp-legend">
        <span><i className="rp-riemannian" />Riemannian</span>
        <span><i className="rp-classical" />Classical</span>
        <span><i className="rp-deep" />Deep</span>
        <span><i className="rp-chance-key" />Chance</span>
      </div>
    </>
  );
}

function cellStyle(pct, diagonal) {
  if (diagonal) return { background: `rgba(52, 199, 123, ${0.25 + (pct / 100) * 0.7})`, color: "#06331c", fontWeight: 700 };
  if (pct >= 5) return { background: `rgba(244, 114, 142, ${Math.min(0.45, pct / 40)})` };
  return {};
}

/**
 * Row-normalised confusion table: each cell is "of the trials that truly were X,
 * the share predicted as Y". Raw counts mislead when classes are unbalanced, so
 * colour by share and keep the count in the tooltip.
 */
export function ConfusionTable({ matrix, classNames }) {
  const f1 = f1Scores(matrix);
  const names = classNames.map(prettyClass);
  return (
    <div className="rp-cm-scroll">
      <table className="rp-cm">
        <thead>
          <tr><th>True ↓ / Pred →</th>{names.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={names[i]}>
                <th>{names[i]}</th>
                {row.map((n, j) => {
                  const pct = (100 * n) / total;
                  return (
                    <td key={j} style={cellStyle(pct, i === j)}
                        title={`${n} of ${total} ${names[i].toLowerCase()} trials predicted ${names[j].toLowerCase()}`}>
                      {pct.toFixed(1)}%
                    </td>
                  );
                })}
              </tr>
            );
          })}
          <tr className="rp-f1">
            <th>F1 score</th>
            {f1.map((f, j) => <td key={j}>{f.toFixed(2)}</td>)}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
