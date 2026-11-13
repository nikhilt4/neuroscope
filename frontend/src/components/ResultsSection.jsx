import { useEffect, useState } from "react";
import ParamsExport from "./ParamsExport";
import PSDChart from "./PSDChart";
import { StatTile, AccuracyBars, ConfusionTable } from "./results/ResultParts";
import { FAMILY, describeConfusion, macroF1, prettyClass } from "./results/metrics";
import { MODE_LABELS } from "./wizard/SummaryPanel";

const BAND_LABEL = { delta: "δ delta", theta: "θ theta", alpha: "α alpha", beta: "β beta", gamma: "γ gamma" };

/**
 * Step 5. Same layout and components as the landing-page preview
 * (components/landing/ResultsPreview.jsx), which was approved as the design,
 * fed by the live /analyze response instead of fixed numbers.
 */
function ResultsSection({ results, datasetLabel }) {
  const [cmFor, setCmFor] = useState(results?.best_classifier ?? null);
  const [band, setBand] = useState(null);
  const [revealed, setRevealed] = useState(false);

  // bars grow in from zero once mounted (the shared CSS animates on .is-in)
  useEffect(() => {
    const id = requestAnimationFrame(() => setRevealed(true));
    return () => cancelAnimationFrame(id);
  }, []);

  if (!results) {
    return <p className="ch-empty">Run the analysis to see results here.</p>;
  }

  const summary = results.dataset_summary;
  const succeeded = results.classifiers.filter((c) => c.accuracy_mean !== null);
  const failed = results.classifiers.filter((c) => c.accuracy_mean === null);
  const best = succeeded.find((c) => c.classifier === results.best_classifier);
  const shown = succeeded.find((c) => c.classifier === cmFor) || best;

  // Chance = share of the largest class, not 1/k: with unbalanced classes,
  // always guessing the biggest one already beats 1/k.
  const counts = Object.values(summary.class_counts || {});
  const chance = counts.length ? Math.max(...counts) / counts.reduce((a, b) => a + b, 0) : 1 / summary.classes.length;
  const pct = (v) => `${(v * 100).toFixed(1)}%`;

  const rows = succeeded
    .slice()
    .sort((a, b) => b.accuracy_mean - a.accuracy_mean)
    .map((c) => ({
      name: c.classifier,
      family: FAMILY[c.category] || "classical",
      acc: c.accuracy_mean * 100,
      sd: c.accuracy_std * 100,
    }));

  const bands = results.topomaps ? Object.keys(results.topomaps) : [];
  const activeBand = band && results.topomaps?.[band] ? band : (bands.includes("alpha") ? "alpha" : bands[0]);
  const trialsPerFold = best ? Math.round(summary.n_epochs / best.n_folds) : null;
  const subject = summary.subjects?.length ? `Subject ${summary.subjects.join(", ")}` : "Your recording";

  return (
    <div className={`rp-window ${revealed ? "is-in" : ""}`}>
      <div className="rp-chrome">
        <span className="rp-url">neuroscope / results</span>
        <span className="rp-badge">{subject}</span>
      </div>

      <div className="rp-body">
        <div className="rp-title-row">
          <div>
            <h3>{datasetLabel || summary.dataset} · {MODE_LABELS[results.mode] || results.mode}</h3>
            <p>
              {summary.n_epochs} trials · {summary.n_channels} channels · {summary.sampling_rate} Hz ·{" "}
              {summary.classes.length} classes{best ? ` · ${best.n_folds}-fold cross-validation` : ""}
            </p>
          </div>
        </div>

        {best && (
          <div className="rp-stats">
            <StatTile tone="blue" icon="%" label="Best accuracy" value={pct(best.accuracy_mean)} sub={best.classifier} />
            <StatTile tone="green" icon="±" label="Fold spread" value={pct(best.accuracy_std)}
                      sub={`${pct(Math.min(...best.fold_scores))} – ${pct(Math.max(...best.fold_scores))}`} />
            <StatTile tone="violet" icon="F1" label="Macro F1"
                      value={best.confusion_matrix ? macroF1(best.confusion_matrix).toFixed(2) : "—"} />
            <StatTile tone="slate" icon="∅" label="Chance level" value={pct(chance)}
                      sub={counts.length ? "largest class share" : null} />
          </div>
        )}

        {succeeded.length > 0 && (
          <div className="rp-grid">
            <section className="rp-card">
              <h4>Accuracy by decoder <span>click one to inspect it</span></h4>
              <AccuracyBars rows={rows} chance={chance * 100} selected={shown?.classifier}
                            onSelect={setCmFor} best={results.best_classifier} showSd />
            </section>

            {shown?.confusion_matrix && (
              <section className="rp-card">
                <h4>Confusion matrix <span>{shown.classifier} · row %</span></h4>
                <ConfusionTable matrix={shown.confusion_matrix} classNames={summary.classes} />
                <p className="rp-note">{describeConfusion(shown.confusion_matrix, summary.classes)}</p>
              </section>
            )}
          </div>
        )}

        {failed.length > 0 && (
          // run_all_classifiers isolates failures so one broken decoder doesn't sink
          // the batch; say which ones didn't run rather than silently dropping them
          <section className="rp-card rp-failed">
            <h4>Did not run <span>{failed.length} decoder{failed.length > 1 ? "s" : ""}</span></h4>
            <ul>{failed.map((c) => <li key={c.classifier}><b>{c.classifier}</b>: {c.error}</li>)}</ul>
          </section>
        )}

        {activeBand && (
          <section className="rp-card rp-topo">
            <h4>
              Scalp maps by class
              <span className="rp-tabs" role="tablist">
                {bands.map((b) => (
                  <button key={b} role="tab" aria-selected={b === activeBand}
                          className={b === activeBand ? "is-active" : ""} onClick={() => setBand(b)}>
                    {BAND_LABEL[b] || b}
                  </button>
                ))}
              </span>
            </h4>
            <img src={`data:image/png;base64,${results.topomaps[activeBand].image}`}
                 alt={`${activeBand} band power for each class, relative to the average across classes`} />
            <p className="rp-note">
              Power in the {activeBand} band ({results.topomaps[activeBand].range_hz.join("–")} Hz) per class, relative
              to the average. Blue means less power than average and red more, so the pattern shows where the classes differ.
            </p>
          </section>
        )}

        {results.psd && (
          <section className="rp-card">
            <h4>Power spectrum by class <span>log₁₀ power</span></h4>
            <PSDChart psd={results.psd} />
          </section>
        )}

        {succeeded.length > 1 && trialsPerFold && (
          <p className="rp-caveat">
            <b>Reading the differences:</b> with about {trialsPerFold} trials per fold, one trial changing sides moves
            accuracy by roughly {pct(1 / trialsPerFold)}. Gaps smaller than the ± spread aren&apos;t meaningful.
            {summary.subjects?.length ? ` Results are for ${subject.toLowerCase()} only; other people can score very differently.` : ""}
          </p>
        )}

        <section className="rp-card">
          <ParamsExport results={results} />
        </section>

        <p className="rp-caveat">Classes: {summary.classes.map((c) => `${prettyClass(c)} (${summary.class_counts?.[c] ?? "?"})`).join(" · ")}</p>
      </div>
    </div>
  );
}

export default ResultsSection;
