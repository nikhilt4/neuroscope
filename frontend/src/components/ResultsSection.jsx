import { useEffect, useState } from "react";
import ParamsExport from "./ParamsExport";
import PSDChart from "./PSDChart";
import { StatTile, AccuracyBars, ConfusionTable, SubjectGrid } from "./results/ResultParts";
import { FAMILY, describeConfusion, macroF1, prettyClass } from "./results/metrics";
import { MODE_LABELS } from "./wizard/SummaryPanel";

const BAND_LABEL = { delta: "δ delta", theta: "θ theta", alpha: "α alpha", beta: "β beta", gamma: "γ gamma" };

/**
 * Step 5. Same layout and components as the landing-page preview
 * (components/landing/ResultsPreview.jsx), which was approved as the design,
 * fed by the live /analyze response instead of fixed numbers.
 */
function ResultsSection({ results, datasetLabel }) {
  // null = follow the best decoder by the page's own metric (balanced accuracy when the
  // classes are imbalanced), not the backend's plain-accuracy best_classifier
  const [cmFor, setCmFor] = useState(null);
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

  // Majority-class share: the accuracy you get by always guessing the biggest class.
  const counts = Object.values(summary.class_counts || {});
  const k = summary.classes.length;
  const majority = counts.length ? Math.max(...counts) / counts.reduce((a, b) => a + b, 0) : 1 / k;
  // Imbalanced data (ERP CORE: 87% correct responses) makes plain accuracy misleading —
  // a do-nothing model scores 87%. Rank and plot by balanced accuracy (mean recall per
  // class), whose chance level is 1/k regardless of the split.
  const imbalanced = majority > 1 / k + 0.05 && succeeded.every((c) => c.balanced_accuracy_mean != null);
  const score = (c) => (imbalanced ? c.balanced_accuracy_mean : c.accuracy_mean);
  const spread = (c) => (imbalanced ? c.balanced_accuracy_std : c.accuracy_std);
  const chance = imbalanced ? 1 / k : majority;
  const pct = (v) => `${(v * 100).toFixed(1)}%`;

  const best = succeeded.length ? succeeded.reduce((a, b) => (score(b) > score(a) ? b : a)) : null;
  const shown = succeeded.find((c) => c.classifier === cmFor) || best;

  const rows = succeeded
    .slice()
    .sort((a, b) => score(b) - score(a))
    .map((c) => ({
      name: c.classifier,
      family: FAMILY[c.category] || "classical",
      acc: score(c) * 100,
      sd: spread(c) * 100,
    }));

  const bands = results.topomaps ? Object.keys(results.topomaps) : [];
  const activeBand = band && results.topomaps?.[band] ? band : (bands.includes("alpha") ? "alpha" : bands[0]);
  const trialsPerFold = best ? Math.round(summary.n_epochs / best.n_folds) : null;
  // multi-subject runs: "folds" are subjects, so the spread is between PEOPLE, not folds
  const multi = results.evaluation && results.evaluation !== "single";
  const evalLabel = { within_subject: "Within-subject", cross_subject: "Cross-subject (leave one out)" }[results.evaluation];
  const subject = multi
    ? `${results.subjects.length} subjects`
    : summary.subjects?.length ? `Subject ${summary.subjects.join(", ")}` : "Your recording";
  const subjectOf = (i) => best?.fold_groups?.[i];
  const minIdx = best ? best.fold_scores.indexOf(Math.min(...best.fold_scores)) : -1;
  const maxIdx = best ? best.fold_scores.indexOf(Math.max(...best.fold_scores)) : -1;

  return (
    <div className={`rp-window ${revealed ? "is-in" : ""}`}>
      <div className="rp-chrome">
        <span className="rp-url">neuroscope / results</span>
        {multi && <span className="rp-eval-badge">{evalLabel}</span>}
        <span className="rp-badge">{subject}</span>
      </div>

      <div className="rp-body">
        <div className="rp-title-row">
          <div>
            <h3>{datasetLabel || summary.dataset} · {MODE_LABELS[results.mode] || results.mode}</h3>
            <p>
              {summary.n_epochs} trials{multi ? ` from subjects ${results.subjects.join(", ")}` : ""} · {summary.n_channels} channels ·{" "}
              {summary.sampling_rate} Hz · {summary.classes.length} classes
              {best && !multi ? ` · ${best.n_folds}-fold cross-validation` : ""}
              {multi ? ` · ${evalLabel.toLowerCase()}` : ""}
            </p>
          </div>
        </div>

        {best && (
          <div className="rp-stats">
            <StatTile tone="blue" icon="%"
                      label={`Best ${multi ? "mean " : ""}${imbalanced ? "balanced " : ""}accuracy`}
                      value={pct(score(best))}
                      sub={imbalanced ? `${best.classifier} · plain accuracy ${pct(best.accuracy_mean)}` : best.classifier} />
            <StatTile tone="green" icon="±" label={multi ? "Spread across subjects" : "Fold spread"} value={pct(spread(best))}
                      sub={multi && !imbalanced
                        ? `S${subjectOf(minIdx)} ${pct(best.fold_scores[minIdx])} – S${subjectOf(maxIdx)} ${pct(best.fold_scores[maxIdx])}`
                        : imbalanced ? "balanced accuracy, across folds"
                        : `${pct(Math.min(...best.fold_scores))} – ${pct(Math.max(...best.fold_scores))}`} />
            <StatTile tone="violet" icon="F1" label="Macro F1"
                      value={best.confusion_matrix ? macroF1(best.confusion_matrix).toFixed(2) : "—"}
                      sub={multi ? "all subjects' trials" : null} />
            <StatTile tone="slate" icon="∅" label={imbalanced ? "Chance (balanced)" : "Chance level"} value={pct(chance)}
                      sub={imbalanced ? `always guessing the biggest class: ${pct(majority)} accuracy`
                        : counts.length ? "largest class share" : null} />
          </div>
        )}

        {imbalanced && best && (
          <p className="rp-caveat">
            <b>Classes are imbalanced</b> (the biggest is {pct(majority)} of trials), so a model that always guesses it
            would look {pct(majority)} accurate while detecting nothing. Decoders are ranked by <b>balanced accuracy</b>,
            the average of each class&apos;s hit rate, where chance is {pct(1 / k)}.
          </p>
        )}

        {multi && succeeded.length > 0 && (
          <section className="rp-card">
            <h4>Accuracy by subject <span>{results.evaluation === "cross_subject" ? "each subject tested on a model trained on the others" : "each subject trained and tested on their own data"}</span></h4>
            <SubjectGrid classifiers={succeeded} subjects={results.subjects} chance={chance} best={best?.classifier} />
            <p className="rp-note">
              Greener means further above chance. A pale column is a subject every decoder struggles with,
              which is usually the person, not the method.
            </p>
          </section>
        )}

        {succeeded.length > 0 && (
          <div className="rp-grid">
            <section className="rp-card">
              <h4>{imbalanced ? "Balanced accuracy" : "Accuracy"} by decoder <span>{multi ? "mean ± spread across subjects" : "click one to inspect it"}</span></h4>
              <AccuracyBars rows={rows} chance={chance * 100} selected={shown?.classifier}
                            onSelect={setCmFor} best={best?.classifier} showSd />
            </section>

            {shown?.confusion_matrix && (
              <section className="rp-card">
                <h4>Confusion matrix <span>{shown.classifier} · row %{multi ? ` · all ${results.subjects.length} subjects` : ""}</span></h4>
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
              {results.visuals_note ? ` ${results.visuals_note}` : ""}
            </p>
          </section>
        )}

        {results.psd && (
          <section className="rp-card">
            <h4>Power spectrum by class <span>log₁₀ power</span></h4>
            <PSDChart psd={results.psd} />
          </section>
        )}

        {multi && best && (
          <p className="rp-caveat">
            <b>Reading the spread:</b> ± here is the spread between people, not between folds.
            {results.evaluation === "cross_subject"
              ? " Cross-subject accuracy is usually far lower than within-subject: each person's brain signals look different, so a model trained on others has to generalise to a stranger."
              : " A wide spread is normal in BCI: some people's signals separate cleanly and others barely at all."}
          </p>
        )}

        {!multi && succeeded.length > 1 && trialsPerFold && (
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
