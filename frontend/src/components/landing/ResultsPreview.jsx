import topomapAlpha from "../../assets/topomap-alpha-bnci2014001-s1.png";
import { StatTile, AccuracyBars, ConfusionTable } from "../results/ResultParts";
import { macroF1 } from "../results/metrics";

/**
 * A static preview of the results page, built from a REAL run:
 * BNCI2014_001 (BCI Competition IV 2a), subject 1, motor imagery mode,
 * 5-fold stratified CV (random_state=42), default settings, 13 Nov 2026.
 * Accuracies are pinned by backend/tests/test_integration_bnci.py.
 * Uses the same components as the live Results step (components/results/).
 * If the pipeline changes, regenerate these rather than editing by hand.
 */
const CLASSES = ["left_hand", "right_hand", "feet", "tongue"];
// riemann_ts out-of-fold predictions; rows = true class, columns = predicted
const CONFUSION = [
  [122, 16, 4, 2],
  [12, 129, 3, 0],
  [4, 5, 113, 22],
  [4, 0, 24, 116],
];
const ACCURACY = [
  { name: "riemann_ts", label: "Riemannian tangent space", family: "riemannian", acc: 83.3 },
  { name: "riemann_fgmdm", label: "FgMDM", family: "riemannian", acc: 79.7 },
  { name: "xdawn_ts", label: "xDAWN + tangent space", family: "riemannian", acc: 75.7 },
  { name: "riemann_mdm", label: "MDM", family: "riemannian", acc: 74.1 },
  { name: "svm", label: "CSP + SVM", family: "classical", acc: 74.1 },
  { name: "lda", label: "CSP + LDA", family: "classical", acc: 73.4 },
  { name: "eegnet", label: "EEGNet", family: "deep", acc: 71.9 },
];

function ResultsPreview() {
  return (
    <div className="rp-window">
      <div className="rp-chrome">
        <span className="rp-url">neuroscope / results</span>
        <span className="rp-badge">Subject 1</span>
      </div>

      <div className="rp-body">
        <div className="rp-title-row">
          <div>
            <h3>BCI Competition IV 2a · Motor imagery</h3>
            <p>576 trials · 22 channels · 250 Hz · 4 classes · 5-fold cross-validation</p>
          </div>
        </div>

        <div className="rp-stats">
          <StatTile tone="blue" icon="%" label="Best accuracy" value="83.3%" />
          <StatTile tone="green" icon="±" label="Fold spread" value="3.3%" />
          <StatTile tone="violet" icon="F1" label="Macro F1" value={macroF1(CONFUSION).toFixed(2)} />
          <StatTile tone="slate" icon="∅" label="Chance level" value="25%" />
        </div>

        <div className="rp-grid">
          <section className="rp-card">
            <h4>Accuracy by decoder <span>mean of 5 folds</span></h4>
            <AccuracyBars rows={ACCURACY} chance={25} />
          </section>

          <section className="rp-card">
            <h4>Confusion matrix <span>riemann_ts · row %</span></h4>
            <ConfusionTable matrix={CONFUSION} classNames={CLASSES} />
            <p className="rp-note">Hands separate cleanly. Feet and tongue blur into each other: both sit on the midline of the motor cortex.</p>
          </section>
        </div>

        <section className="rp-card rp-topo">
          <h4>Alpha-band scalp maps <span>8–13 Hz · power vs. mean</span></h4>
          <img src={topomapAlpha} alt="Alpha-band scalp maps for left hand, right hand, feet and tongue imagery. Left-hand imagery lowers power over the right hemisphere; right-hand imagery over the left." />
          <p className="rp-note">Blue means less power than average. Each hand quietens the <b>opposite</b> hemisphere: the mu-rhythm drop motor imagery BCIs are built on.</p>
        </section>
      </div>
    </div>
  );
}

export default ResultsPreview;
