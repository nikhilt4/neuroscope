import { useState, useEffect, useMemo } from "react";
import UploadSection from "../components/UploadSection";
import ChooseSection from "../components/ChooseSection";
import PreprocessSection from "../components/PreprocessSection";
import ClassifySection from "../components/ClassifySection";
import ResultsSection from "../components/ResultsSection";
import Stepper from "../components/wizard/Stepper";
import SummaryPanel from "../components/wizard/SummaryPanel";
import { Icon, Logo } from "../components/ui/Icons";
import * as api from "../api";
import "./Wizard.css";

const STEPS = [
  { name: "Load data", icon: "database",
    title: "Load your data",
    sub: "Pick a public dataset to explore, or bring your own recording. NeuroScope inspects it before anything runs." },
  { name: "Analysis", icon: "choose",
    title: "Choose an analysis",
    sub: "NeuroScope checked what your data supports. Pick how features are computed from it." },
  { name: "Preprocess", icon: "filter",
    title: "Preprocess",
    sub: "Cleaning applied before classification. Presets come from published pipelines, and every value can be changed." },
  { name: "Classify", icon: "classify",
    title: "Pick your decoders",
    sub: "Every decoder is scored with the same stratified cross-validation, so the comparison is fair." },
  { name: "Results", icon: "results",
    title: "Results",
    sub: "How each decoder did, and what the data looks like." },
];

function Wizard({ onExit }) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState("forward");
  const [readyData, setReadyData] = useState(null);
  const [selectedMode, setSelectedMode] = useState(null);
  const [preset, setPreset] = useState({});
  const [preprocessOverrides, setPreprocessOverrides] = useState({});
  const [classifierConfig, setClassifierConfig] = useState({
    selected: ["lda", "svm"],
    n_folds: 5,
    csp_components: 4,
    csp_reg: null,
    configs: {},
  });
  const [results, setResults] = useState(null);
  const [resultsKey, setResultsKey] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!readyData || !selectedMode) return;
    api
      .getPreset({ datasetName: readyData.dataset_name, mode: selectedMode })
      .then(setPreset)
      .catch(() => setPreset({}));   // non-fatal: backend applies it anyway
  }, [readyData?.session_id, selectedMode]);

  // Results are only reachable while they still describe the current settings.
  // Change the mode, a filter or a model and the old results stop being linkable
  // instead of silently sitting there looking current.
  const configKey = useMemo(
    () => JSON.stringify([readyData?.session_id, selectedMode, preprocessOverrides, classifierConfig]),
    [readyData?.session_id, selectedMode, preprocessOverrides, classifierConfig]
  );
  const resultsCurrent = !!results && resultsKey === configKey;

  const canLeave = [!!readyData, !!selectedMode, true, resultsCurrent];
  // furthest step reachable without skipping a required choice
  let reachable = 0;
  while (reachable < STEPS.length - 1 && canLeave[reachable]) reachable += 1;

  const goTo = (i) => {
    setDirection(i > step ? "forward" : "backward");
    setStep(i);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const goNext = () => goTo(Math.min(step + 1, STEPS.length - 1));
  const goBack = () => goTo(Math.max(step - 1, 0));

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await api.analyze({
        session_id: readyData.session_id,
        mode: selectedMode,
        preprocessing: preprocessOverrides,
        classifiers: classifierConfig,
      });
      setResults(data);
      setResultsKey(configKey);
      goTo(4);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Length of each trial the decoders will see, for the suitability notes in Step 4
  // (e.g. EEGNet on 30-s sleep windows). Pre-epoched data: total duration / trials.
  const epochSeconds = (() => {
    if (!readyData) return null;
    if (readyData.is_epochs) return readyData.n_epochs ? readyData.duration_seconds / readyData.n_epochs : null;
    const pick = (k) => (k in preprocessOverrides ? preprocessOverrides[k] : preset[k]);
    const crop = pick("epoch_train_crop");
    if (Array.isArray(crop) && crop.length === 2) return crop[1] - crop[0];
    const tmin = pick("epoch_tmin"), tmax = pick("epoch_tmax");
    return tmin != null && tmax != null ? tmax - tmin : null;
  })();

  const meta = STEPS[step];

  return (
    <div className="wz">
      <div className="wz-bg" aria-hidden="true"><i /><i /></div>

      <header className="wz-nav">
        <button className="wz-brand" onClick={onExit} title="Back to the home page">
          <Logo className="wz-logo" />NeuroScope
        </button>
        <Stepper steps={STEPS} current={step} reachable={reachable} onGo={goTo} />
        <button className="wz-exit" onClick={onExit}><Icon name="x" /><span>Exit</span></button>
      </header>

      <div className="wz-layout">
        <main className="wz-main">
          <div className="wz-head">
            <p className="wz-kicker">Step {step + 1} of {STEPS.length}</p>
            <h1>{meta.title}</h1>
            <p className="wz-sub">{meta.sub}</p>
          </div>

          {error && (
            <div className="wz-alert" role="alert">
              <Icon name="alert" />
              <span>{error}</span>
              <button className="up-link" onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}

          <div key={step} className={`wz-slide wz-slide-${direction}`}>
            {step === 0 && (
              <UploadSection
                loaded={readyData}
                onReady={(data) => {
                  setReadyData(data);
                  setSelectedMode(data.recommended_mode ?? null);
                  setResults(null);
                  setPreprocessOverrides({});
                }}
              />
            )}
            {step === 1 && (
              <ChooseSection data={readyData} selectedMode={selectedMode} onModeSelected={setSelectedMode} />
            )}
            {step === 2 && (
              <PreprocessSection
                mode={selectedMode}
                capabilities={readyData}
                preset={preset}
                overrides={preprocessOverrides}
                setOverrides={setPreprocessOverrides}
              />
            )}
            {step === 3 && (
              <ClassifySection
                capabilities={readyData}
                mode={selectedMode}
                config={classifierConfig}
                setConfig={setClassifierConfig}
                epochSeconds={epochSeconds}
              />
            )}
            {step === 4 && <ResultsSection results={results} datasetLabel={readyData?.display_name} />}
          </div>
        </main>

        <SummaryPanel
          data={readyData}
          mode={selectedMode}
          preset={preset}
          overrides={preprocessOverrides}
          classifiers={step >= 3 || results ? classifierConfig : null}
        />
      </div>

      <footer className="wz-footer">
        <div className="wz-footer-inner">
          <button className="wz-btn wz-btn-ghost" onClick={goBack} disabled={step === 0 || isAnalyzing}>
            <Icon name="arrowLeft" />Back
          </button>
          <span className="wz-footer-step">
            <b>{meta.name}</b>
            {step < STEPS.length - 1 && <> · next: {STEPS[step + 1].name}</>}
          </span>
          {step < 3 && (
            <button className="wz-btn" onClick={goNext} disabled={!canLeave[step]}>
              Continue<Icon name="arrowRight" />
            </button>
          )}
          {step === 3 && (
            <button className="wz-btn" onClick={resultsCurrent ? goNext : runAnalysis}
                    disabled={isAnalyzing || !classifierConfig.selected.length}>
              {isAnalyzing
                ? <><span className="wz-spinner" />Running…</>
                : resultsCurrent ? <>View results<Icon name="arrowRight" /></> : <>Run analysis<Icon name="bolt" /></>}
            </button>
          )}
          {step === 4 && <span />}
        </div>
      </footer>
    </div>
  );
}

export default Wizard;
