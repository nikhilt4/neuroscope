import { useState } from "react";
import ProgressBar from "../components/ProgressBar";
import UploadSection from "../components/UploadSection";
import ChooseSection from "../components/ChooseSection";
import PreprocessSection from "../components/PreprocessSection";
import ClassifySection from "../components/ClassifySection";
import ResultsSection from "../components/ResultsSection";

const STEP_NAMES = ["Upload", "Choose", "Pre-Process", "Classify", "Results"];

const DEFAULT_PREPROCESS_CONFIG = {
  bandpass_low: 1,
  bandpass_high: 40,
  notch_enabled: true,
  notch_freq: "50",
  epoch_tmin: -0.2,
  epoch_tmax: 0.8,
  baseline_enabled: true,
  ica_enabled: false,
  ica_components: 20,
  reference: "average",
};

function Wizard({ onExit }) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState("forward");

  const [readyData, setReadyData] = useState(null);
  const [selectedMode, setSelectedMode] = useState(null);
  const [preprocessConfig, setPreprocessConfig] = useState(DEFAULT_PREPROCESS_CONFIG);
  const [results, setResults] = useState(null);

  const goNext = () => {
    setDirection("forward");
    setStep((s) => Math.min(s + 1, STEP_NAMES.length - 1));
  };

  const goBack = () => {
    setDirection("backward");
    setStep((s) => Math.max(s - 1, 0));
  };

  const canGoNext = [
    !!readyData,          // step 0 -> 1: needs upload done
    !!selectedMode,        // step 1 -> 2: needs mode selected
    true,                   // step 2 -> 3: preprocess always advances via its own button
    !!results,              // step 3 -> 4: needs results
  ];

  return (
    <div className="wizard">
      <div className="wizard-header">
        <button className="exit-btn" onClick={onExit}>← Exit</button>
        <ProgressBar steps={STEP_NAMES} current={step} />
      </div>

      <div className="wizard-frame">
        <div key={step} className={`wizard-slide slide-${direction}`}>
          {step === 0 && (
            <UploadSection
              onReady={(data) => {
                setReadyData(data);
                goNext();
              }}
            />
          )}

          {step === 1 && (
            <ChooseSection
              data={readyData}
              selectedMode={selectedMode}
              onModeSelected={(mode) => {
                setSelectedMode(mode);
              }}
            />
          )}

          {step === 2 && (
            <PreprocessSection
              mode={selectedMode}
              config={preprocessConfig}
              setConfig={setPreprocessConfig}
              onConfigured={goNext}
            />
          )}

          {step === 3 && (
            <ClassifySection
              enabled={true}
              onRun={(mockResults) => {
                setResults(mockResults);
                goNext();
              }}
            />
          )}

          {step === 4 && (
            <ResultsSection
              results={results}
              preprocessConfig={preprocessConfig}
            />
          )}
        </div>
      </div>

      <div className="wizard-nav">
        <button className="secondary-btn" onClick={goBack} disabled={step === 0}>
          ← Back
        </button>
        {step < 4 && (
          <button
            className="primary-btn"
            onClick={goNext}
            disabled={!canGoNext[step]}
          >
            Next →
          </button>
        )}
      </div>
    </div>
  );
}

export default Wizard;