// import { useState } from "react";
// import ProgressBar from "../components/ProgressBar";
// import UploadSection from "../components/UploadSection";
// import ChooseSection from "../components/ChooseSection";
// import PreprocessSection from "../components/PreprocessSection";
// import ClassifySection from "../components/ClassifySection";
// import ResultsSection from "../components/ResultsSection";

// const STEP_NAMES = ["Upload", "Choose", "Pre-Process", "Classify", "Results"];

// const DEFAULT_PREPROCESS_CONFIG = {
//   bandpass_low: 1,
//   bandpass_high: 40,
//   notch_enabled: true,
//   notch_freq: "50",
//   epoch_tmin: -0.2,
//   epoch_tmax: 0.8,
//   baseline_enabled: true,
//   ica_enabled: false,
//   ica_components: 20,
//   reference: "average",
// };

// function Wizard({ onExit }) {
//   const [step, setStep] = useState(0);
//   const [direction, setDirection] = useState("forward");

//   const [readyData, setReadyData] = useState(null);
//   const [selectedMode, setSelectedMode] = useState(null);
//   const [preprocessConfig, setPreprocessConfig] = useState(DEFAULT_PREPROCESS_CONFIG);
//   const [results, setResults] = useState(null);

//   const goNext = () => {
//     setDirection("forward");
//     setStep((s) => Math.min(s + 1, STEP_NAMES.length - 1));
//   };

//   const goBack = () => {
//     setDirection("backward");
//     setStep((s) => Math.max(s - 1, 0));
//   };

//   const canGoNext = [
//     !!readyData,          // step 0 -> 1: needs upload done
//     !!selectedMode,        // step 1 -> 2: needs mode selected
//     true,                   // step 2 -> 3: preprocess always advances via its own button
//     !!results,              // step 3 -> 4: needs results
//   ];

//   return (
//     <div className="wizard">
//       <div className="wizard-header">
//         <button className="exit-btn" onClick={onExit}>← Exit</button>
//         <ProgressBar steps={STEP_NAMES} current={step} />
//       </div>

//       <div className="wizard-frame">
//         <div key={step} className={`wizard-slide slide-${direction}`}>
//           {step === 0 && (
//             <UploadSection
//               onReady={(data) => {
//                 setReadyData(data);
//                 goNext();
//               }}
//             />
//           )}

//           {step === 1 && (
//             <ChooseSection
//               data={readyData}
//               selectedMode={selectedMode}
//               onModeSelected={(mode) => {
//                 setSelectedMode(mode);
//               }}
//             />
//           )}

//           {step === 2 && (
//             <PreprocessSection
//               mode={selectedMode}
//               config={preprocessConfig}
//               setConfig={setPreprocessConfig}
//               onConfigured={goNext}
//             />
//           )}

//           {step === 3 && (
//             <ClassifySection
//               enabled={true}
//               onRun={(mockResults) => {
//                 setResults(mockResults);
//                 goNext();
//               }}
//             />
//           )}

//           {step === 4 && (
//             <ResultsSection
//               results={results}
//               preprocessConfig={preprocessConfig}
//             />
//           )}
//         </div>
//       </div>

//       <div className="wizard-nav">
//         <button className="secondary-btn" onClick={goBack} disabled={step === 0}>
//           ← Back
//         </button>
//         {step < 4 && (
//           <button
//             className="primary-btn"
//             onClick={goNext}
//             disabled={!canGoNext[step]}
//           >
//             Next →
//           </button>
//         )}
//       </div>
//     </div>
//   );
// }

// export default Wizard;




import { useState, useEffect } from "react";
import ProgressBar from "../components/ProgressBar";
import UploadSection from "../components/UploadSection";
import ChooseSection from "../components/ChooseSection";
import PreprocessSection from "../components/PreprocessSection";
import ClassifySection from "../components/ClassifySection";
import ResultsSection from "../components/ResultsSection";
import * as api from "../api";

const STEP_NAMES = ["Upload", "Choose", "Pre-Process", "Classify", "Results"];

function Wizard({ onExit }) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState("forward");

  const [readyData, setReadyData] = useState(null);
  const [selectedMode, setSelectedMode] = useState(null);

  // Preset = read-only defaults from the backend, shown as placeholders.
  // preprocessOverrides = ONLY what the user actually changed. Sending a full
  // config would override the dataset's researched preset with generic values.
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!readyData || !selectedMode) return;
    api
      .getPreset({ datasetName: readyData.dataset_name, mode: selectedMode })
      .then(setPreset)
      .catch(() => setPreset({}));   // non-fatal: backend applies it anyway
  }, [readyData?.session_id, selectedMode]);

  const goNext = () => {
    setDirection("forward");
    setStep((s) => Math.min(s + 1, STEP_NAMES.length - 1));
  };

  const goBack = () => {
    setDirection("backward");
    setStep((s) => Math.max(s - 1, 0));
  };

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
      goNext();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const canGoNext = [
    !!readyData,
    !!selectedMode,
    true,
    !!results,
  ];

  return (
    <div className="wizard">
      <div className="wizard-header">
        <button className="exit-btn" onClick={onExit}>← Exit</button>
        <ProgressBar steps={STEP_NAMES} current={step} />
      </div>

      {error && (
        <div className="error-box">
          {error}
          <button className="link-btn" onClick={() => setError(null)}>dismiss</button>
        </div>
      )}

      <div className="wizard-frame">
        <div key={step} className={`wizard-slide slide-${direction}`}>
          {step === 0 && (
            <UploadSection
              onReady={(data) => {
                setReadyData(data);
                setSelectedMode(data.recommended_mode ?? null);
                setResults(null);
                setPreprocessOverrides({});
                goNext();
              }}
            />
          )}

          {step === 1 && (
            <ChooseSection
              data={readyData}
              selectedMode={selectedMode}
              onModeSelected={setSelectedMode}
            />
          )}

          {step === 2 && (
            <PreprocessSection
              mode={selectedMode}
              capabilities={readyData}
              preset={preset}
              overrides={preprocessOverrides}
              setOverrides={setPreprocessOverrides}
              onConfigured={goNext}
            />
          )}

          {step === 3 && (
            <ClassifySection
              capabilities={readyData}
              mode={selectedMode}
              config={classifierConfig}
              setConfig={setClassifierConfig}
              onRun={runAnalysis}
              isRunning={isAnalyzing}
            />
          )}

          {step === 4 && (
            <ResultsSection results={results} />
          )}
        </div>
      </div>

      <div className="wizard-nav">
        <button className="secondary-btn" onClick={goBack} disabled={step === 0 || isAnalyzing}>
          ← Back
        </button>
        {step < 4 && (
          <button className="primary-btn" onClick={goNext} disabled={!canGoNext[step]}>
            Next →
          </button>
        )}
      </div>
    </div>
  );
}

export default Wizard;