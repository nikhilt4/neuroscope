// import { useState } from "react";
// import Navbar from "./components/Navbar";
// import Section from "./components/Section";
// import UploadSection from "./components/UploadSection";
// import ChooseSection from "./components/ChooseSection";
// import PreprocessSection from "./components/PreprocessSection";
// import ClassifySection from "./components/ClassifySection";
// import ResultsSection from "./components/ResultsSection";
// import { MOCK_UPLOAD_DATA, MOCK_RESULTS } from "./mockData";
// import "./App.css";

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

// function App() {
//   const [previewMode, setPreviewMode] = useState(false);

//   const [readyData, setReadyData] = useState(null);
//   const [selectedMode, setSelectedMode] = useState(null);
//   const [preprocessConfig, setPreprocessConfig] = useState(DEFAULT_PREPROCESS_CONFIG);
//   const [preprocessDone, setPreprocessDone] = useState(false);
//   const [results, setResults] = useState(null);

//   const togglePreview = () => {
//     if (previewMode) {
//       setReadyData(null);
//       setSelectedMode(null);
//       setPreprocessDone(false);
//       setResults(null);
//     } else {
//       setReadyData(MOCK_UPLOAD_DATA);
//       setSelectedMode(MOCK_UPLOAD_DATA.recommended_mode);
//       setPreprocessDone(true);
//       setResults(MOCK_RESULTS);
//     }
//     setPreviewMode(!previewMode);
//   };

//   return (
//     <div className="page">
//       <Navbar />

//       <main className="content">
//         <h1 className="page-title">NeuroScope</h1>
//         <p className="page-subtitle">EEG Analysis & BCI Platform</p>

//         <p className="page-description">
//           Upload EEG data in almost any format and run a full analysis pipeline —
//           preprocessing, feature extraction, and classification — right in your browser.
//           Need more detail? Check the <a href="#info">Info</a> page for a full walkthrough.
//         </p>

//         <button className="preview-toggle" onClick={togglePreview}>
//           {previewMode ? "Exit Preview" : "Preview All Sections (Demo Data)"}
//         </button>

//         <Section number="1" title="Upload">
//           {previewMode ? (
//             <p className="placeholder-text">Preview mode — showing demo upload result.</p>
//           ) : (
//             <UploadSection onReady={setReadyData} />
//           )}
//         </Section>

//         <Section number="2" title="Choose" disabled={!readyData}>
//           <ChooseSection
//             data={readyData}
//             selectedMode={selectedMode}
//             onModeSelected={setSelectedMode}
//           />
//         </Section>

//         <Section number="3" title="Pre-Process" disabled={!selectedMode}>
//           <PreprocessSection
//             mode={selectedMode}
//             config={preprocessConfig}
//             setConfig={setPreprocessConfig}
//             onConfigured={() => setPreprocessDone(true)}
//           />
//         </Section>

//         <Section number="4" title="Classify" disabled={!preprocessDone}>
//           <ClassifySection
//             enabled={preprocessDone}
//             onRun={() => setResults(MOCK_RESULTS)}
//           />
//         </Section>

//         <Section number="5" title="Results" disabled={!results}>
//           <ResultsSection results={results} />
//         </Section>
//       </main>
//     </div>
//   );
// }

// export default App;


import { useState } from "react";
import LandingPage from "./pages/LandingPage";
import Wizard from "./pages/Wizard";
import "./App.css";

function App() {
  const [started, setStarted] = useState(false);

  return started ? (
    <Wizard onExit={() => setStarted(false)} />
  ) : (
    <LandingPage onStart={() => setStarted(true)} />
  );
}

export default App;