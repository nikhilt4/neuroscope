import Navbar from "../components/Navbar";
import ProcessDiagram from "../components/ProcessDiagram";

function LandingPage({ onStart }) {
  return (
    <div className="page">
      <Navbar />
      <main className="landing-content">
        <h1 className="landing-title">NeuroScope</h1>
        <p className="landing-subtitle">EEG Analysis & BCI Platform</p>

        <p className="landing-story">
          I built NeuroScope after working on EEG-based brain-computer interface
          projects and running into the same problem every time — every dataset
          comes in a different format, every pipeline has to be rebuilt from scratch,
          and there's no easy way to compare classifiers side by side. NeuroScope
          is the tool I wished existed: upload almost any EEG file, run a complete
          analysis pipeline, and compare results — all in the browser, no setup required.
        </p>

        <ProcessDiagram />

        <button className="cta-btn" onClick={onStart}>
          Start Analyzing →
        </button>
      </main>
    </div>
  );
}

export default LandingPage;