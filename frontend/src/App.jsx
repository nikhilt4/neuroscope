import { useState } from "react";
import LandingPage from "./pages/LandingPage";
import Wizard from "./pages/Wizard";
import GuidePage from "./pages/GuidePage";
import "./App.css";

/**
 * Three views: landing, the app (wizard) and the guide. The wizard stays MOUNTED (just
 * hidden) while the guide is open, so opening the guide mid-analysis doesn't wipe the
 * loaded data, settings or results.
 */
function App() {
  const [started, setStarted] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);

  const openGuide = () => { setGuideOpen(true); window.scrollTo(0, 0); };

  return (
    <>
      {started && (
        <div hidden={guideOpen}>
          <Wizard onExit={() => setStarted(false)} onGuide={openGuide} />
        </div>
      )}
      {!started && !guideOpen && (
        <LandingPage onStart={() => setStarted(true)} onGuide={openGuide} />
      )}
      {guideOpen && (
        <GuidePage
          backLabel={started ? "Back to the app" : "Back to home"}
          onBack={() => setGuideOpen(false)}
          onStart={() => { setGuideOpen(false); setStarted(true); }}
        />
      )}
    </>
  );
}

export default App;
