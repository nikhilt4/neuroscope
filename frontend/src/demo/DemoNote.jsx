import { DEMO, REPO_URL } from "./config";
import { Icon } from "../components/ui/Icons";

// What the demo can and can't do at each wizard step. Said up front, so a visitor never
// mistakes a recorded result for one their edits produced.
const STEP_TEXT = {
  0: "Online demo: two datasets are included, one subject each, BCI Competition IV 2a and Sleep-EDF. Everything you see was recorded from the real backend. Uploading your own files needs the full app.",
  2: "In the demo, results were recorded with these preset values. You can change them to explore the options, but the recorded results won't change.",
  3: "Running replays the real backend's recorded results for the decoders you pick. Hyperparameter edits aren't re-trained. Multi-subject results are recorded for BCI IV 2a in motor imagery mode, with all 9 subjects.",
};

function DemoNote({ step, results }) {
  if (!DEMO) return null;
  let text = STEP_TEXT[step];
  if (step === 4 && results?.demo) {
    const { recorded, missing } = results.demo;
    text = `Recorded by the real backend on ${recorded}, with each dataset's preset preprocessing.` +
      (missing?.length ? ` Not in this recording: ${missing.join(", ")}.` : "");
  }
  if (!text) return null;
  return (
    <div className="wz-alert wz-alert-info demo-note" role="note">
      <Icon name="info" />
      <span>
        {text}
        {REPO_URL && <> <a href={REPO_URL} target="_blank" rel="noreferrer">Get the full app</a>.</>}
      </span>
    </div>
  );
}

export default DemoNote;
