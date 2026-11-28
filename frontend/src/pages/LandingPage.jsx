import { useEffect, useRef } from "react";
import EegTraces from "../components/landing/EegTraces";
import HeadMap from "../components/landing/HeadMap";
import ResultsPreview from "../components/landing/ResultsPreview";
import { Icon, Logo as LogoMark } from "../components/ui/Icons";
import "./LandingPage.css";
import { DEMO } from "../demo/config";

const Svg = ({ name }) => <Icon name={name} />;
const Logo = () => <LogoMark className="lp-logo" />;

// "How it works", not buttons: a tester kept clicking "Upload" here expecting it to start the app
const STEPS = [
  { icon: "database", title: "Load data", desc: "A public dataset or your own file" },
  { icon: "detect", title: "Check it", desc: "Channels and events found for you" },
  { icon: "filter", title: "Clean it", desc: "Filters set from published methods" },
  { icon: "classify", title: "Train models", desc: "Up to 14, all tested the same way" },
  { icon: "results", title: "Compare", desc: "Accuracy, charts and scalp maps" },
];

const FEATURES = [
  { icon: "browser", title: "Free, in your browser", desc: "No install, no code" },
  { icon: "wave", title: "Data to results in minutes", desc: "Load, clean, train, compare" },
  { icon: "layers", title: "14 models, 3 families", desc: "Classical, Riemannian, deep" },
  { icon: "shield", title: "Results you can trust", desc: "Cross-validation that can’t cheat" },
];

// What EEG can actually be used for, and which NeuroScope mode analyses it. Kept to
// well-established BCI/clinical uses so the claims hold up for people in the field.
const USES = [
  { icon: "hand", tone: "blue", mode: "Motor imagery mode", title: "Imagine moving your hand",
    body: "The 8–13 Hz mu rhythm drops over the opposite side of the motor cortex, even when you don’t move. BCIs use it to steer a cursor, a wheelchair or a game." },
  { icon: "bolt", tone: "blue", mode: "Motor imagery mode", title: "Clench your fist",
    body: "Real movements change the same motor rhythms. That’s the basis of controlling a robotic hand, and of brain-driven rehab after a stroke." },
  { icon: "keyboard", tone: "violet", mode: "ERP mode", title: "Focus on a flashing letter",
    body: "A P300 wave appears about 300 ms after the letter you’re waiting for. Spellers use it to let people type without moving a muscle." },
  { icon: "eye", tone: "violet", mode: "Spectral mode", title: "Look at a flickering light",
    body: "Your visual cortex echoes the flicker’s frequency. Look at one of several lights to pick an option, quickly and with almost no training." },
  { icon: "moon", tone: "pink", mode: "Spectral mode · playback", title: "Fall asleep",
    body: "Each sleep stage has its own signature, from alpha as you drift off to slow delta waves in deep sleep. Clinics score whole nights from it." },
  { icon: "car", tone: "pink", mode: "Spectral mode", title: "Start to nod off at the wheel",
    body: "Rising theta and alpha power flag drowsiness before it turns dangerous. NeuroScope’s creator built a headband that detects it on the chip." },
];

const FAMILIES = [
  {
    tag: "Classical", tone: "blue",
    title: "Features + a proven classifier",
    body: "CSP spatial filters, band power or signal statistics, fed to a standard model.",
    members: ["LDA", "SVM", "Random Forest", "Logistic regression", "Extra Trees", "Gradient boosting", "MLP", "kNN", "Naive Bayes"],
  },
  {
    tag: "Riemannian geometry", tone: "violet",
    title: "Covariance on a curved manifold",
    body: "Each trial’s covariance matrix becomes a point on a manifold and is classified there. Often the strongest choice on small BCI datasets.",
    members: ["Tangent space", "MDM", "FgMDM", "xDAWN + tangent space"],
  },
  {
    tag: "Deep learning", tone: "pink",
    title: "A compact CNN built for EEG",
    body: "EEGNet: about 3,400 parameters, trained on your GPU through braindecode. More architectures are planned.",
    members: ["EEGNet"],
  },
];

const PROBLEMS = [
  { title: "Every dataset speaks a different format", body: "EDF, BDF, FIF, BrainVision, BIDS, MATLAB. Each one needs its own loader before you see a single trial." },
  { title: "Every project rebuilds the pipeline", body: "Filters, epoch windows, re-referencing and event codes, re-derived from a methods section every time." },
  { title: "Quick comparisons quietly cheat", body: "Fit CSP or a scaler on all the data before cross-validating and accuracy goes up for the wrong reason." },
];

const AUDIENCES = [
  { title: "Students learning BCI", body: "Change a filter or an epoch window and watch the accuracy move, with the reasoning shown next to every default." },
  { title: "Researchers benchmarking", body: "Put a new dataset up against 14 baselines in minutes, and export the exact parameters for your methods section." },
  { title: "Hardware & BCI builders", body: "Compare decoders on your own recordings before committing one to a microcontroller." },
];

const FORMATS = [
  { name: "EDF · BDF · GDF", status: "ready" },
  { name: "FIF (raw or epochs)", status: "ready" },
  { name: "BIDS (ZIP)", status: "ready" },
  { name: "MNE · MOABB · braindecode", status: "ready" },
  { name: "BrainVision", status: "beta" },
  { name: "CSV / TSV", status: "beta" },
  { name: "MATLAB .mat", status: "soon" },
];
const STATUS_LABEL = { ready: "Ready", beta: "Beta", soon: "Coming soon" };

/** Adds .is-in to every [data-reveal] element as it scrolls into view. */
function useReveal(rootRef) {
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); }
      }),
      { threshold: 0.12 }
    );
    rootRef.current.querySelectorAll("[data-reveal]").forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [rootRef]);
}


function LandingPage({ onStart, onGuide }) {
  const rootRef = useRef(null);
  useReveal(rootRef);
  const go = (id) => (e) => {
    e?.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="lp" ref={rootRef}>
      <div className="lp-bg" aria-hidden="true"><i /><i /><i /></div>

      <header className="lp-nav">
        <a className="lp-brand" href="#top" onClick={go("top")}><Logo />NeuroScope</a>
        <nav className="lp-nav-links">
          <a href="#uses" onClick={go("uses")}>Uses</a>
          <a href="#preview" onClick={go("preview")}>Results</a>
          <a href="#models" onClick={go("models")}>Models</a>
          <a href="#why" onClick={go("why")}>Why</a>
          <a href="#formats" onClick={go("formats")}>Formats</a>
        </nav>
        {/* a button, not a nav link: the nav links are hidden on phones, the guide shouldn't be */}
        <button className="lp-btn lp-btn-ghost lp-btn-sm lp-guide-btn" onClick={onGuide}>Guide</button>
        {/* "Try it free", not "Launch app": it opens in the browser, nothing installs */}
        <button className="lp-btn lp-btn-sm" onClick={onStart}>Try it free</button>
      </header>

      {/* ── hero ─────────────────────────────────────────────── */}
      <section className="lp-hero" id="top">
        <HeadMap className="lp-head" />
        <EegTraces className="lp-waves" channels={5} />

        <div className="lp-hero-text">
          <p className="lp-eyebrow"><span className="lp-dot" />Free EEG analysis &amp; BCI platform{DEMO && <> · online demo</>}</p>
          <h1 className="lp-title">NeuroScope</h1>
          <p className="lp-tagline">
            Analyse brainwave data and train machine-learning models on it.<br />
            <span className="lp-grad">Free, in your browser, no code.</span>
          </p>
          <p className="lp-lede">
            Pick a public EEG dataset or upload a recording. NeuroScope cleans it, trains up to 14 models, and
            shows which one best tells the brain states apart, such as imagined left vs right hand movements.
          </p>
        </div>

        <div className="lp-ctas">
          <button className="lp-btn lp-btn-lg" onClick={onStart}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>
            Start analysing
          </button>
          <a className="lp-btn lp-btn-ghost lp-btn-lg" href="#preview" onClick={go("preview")}>See real results</a>
        </div>

        <p className="lp-steps-label">How it works</p>

        <ol className="lp-steps">
          {STEPS.map((s, i) => (
            <li key={s.title} style={{ "--d": `${i * 70}ms` }}>
              <span className="lp-step-icon"><Svg name={s.icon} /></span>
              <b><span className="lp-step-num">{i + 1}</span>{s.title}</b>
              <span>{s.desc}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="lp-features">
        {FEATURES.map((f) => (
          <div key={f.title}>
            <span className="lp-feature-icon"><Svg name={f.icon} /></span>
            <div><b>{f.title}</b><span>{f.desc}</span></div>
          </div>
        ))}
      </section>

      {/* ── what brain signals can do ────────────────────────── */}
      <section className="lp-section" id="uses">
        <div className="lp-head-block" data-reveal>
          <p className="lp-kicker">What brain signals can do</p>
          <h2 className="lp-h2">Think it, and a computer can read it.</h2>
          <p className="lp-sub">
            Thoughts and actions leave patterns in your EEG. Here’s what researchers and BCI builders already
            do with them, and the NeuroScope mode that analyses each one.
          </p>
        </div>
        <div className="lp-grid-3">
          {USES.map((u, i) => (
            <article className="lp-card lp-use" key={u.title} data-reveal style={{ "--d": `${(i % 3) * 80}ms` }}>
              <span className={`lp-use-icon lp-use-${u.tone}`}><Svg name={u.icon} /></span>
              <h3>{u.title}</h3>
              <p>{u.body}</p>
              <span className={`lp-tag lp-tag-${u.tone} lp-use-mode`}>{u.mode}</span>
            </article>
          ))}
        </div>
        <div className="lp-callout" data-reveal>
          <span className="lp-callout-icon"><Svg name="sparkle" /></span>
          <p>
            <b>Why machine learning?</b> EEG is a few microvolts at the scalp, buried under noise from muscles,
            blinks and mains power, and every brain is different. Hand-tuned thresholds break from one person to
            the next; machine learning learns each person’s pattern from a few minutes of examples. The catch is
            that it can quietly fool you, which is why NeuroScope compares models fairly and labels every result
            with the person it came from.
          </p>
        </div>
      </section>

      {/* ── results preview ──────────────────────────────────── */}
      <section className="lp-section" id="preview">
        <div className="lp-head-block" data-reveal>
          <p className="lp-kicker">A look at the results page</p>
          <h2 className="lp-h2">Real output, not a mock-up.</h2>
          <p className="lp-sub">
            Every number below comes from NeuroScope running on BCI Competition IV 2a, subject 1:
            four imagined movements, chance level 25%.
          </p>
        </div>
        <div data-reveal><ResultsPreview /></div>
        <p className="lp-caveat" data-reveal>
          <b>An honest caveat:</b> brains differ a lot. The same pipeline scores about 50% on subject 5 of this
          dataset, in line with published per-subject results of roughly 45–90%. That’s why every result in
          NeuroScope is labelled with the subject it came from.
        </p>
      </section>

      {/* ── models ───────────────────────────────────────────── */}
      <section className="lp-section" id="models">
        <div className="lp-head-block" data-reveal>
          <p className="lp-kicker">The models</p>
          <h2 className="lp-h2">Three schools of decoding, one leaderboard.</h2>
        </div>
        <div className="lp-grid-3">
          {FAMILIES.map((f, i) => (
            <article className="lp-card" key={f.tag} data-reveal style={{ "--d": `${i * 80}ms` }}>
              <span className={`lp-tag lp-tag-${f.tone}`}>{f.tag}</span>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
              <ul className="lp-chips">{f.members.map((m) => <li key={m}>{m}</li>)}</ul>
            </article>
          ))}
        </div>
        <div className="lp-callout" data-reveal>
          <span className="lp-callout-icon"><Svg name="shield" /></span>
          <p>
            <b>Leakage-safe by construction.</b> CSP filters and feature scalers sit <em>inside</em> the
            cross-validation pipeline, so each fold fits them on its own training trials only. Fitting them on
            all the data first is one of the most common ways EEG benchmarks overstate accuracy.
          </p>
        </div>
      </section>

      {/* ── why ──────────────────────────────────────────────── */}
      <section className="lp-section" id="why">
        <div className="lp-head-block" data-reveal>
          <p className="lp-kicker">Why it exists</p>
          <h2 className="lp-h2">EEG analysis shouldn’t start with plumbing.</h2>
        </div>
        <div className="lp-grid-3">
          {PROBLEMS.map((p, i) => (
            <article className="lp-card lp-problem" key={p.title} data-reveal style={{ "--d": `${i * 80}ms` }}>
              <span className="lp-num">0{i + 1}</span>
              <h3>{p.title}</h3>
              <p>{p.body}</p>
            </article>
          ))}
        </div>
        <blockquote className="lp-quote" data-reveal>
          <p>
            “I built two EEG brain–computer interfaces before this one, and both times I lost weeks to the same
            problems: reformatting data, rebuilding pipelines, and never being sure a comparison was fair.
            NeuroScope is the tool I wished I’d had.”
          </p>
          <footer>Nikhil Talwalkar, creator</footer>
        </blockquote>
      </section>

      {/* ── who + formats ────────────────────────────────────── */}
      <section className="lp-section">
        <div className="lp-head-block" data-reveal>
          <p className="lp-kicker">Who it’s for</p>
          <h2 className="lp-h2">Built for people who work with brain signals.</h2>
        </div>
        <div className="lp-grid-3">
          {AUDIENCES.map((a, i) => (
            <article className="lp-card" key={a.title} data-reveal style={{ "--d": `${i * 80}ms` }}>
              <h3>{a.title}</h3>
              <p>{a.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="lp-section" id="formats">
        <div className="lp-head-block" data-reveal>
          <p className="lp-kicker">Data in</p>
          <h2 className="lp-h2">Bring almost any EEG data.</h2>
        </div>
        <ul className="lp-formats" data-reveal>
          {FORMATS.map((f) => (
            <li key={f.name}>
              <span>{f.name}</span>
              <span className={`lp-status lp-status-${f.status}`}>{STATUS_LABEL[f.status]}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="lp-final" data-reveal>
        <EegTraces className="lp-final-waves" channels={3} color="#ffffff" speed={0.4} />
        <h2>Ready to look inside your data?</h2>
        <p>{DEMO
          ? "Explore two recorded datasets in the free online demo: the full wizard, real results and live playback."
          : "It’s free. Start with a public dataset, with nothing to download or install, or upload your own recording."}</p>
        <button className="lp-btn lp-btn-light lp-btn-lg" onClick={onStart}>Start analysing →</button>
      </section>

      <footer className="lp-footer">
        <span className="lp-brand"><Logo />NeuroScope</span>
        <span>Built on MNE-Python · scikit-learn · pyRiemann · braindecode</span>
      </footer>
    </div>
  );
}

export default LandingPage;
