import { useEffect, useState } from "react";
import { Icon, Logo } from "../components/ui/Icons";
import "./Guide.css";

/**
 * How to use NeuroScope. Everything here describes what the app actually does today;
 * numbers quoted are this project's own verified runs (NEUROSCOPE_CONTEXT.md). If a
 * feature changes, change its section here too.
 */
const SECTIONS = [
  { id: "start", title: "Quick start" },
  { id: "data", title: "Loading data" },
  { id: "modes", title: "Choosing an analysis" },
  { id: "preprocess", title: "Preprocessing" },
  { id: "classify", title: "Decoders" },
  { id: "evaluation", title: "Evaluation" },
  { id: "results", title: "Reading the results" },
  { id: "playback", title: "Viewing a recording" },
  { id: "notes", title: "Good to know" },
];

function Term({ name, children }) {
  return (
    <div className="gd-term">
      <dt>{name}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function GuidePage({ onBack, onStart, backLabel = "Back" }) {
  const [active, setActive] = useState("start");

  // highlight the section currently in view in the table of contents
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && setActive(e.target.id)),
      { rootMargin: "-30% 0px -60% 0px" }
    );
    SECTIONS.forEach((s) => { const el = document.getElementById(s.id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  const go = (id) => (e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: "smooth" }); };

  return (
    <div className="gd">
      <div className="gd-bg" aria-hidden="true"><i /><i /></div>

      <header className="gd-nav">
        <button className="gd-brand" onClick={onBack}><Logo className="gd-logo" />NeuroScope</button>
        <span className="gd-nav-title">Guide</span>
        <button className="gd-back" onClick={onBack}><Icon name="arrowLeft" />{backLabel}</button>
        <button className="wz-btn gd-launch" onClick={onStart}>Try it free<Icon name="arrowRight" /></button>
      </header>

      <div className="gd-layout">
        <nav className="gd-toc" aria-label="Guide sections">
          <p>On this page</p>
          <ol>
            {SECTIONS.map((s) => (
              <li key={s.id}><a href={`#${s.id}`} onClick={go(s.id)} className={active === s.id ? "is-active" : ""}>{s.title}</a></li>
            ))}
          </ol>
        </nav>

        <main className="gd-main">
          <div className="gd-hero">
            <p className="gd-kicker">Guide</p>
            <h1>How to use NeuroScope</h1>
            <p>From loading a recording to reading the results, step by step. No coding needed.</p>
          </div>

          {/* ── quick start ── */}
          <section id="start" className="gd-section">
            <h2>Quick start</h2>
            <p>The app walks you through five steps. The fastest way to see it work:</p>
            <ol className="gd-steps">
              <li><b>Load data.</b> Choose <i>Use a public dataset → MOABB → BCI Competition IV 2a</i>, keep subject 1, and press <i>Fetch</i>.</li>
              <li><b>Analysis.</b> Keep the recommended mode (Motor imagery).</li>
              <li><b>Preprocess.</b> Nothing to change: the dataset arrives already prepared.</li>
              <li><b>Decoders.</b> Press <i>Recommended for this data</i>.</li>
              <li><b>Results.</b> Press <i>Run analysis</i>. In under a minute you&apos;ll see each decoder&apos;s accuracy (the best reaches about 83% here, against 25% chance).</li>
            </ol>
            <p className="gd-tip"><Icon name="info" /><span>The <b>Run summary</b> panel on the right always shows what your run will use: dataset, subject, mode, filters and decoders.</span></p>
          </section>

          {/* ── data ── */}
          <section id="data" className="gd-section">
            <h2>Loading data</h2>
            <h3>Public datasets</h3>
            <p>Pick a library (MNE, MOABB or braindecode), then a dataset card. Each card shows the task, number of subjects, channels, sampling rate and download size.</p>
            <dl className="gd-terms">
              <Term name="Subject">Which participant to analyse. Brain signals differ a lot between people, so decoders are usually trained per person, and accuracy varies widely (see <a href="#notes" onClick={go("notes")}>Good to know</a>).</Term>
              <Term name="Task">Some datasets recorded several tasks (e.g. PhysioNet Motor Imagery: left vs right hand, or hands vs feet). Pick the one to classify.</Term>
              <Term name="Fetch">The first fetch downloads the subject; later fetches load from the local cache in seconds. A progress line shows what&apos;s happening, and <i>Cancel</i> stops it. Downloads stop automatically after 10 minutes.</Term>
            </dl>
            <h3>Your own files</h3>
            <p>Drop files on the upload area or browse. NeuroScope works out what you&apos;ve selected:</p>
            <dl className="gd-terms">
              <Term name="One file">EDF, BDF, GDF or FIF. Events are read from the file&apos;s markers.</Term>
              <Term name="BrainVision (beta)">Select all three files together: <code>.vhdr</code>, <code>.vmrk</code> and <code>.eeg</code>.</Term>
              <Term name="BIDS dataset">One <code>.zip</code> containing <code>dataset_description.json</code> and <code>sub-XX/eeg/</code> folders. After upload you pick the subject, task, session and run.</Term>
              <Term name="CSV / TSV (beta)">You&apos;ll be asked which columns are EEG channels, time and labels, and the sampling rate.</Term>
              <Term name="MATLAB .mat">Not supported yet. Export to EDF or CSV for now.</Term>
            </dl>
            <p>After loading, the <b>Data loaded</b> card shows a head map of the channels found, how long the recording is, the classes that will be compared, and which analyses the data supports.</p>
          </section>

          {/* ── modes ── */}
          <section id="modes" className="gd-section">
            <h2>Choosing an analysis</h2>
            <p>The mode decides which <b>features</b> the classical decoders learn from. NeuroScope recommends one from the dataset and channels; a mode that can&apos;t run on your data is greyed out with the reason.</p>
            <dl className="gd-terms">
              <Term name="Motor imagery (CSP)">Common Spatial Patterns learn where on the scalp the classes differ and measure power there. The standard choice for imagined movements. Needs event markers and at least two EEG channels; recommended when C3 and C4 are present.</Term>
              <Term name="Spectral (band power)">How strong each brain rhythm is on every channel: delta (1–4 Hz), theta (4–8), alpha (8–13), beta (13–30) and gamma (30–45). Good for sleep staging, resting state and steady-state flicker (SSVEP).</Term>
              <Term name="ERP (time-domain)">The voltage waveform after each event, averaged in 50 ms steps. For evoked responses like the P300, N400 or error-related negativity.</Term>
              <Term name="Basic (statistics)">Mean, spread, variance, minimum, maximum and peak-to-peak per channel. A quick first look at unknown data.</Term>
            </dl>
            <p className="gd-tip"><Icon name="info" /><span>Riemannian and deep-learning decoders read the raw trials themselves, so the mode only changes the <b>classical</b> decoders.</span></p>
          </section>

          {/* ── preprocess ── */}
          <section id="preprocess" className="gd-section">
            <h2>Preprocessing</h2>
            <p>Every value starts from the dataset&apos;s <b>preset</b>, taken from its published analysis pipeline where one exists (generic defaults otherwise). Change anything and it&apos;s highlighted; <i>Reset</i> restores the preset. Datasets that arrive already filtered and cut into trials skip this step.</p>
            <dl className="gd-terms">
              <Term name="Bandpass filter">Keeps frequencies between <i>Low</i> and <i>High</i>. Low removes slow drift; high removes muscle noise. Leave <i>High</i> empty for a high-pass only.</Term>
              <Term name="Notch filter">Removes mains hum from the power supply: 50 Hz in most of the world, 60 Hz in North America and some other countries.</Term>
              <Term name="Epoching (tmin, tmax)">The window cut around each event, in seconds. Negative tmin starts before the event.</Term>
              <Term name="Analysis window">The part of each epoch the decoders actually see. Some presets epoch wide and analyse narrower (PhysioNet Motor Imagery: epoch −1 to 4 s, analyse 1 to 2 s, skipping the visual response to the cue). <i>Use the whole epoch</i> turns it off.</Term>
              <Term name="Baseline correction">Subtracts each epoch&apos;s average over a pre-event window, removing slow offsets.</Term>
              <Term name="ICA artifact removal">Removes eye-blink components automatically. Slow, and needs an eye (EOG) channel.</Term>
              <Term name="Re-referencing">What each channel is measured against: the average of all channels, a single channel (Cz), unchanged, or a preset pair such as P9 + P10.</Term>
            </dl>
            <p className="gd-tip"><Icon name="info" /><span>Sleep recordings are split into 30-second windows, one trial each, the standard unit for sleep scoring.</span></p>
          </section>

          {/* ── classify ── */}
          <section id="classify" className="gd-section">
            <h2>Decoders</h2>
            <p>Tick the decoders to compare, or use <b>Quick pick</b>: <i>Recommended for this data</i> picks a set that has worked well on similar data (in ERP mode it also switches LDA to shrinkage). <i>All classical</i> and <i>Clear</i> do what they say.</p>
            <h3>Three families</h3>
            <dl className="gd-terms">
              <Term name="Classical">LDA, SVM, Random Forest, logistic regression, Extra Trees, gradient boosting, MLP, k-nearest neighbours and Naive Bayes, trained on the mode&apos;s features.</Term>
              <Term name="Riemannian">Tangent space, MDM, FgMDM and xDAWN + tangent space. Each trial&apos;s covariance matrix is treated as a point on a curved surface and classified there. Often the strongest on small motor-imagery datasets. xDAWN is meant for evoked responses (ERP).</Term>
              <Term name="Deep learning">EEGNet, a compact convolutional network built for EEG. The slowest by far; training is capped at 50 passes to keep runs reasonable.</Term>
            </dl>
            <h3>Options</h3>
            <p>Click <i>Options</i> on a selected decoder to change its settings. The defaults are sensible starting points:</p>
            <dl className="gd-terms">
              <Term name="LDA">Solver (svd, lsqr, eigen) and shrinkage. Shrinkage only works with lsqr or eigen, and helps when there are many features (ERP mode).</Term>
              <Term name="SVM">Kernel (rbf, linear, poly), C (how strictly it fits the training data) and gamma.</Term>
              <Term name="Trees and boosting">Number of trees, maximum depth, features per split; learning rate and iterations for gradient boosting.</Term>
              <Term name="Riemannian">Covariance estimator: oas and lwf are shrinkage estimators that stay stable on short trials; scm is the plain estimate.</Term>
              <Term name="EEGNet">Training passes, learning rate, batch size and dropout.</Term>
            </dl>
            <p>An amber note on a decoder means it probably won&apos;t suit your data (for example, covariance-based decoders with only two EEG channels, or EEGNet on 30-second windows). A decoder that can&apos;t run at all is disabled with the reason.</p>
          </section>

          {/* ── evaluation ── */}
          <section id="evaluation" className="gd-section">
            <h2>Evaluation</h2>
            <p>Every decoder is scored the same way, so the comparison is fair. <b>Cross-validation</b> splits the trials into folds (5 by default), trains on all but one and tests on the one left out, in turn. Spatial filters and scalers are re-fitted inside every fold, so test trials never leak into training.</p>
            <p>For public datasets with several subjects you can also choose:</p>
            <dl className="gd-terms">
              <Term name="Single subject">Train and test on the subject you loaded.</Term>
              <Term name="Within-subject, across a group">Each chosen subject gets their own model, and the result is the mean ± spread across subjects, as BCI papers report it.</Term>
              <Term name="Cross-subject (leave one out)">Train on everyone else, test on the person left out. It asks whether a decoder works on someone new without calibrating to them first. Expect much lower accuracy.</Term>
            </dl>
            <p>Analyses run in the background: the footer shows which decoder is training, and <i>Cancel</i> stops it.</p>
          </section>

          {/* ── results ── */}
          <section id="results" className="gd-section">
            <h2>Reading the results</h2>
            <dl className="gd-terms">
              <Term name="Best accuracy">The share of test trials the best decoder got right, averaged over folds.</Term>
              <Term name="Fold spread (±)">How much accuracy varied between folds, or between subjects in multi-subject runs.</Term>
              <Term name="Chance level">What guessing would score: the share of the biggest class (25% for four balanced classes).</Term>
              <Term name="Balanced accuracy">The average of each class&apos;s hit rate. When classes are imbalanced (one class much larger than the others), plain accuracy misleads: always guessing the big class looks good while detecting nothing. NeuroScope then ranks decoders by balanced accuracy, whose chance level is 1 / number of classes.</Term>
              <Term name="Macro F1">Combines precision and recall for every class equally. Another check that small classes aren&apos;t being ignored.</Term>
              <Term name="Accuracy bars">One bar per decoder, coloured by family, with the dashed chance line. Click a bar to show that decoder&apos;s confusion matrix.</Term>
              <Term name="Confusion matrix">Rows are the true class, columns the prediction, as a percentage of each row. Green on the diagonal is correct; pink cells are common mistakes. The line underneath says which class is recognised best and what gets confused most.</Term>
              <Term name="Accuracy by subject">(multi-subject runs) Decoder × subject grid. A pale column is a person every decoder struggles with.</Term>
              <Term name="Scalp maps">Power in one frequency band per class, relative to the average. Blue is less power than average, red more, so the pattern shows where classes differ. For imagined hand movements, the hemisphere opposite the hand turns blue.</Term>
              <Term name="Power spectrum">Average power at each frequency per class. Where curves separate, the classes differ in frequency content.</Term>
              <Term name="Reproduce this run">A Python snippet with the preprocessing that actually ran and every score (per subject in multi-subject runs). Copy it or download it as a <code>.py</code> file.</Term>
            </dl>
            <p className="gd-tip"><Icon name="info" /><span>Changing a setting after a run makes the old results unreachable until you run again, so you never read results that don&apos;t match your settings.</span></p>
          </section>

          {/* ── playback ── */}
          <section id="playback" className="gd-section">
            <h2>Viewing a recording</h2>
            <p>After loading a continuous recording, from a public dataset or your own file, press <b>View recording</b> on the Data loaded card. It opens as a page of EEG, laid out like a clinical EEG reader.</p>
            <dl className="gd-terms">
              <Term name="The page">Every channel stacked, coloured by side of the head (left blue, right red, midline dark; eye and heart channels in their own colours), with a line every second.</Term>
              <Term name="Montage">Referential (as recorded), average reference, or bipolar &quot;double banana&quot; (Fp1-F3, F3-C3…) when the standard electrodes are present.</Term>
              <Term name="Moving through it">Scroll or drag the page, use ← → (1 s) and Shift+← → (one page), the buttons under the page, or the position bar. Page length is 5 to 30 s.</Term>
              <Term name="Events">Drawn on the page where they happen (e.g. <i>left hand</i>, or a sleep stage). Click any event in the list, or use Previous / Next, to jump straight to it. Click a label to list only that kind.</Term>
              <Term name="Play">Optional: scrolls the page automatically at 1× to 10×. Space starts and stops it.</Term>
            </dl>
            <p>Datasets that arrive already cut into trials (the MOABB versions, LIMO, kiloword) can&apos;t be viewed this way. For BCI Competition IV 2a, use the braindecode version, which is continuous.</p>
          </section>

          {/* ── notes ── */}
          <section id="notes" className="gd-section">
            <h2>Good to know</h2>
            <ul className="gd-notes">
              <li><b>People differ more than methods do.</b> On BCI Competition IV 2a, the same pipeline scores about 83% on subject 1 and about 50% on subject 5. Always compare runs on the same subject, or use a within-subject evaluation.</li>
              <li><b>Cross-subject is hard.</b> Without adjusting for each person, decoders transfer poorly: about 30% on three subjects of BCI IV 2a and about 38% on eight, against 25% chance.</li>
              <li><b>Small differences aren&apos;t meaningful.</b> With few trials per fold, one trial can move accuracy by several points. Gaps smaller than the ± spread are noise.</li>
              <li><b>Some labels are exploratory.</b> kiloword is split into abstract and concrete words at the median rating, a choice made by this app rather than the dataset&apos;s authors. LIMO&apos;s face A vs face B scores at chance in every mode.</li>
              <li><b>Long jobs time out.</b> Fetches stop after 10 minutes, analyses after 30 (60 for multi-subject runs).</li>
              <li><b>Uploaded files</b> are analysed as a single recording; multi-subject evaluation is for public datasets. You can rename an upload&apos;s classes (e.g. <i>Stimulus/S 1</i> to <i>target</i>) in the Preprocess step.</li>
            </ul>
            <div className="gd-cta">
              <h3>Ready to try it?</h3>
              <button className="wz-btn" onClick={onStart}>Try it free<Icon name="arrowRight" /></button>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

export default GuidePage;
