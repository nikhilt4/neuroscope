# NeuroScope (EEG-ai1): Project Context & Handoff

> **Last updated: 13 November 2026** (Claude Code session: test suite added; section 5
> items 1, 3, 4 and four Correctness backlog items fixed — see "13 Nov changes" in section 4).
> Originally rewritten 12 November 2026 from the end of a long working session on
> claude.ai. This supersedes the July 2026 version, which was significantly out of date —
> it listed `routes/analyze.py` and `visualizer.py` as TODO when both now exist and work.
>
> Everything in sections 4–6 was verified against real runs during that session. Where
> something is unverified or untested, it says so explicitly. Still worth scanning the repo
> before trusting any individual claim.

---

## 0. How to work with me (Nikhil)

- **You write the code.** Implement features end to end. Give me a short plan for anything
  big, then build it. I review the diff.

- **SHOW ME WHERE YOU GOT IT FROM.** The most important rule, and this session proved why.
  - Back every library call, parameter, default and dataset fact with a source: the official
    docs page or tutorial, and the version it applies to. Check installed versions with
    `pip show <pkg>` first.
  - Put the source in a comment next to the call, e.g.
    `# https://mne.tools/stable/generated/mne.decoding.CSP.html (MNE 1.x)`.
  - Preferred sources in order: official API docs and tutorials (mne.tools,
    moabb.neurotechx.com, braindecode.org, scikit-learn.org, pyriemann.readthedocs.io),
    then the library's source on GitHub, then the original papers. Avoid blog posts.
  - **Verify library signatures with `inspect.signature()` before writing code against them.**
    This session caught four separate bugs that way — `eegbci.load_data`'s renamed first
    argument, `EEGClassifier` silently ignoring `lr=`, `erp_core.data_path` taking no task
    parameter, and `kiloword.data_path` returning a `Path` rather than a list.
  - If you can't find a source, say "unverified", mark it in the code, and add it to the
    backlog. Don't present a guess as fact.

- **Test on real data, and name the subject.** Accuracy on BNCI2014_001 ranges from ~45% to
  ~85% depending which subject you fetched. Several hours were lost this session comparing
  runs from different subjects and blaming the code. Always state which subject a number
  came from.

- **Defaults are automatic, but I can always override them.** Every parameter needs a
  sensible sourced default and a way to change it, in both the API and the UI.

- **Frontend and backend defaults must match exactly.** `PARAM_FIELDS` in
  `ClassifySection.jsx` holds a `def` for every field; the backend builder holds a
  `config.get(key, fallback)`. The frontend only sends values that *differ* from `def`, so
  any drift between the two means the UI displays one value while the backend runs another,
  silently. This caused the EEGNet learning-rate confusion.

- **Flag things for later out loud.** Add to the backlog (section 7) rather than dropping it.

- **Environment:** Windows, PowerShell, repo at `C:\Users\nikhi\Desktop\Projects\EEG-ai1`,
  Python 3.12 venv at `./venv`, NVIDIA RTX 4060.
  - Backend: `cd` to **repo root**, activate venv, `uvicorn backend.main:app --reload`.
    Running from inside `backend/` breaks the `backend.*` imports.
  - Frontend: `cd frontend`, `npm run dev` → http://localhost:5173
  - **Watch the interpreter.** VS Code has repeatedly launched uvicorn on system Python
    (`C:\Python312`) rather than the venv. Check with
    `python -c "import sys; print(sys.executable)"` before starting anything. Open the
    `EEG-ai1` folder as the VS Code workspace root, not the parent `Projects` folder.
  - `Get-Process python | Stop-Process -Force` is often needed — a stalled download inside
    a sync request ignores Ctrl+C entirely.

---

## 1. What the project is

**NeuroScope: EEG Analysis & BCI Platform.** A full-stack web app. Upload almost any EEG
file or pull a public dataset. The backend detects what analysis the data supports, runs a
configurable pipeline (preprocessing → feature extraction → classification), compares up to
fourteen classifiers side by side with leakage-safe cross-validation, and shows the results
with charts and scalp topographies.

**Origin story (landing page):** two hardware EEG BCI projects came first. Every time, the
same problems — every dataset in a different format, every pipeline rebuilt from scratch, no
easy way to compare classifiers. NeuroScope is the tool I wished I'd had.

**Stack:** FastAPI · MNE-Python · MNE-BIDS · MOABB · scikit-learn · pyriemann · PyTorch +
braindecode · React (Vite 5) · Recharts · matplotlib (Agg, server-side topomaps).

**CV lines that are now true and verified:**
- "Leakage-safe cross-validated comparison of 14 classifiers across three families —
  classical (CSP+LDA/SVM/RF), Riemannian geometry (pyriemann tangent space, MDM, FgMDM),
  and deep learning (EEGNet via braindecode)"
- "Reproduces published benchmark accuracy on BCI Competition IV 2a (83.3% four-class,
  single subject)"
- "Dataset-specific preprocessing presets drawn from published pipelines, with generic
  mode-based fallbacks for unknown uploads"
- "Supports BIDS-format EEG datasets via MNE-BIDS" *(end to end since 13 Nov)*

---

## 2. Architecture

### 2.1 Pipeline (as actually built)

```
Input (upload | public dataset fetch | BIDS ZIP)
   │
   ▼
loaders/ or fetchers/ ──► mne.io.Raw or mne.Epochs
   │
   ▼
detector.detect_capabilities() ──► channels, montage, events, available_modes
   │
   ▼
register_session() ──► /tmp/neuroscope/{id}/session.json   (persisted to DISK)
   │
   ═══ user configures mode / preprocessing / classifiers in the wizard ═══
   │
   ▼
POST /analyze
   │
   ├─► pipeline.reload_data(session)      — rebuilds the data from scratch (Option A)
   ├─► presets.get_preset() + user overrides merged
   ├─► pipeline.prepare_epochs()          — routes pre-epoched data past preprocessor
   ├─► pipeline.get_labels()              — y, class_names, class_counts
   ├─► features.extract_features()        — skipped for motor_imagery (CSP lives in CV)
   ├─► classifiers.run_all_classifiers()  — cross_val_predict, per-classifier isolation
   ├─► epochs.compute_psd() ──► PSD curves + visualizer.band_topomaps()
   │
   ▼
results JSON ──► React Results page (Recharts + base64 PNGs)
```

### 2.2 Session storage — "Option A", stateless reload

Chosen deliberately over in-memory caching. `register_session()` writes a `session.json` to
`/tmp/neuroscope/{session_id}/` recording the source, format, file path or dataset name,
subjects, runs, column/variable mapping, and detected capabilities. `/analyze` reads it and
**reconstructs the data from scratch on every call** via `pipeline.reload_data()`.

Costs a reload per request. Buys: survives uvicorn `--reload` restarts (which happen
constantly in development), works across multiple workers, and no memory leak. This was
verified working — a session ID from before several code edits and restarts still analysed
correctly.

**Plan C, designed but not built:** cache preprocessed epochs to `.fif` keyed by a hash of
`{**config, "mode": mode}`. Hit → instant reload; miss → recompute and save. Stateless like
A, fast like B. Needs an eviction policy. This becomes necessary once multi-subject
evaluation lands.

### 2.3 The three analysis modes

**The naming is wrong and known to be wrong.** `mode` determines *which feature extraction
runs*, not what task the data represents. "Motor imagery" mode is really "CSP mode" — CSP is
a general supervised spatial filter that works on any task with a spatial power difference.

| `mode` value | What it actually does | Gated on | Best suited to |
|---|---|---|---|
| `motor_imagery` | CSP inside the CV pipeline → 4–8 components/trial | C3 **and** C4 present + events | Motor imagery, any spatially-lateralised task |
| `spectral` | Band power per channel (delta/theta/alpha/beta/gamma) → n_chans × 5 | ≥4 standard channels | Sleep staging, resting state, emotion, workload |
| `basic` | Per-channel stats (mean/std/var/min/max/ptp) → n_chans × 6 | anything | Unknown data, exploration |

Planned rename (UI label only — keep the internal string, it's in presets/sessions):
**Spatial Filtering (CSP)** / **Frequency Features** / **Signal Statistics**. The C3+C4 gate
should loosen to a *recommendation* rather than a hard requirement.

Riemannian and deep classifiers consume raw epochs and are **mode-independent** — selecting
`riemann_ts` in spectral mode runs exactly the same computation as in motor imagery mode,
ignoring the band-power features. Currently not explained anywhere in the UI.

### 2.4 Supported inputs

| Input | Status |
|---|---|
| EDF / BDF / GDF | ✅ working |
| FIF (raw or epochs) | ✅ working, epochs skip preprocessing |
| BrainVision (.vhdr + .eeg + .vmrk) | ⚠️ backend built, never tested with real files |
| CSV / TSV | ⚠️ ColumnPicker exists, not retested since the api.js rewrite |
| MAT (v7.2 / v7.3) | ❌ backend works, **no VariablePicker UI** — dead ends with an error |
| BIDS ZIP | ✅ working end to end (13 Nov) — EDF/BDF/BrainVision inside; tested in `test_bids.py` |
| Public datasets | ✅ MNE, MOABB, braindecode working |
| OpenNeuro | ❌ **removed** — see section 6 |
| EEGLAB .set | deliberately unsupported |

### 2.5 Request models (`backend/models.py`)

```python
class PreprocessingConfig(BaseModel):
    # EVERY field Optional[...] = None. The frontend sends only what the user
    # changed; model_dump(exclude_unset=True) in /analyze means omitted fields
    # fall through to the dataset preset. Sending a full config would silently
    # override researched presets with generic values.
    bandpass_low, bandpass_high, notch_enabled, notch_freq, reference,
    ica_enabled, ica_components, epoch_tmin, epoch_tmax, baseline_enabled,
    baseline, epoch_train_crop, include_events, montage, picks

class ClassifierConfig(BaseModel):
    selected: List[str]                      # ["lda", "riemann_ts", ...]
    n_folds: int = 5
    csp_components: int = 4
    csp_reg: Optional[Union[str, float]] = None
    configs: Dict[str, Dict[str, Any]] = {}  # {"svm": {"C": 10}} — non-defaults only

class AnalysisRequest(BaseModel):
    session_id: str
    mode: str = "motor_imagery"
    preprocessing: Optional[PreprocessingConfig] = None
    classifiers: ClassifierConfig
```

`configs` is a plain dict rather than one Pydantic field per classifier — with fourteen
classifiers the per-field approach was unmaintainable. Trade-off: no validation on
hyperparameter names, so a typo is silently ignored by the builder's `.get()` default.

`UploadResponse` also carries `is_epochs` and `dataset_name` (added this session so the
frontend can request dataset-specific presets and hide preprocessing for pre-epoched data).

### 2.6 Presets (`core/presets.py`)

`DATASET_PRESETS` keyed by dataset name, `MODE_PRESETS` as generic fallback,
`get_preset(dataset, mode, task)` with a dataset+task → dataset → mode fallback chain.

**`resolve_annotation_rename(preset, runs)`** flattens eegbci's nested mapping. T1/T2 mean
left/right hand for runs [4,8,12] and hands/feet for [6,10,14], so the mapping can't be
resolved until run selection is known. `EEGBCI_RUN_VARIANTS` maps `frozenset(runs)` → variant
name. Called by `prepare_epochs`, which passes the resolved config back to the caller so the
response reports what actually ran.

`get_preset` returns a `copy.deepcopy()` of the preset (fixed 13 Nov) — the presets are
module-level dicts shared by every request, so handing out the live object let one caller's
mutation corrupt every later request. Guarded by `test_get_preset_mutation_does_not_leak`.

### 2.7 Classifier design (`core/classifiers.py`)

Dispatch is by **estimator category**, not by mode:

| Category | Input | CSP? | Members |
|---|---|---|---|
| `feature` | 2D matrix, or epochs via CSP | yes, in MI mode | lda, svm, rf, logreg, extra_trees, hgb, mlp, knn, gnb |
| `epoch` | raw 3D epochs | **no — replaces CSP** | riemann_ts, riemann_mdm, riemann_fgmdm, xdawn_ts |
| `deep` | raw 3D epochs + shape info | no | eegnet |

Non-negotiables:

- **CSP sits inside the sklearn Pipeline, inside CV.** `Pipeline([("CSP", csp), (name, clf)])`
  scored with `cross_val_predict` on `StratifiedKFold(shuffle=True, random_state=42)`. Never
  fit CSP on the full dataset before CV.
- **`StandardScaler` also sits inside the pipeline** for scale-sensitive classifiers
  (`SCALE_SENSITIVE = {svm, knn, logreg, mlp}`), same leakage reasoning.
- **`cross_val_predict`, not `cross_val_score`.** Returns out-of-fold predictions for every
  trial, from which fold scores are recomputed and the confusion matrix is built. Produces
  numerically identical accuracies to `cross_val_score` with the same CV object (verified).
- **`run_all_classifiers` isolates failures per classifier.** One crash returns
  `{"error": ..., "accuracy_mean": None}` and the others still report.

Result shape per classifier: `{classifier, category, accuracy_mean, accuracy_std,
fold_scores, n_folds, confusion_matrix, confusion_labels}`.

**EEGNet configuration — four non-obvious requirements**, each of which caused a failure:

```python
clf = EEGClassifier(
    module=EEGNet,                  # the CLASS, not an instance. Passing an instance
    module__n_chans=n_chans,        # means folds 2+ inherit fold 1's trained weights.
    module__n_outputs=n_outputs,
    module__n_times=n_times,
    module__sfreq=sfreq,            # from epochs.info, never from config
    module__drop_prob=config.get("dropout_rate", 0.25),
    max_epochs=config.get("epochs", 50),
    optimizer=torch.optim.AdamW,    # EEGClassifier defaults to SGD, which barely
                                    # trains this architecture. The paper uses Adam.
    optimizer__lr=config.get("learning_rate", 0.01),   # `lr=` is STORED but never
                                    # reaches the optimiser. Verified via get_params().
    batch_size=config.get("batch_size", 32),
    device=device,
    train_split=None,               # CV handled by cross_val_predict
)
```

Plus `data = data * 1e6` before training — MNE stores EEG in volts (~1e-6) and gradients at
that scale are negligible. Verified std ≈ 4–5 after scaling, which is a reasonable input
range. A decimation block (cap `n_times` at 512) was tried and **removed** — EEGNet has only
~3,400 parameters so input length barely matters, and it changed `sfreq` in ways that weren't
thought through for other datasets.

---

## 3. Repo layout

```
EEG-ai1/
├── backend/
│   ├── main.py                 # FastAPI app, CORS, mne.set_log_level("WARNING"), routers
│   ├── models.py               # all Pydantic models
│   ├── routes/
│   │   ├── upload.py           # /upload/*, /dataset/fetch, /dataset/list/{source}, BIDS
│   │   └── analyze.py          # POST /analyze — orchestration
│   ├── loaders/                # file_loader · tabular_loader · bids_loader · zip_handler
│   ├── fetchers/               # mne_fetcher · moabb_fetcher · braindecode_fetcher
│   │                           # openneuro_fetcher (stubbed to 501)
│   ├── core/
│   │   ├── detector.py         # detect_capabilities, check_motor_channels, assign_best_montage
│   │   ├── presets.py          # DATASET_PRESETS, MODE_PRESETS, resolve_annotation_rename
│   │   ├── preprocessor.py     # preprocess_raw + _apply_* helpers
│   │   ├── features.py         # extract_csp / band_power / basic, FREQUENCY_BANDS
│   │   ├── classifiers.py      # 14 classifiers, category dispatch, CV
│   │   ├── pipeline.py         # reload_data, prepare_epochs, get_labels
│   │   └── visualizer.py       # band_topomaps, _fig_to_base64 (matplotlib Agg)
│   └── utils/                  # file_utils.py · session_registry.py (disk-backed)
├── frontend/src/
│   ├── api.js                  # ALL fetch calls, centralised error unwrapping
│   ├── App.jsx                 # LandingPage ↔ Wizard
│   ├── pages/
│   │   ├── LandingPage.jsx
│   │   └── Wizard.jsx          # 5 steps, holds all shared state
│   ├── components/
│   │   ├── UploadSection.jsx   # tabs: Single · Multiple · Public Dataset · BIDS(disabled)
│   │   ├── DatasetPicker.jsx   # source cards + dataset cards + fetch panel
│   │   ├── BidsSelector.jsx    # written, not yet reachable
│   │   ├── ColumnPicker.jsx    # CSV/TSV, predates api.js
│   │   ├── ChooseSection.jsx   # mode buttons (needs redesign)
│   │   ├── PreprocessSection.jsx
│   │   ├── ClassifySection.jsx # CLASSIFIER_GROUPS + PARAM_FIELDS
│   │   ├── ResultsSection.jsx
│   │   ├── AccuracyChart.jsx   # Recharts bars + error bars + chance line
│   │   ├── ConfusionMatrix.jsx # shaded table, row-normalised, recall column
│   │   ├── PSDChart.jsx        # Recharts lines, band shading, passband clipping
│   │   ├── TopoMaps.jsx        # base64 PNG + band selector
│   │   ├── ParamsExport.jsx    # runnable Python from preprocessing_applied
│   │   ├── Navbar · ProgressBar · Section · ProcessDiagram · ResultsSummary
│   └── App.css                 # one growing stylesheet, white + indigo #6366f1
└── requirements.txt
```

`/tmp/neuroscope/{session_id}/` on Windows resolves to `<current drive>\tmp\neuroscope\` —
which depends on where uvicorn was launched. Worth moving to `tempfile.gettempdir()`.

---

## 4. Current state — verified working

**Backend:** complete and end-to-end functional. `/dataset/fetch` → `/analyze` returns real
results through HTTP, numerically identical to running the pipeline directly in a script.

**Frontend:** the whole wizard works for public datasets.
Upload → Choose → Preprocess → Classify → Results, with four visualisations.

### Verified results — BNCI2014_001, subject 1, motor imagery mode, 5-fold, defaults

| Classifier | Accuracy | Category |
|---|---|---|
| riemann_ts | **83.3% ± 3.3%** | Riemannian |
| riemann_fgmdm | 79.7% ± 3.0% | Riemannian |
| xdawn_ts | 75.7% ± 4.9% | Riemannian |
| riemann_mdm | 74.1% ± 1.9% | Riemannian |
| svm | 74.1% | Classical |
| lda | 73.4% | Classical |
| eegnet | 71.9% ± 4.6% | Deep |

Four-class, chance = 25%, 576 epochs, 22 channels. This matches the range published papers
report for single-subject BNCI2014_001. Riemannian methods leading and classical CSP methods
trailing is the expected and documented result at this sample size.

**Confusion matrices show real neurophysiology:** classes 1 and 2 (left/right hand) separate
cleanly; classes 3 and 4 (feet/tongue) confuse each other heavily, because both are midline
representations with overlapping scalp topography. The alpha-band topomap shows contralateral
mu desynchronisation — blue focus over right hemisphere for left-hand imagery and vice versa.

### Same subject, spectral mode

After the 13 Nov log-band-power fix (5-fold, defaults): lda 57.6%, logreg 57.6%, hgb 51.2%,
rf 46.7%, extra_trees 46.7%, svm 45.0%. Before the fix: lda/logreg 58.5%, svm 42.2%,
**rf 24.8% / extra_trees 24.5% (chance)** — see section 6 for why. The ~15-point drop versus
CSP is the expected cost of generic features over task-specific spatial filtering, and is
itself a useful thing for the platform to demonstrate.

### 13 Nov fetch audit — every public dataset, subject 1, LDA 5-fold, recommended mode

Script: each dataset in its own subprocess with a 600 s timeout (fetch → detect → preset →
epochs → labels → LDA). Results after the fixes below:

| Dataset | Result | Notes |
|---|---|---|
| eegbci L/R hand | ✅ 0.711 (chance 0.51) | 45 trials |
| eegbci hands/feet | ✅ 0.933 (chance 0.53) | **was crashing** — preset hardcoded `include_events=[left_hand, right_hand]`; now derived from the resolved rename in `prepare_epochs` |
| sleep_physionet | ✅ **0.845** spectral (chance 0.51) | **Fixed 13 Nov** (`mne_fetcher.load_sleep_recording`, following MNE's sleep tutorial): was 125 trials (one per annotation, not per 30-s window), all 7 channels typed `eeg` incl. temperature/resp, basic mode only → 0.576. Now: `infer_types=True` (Event marker as misc, NOT stim — a stim channel would override the hypnogram), 30 min of wake kept either side of sleep, R&K → AASM names with 3+4 merged into N3, `event_chunk_duration=30` → 1,103 windows on Fpz-Cz + Pz-Oz. Spectral now unlocked by "≥1 EEG channel" instead of "≥4 standard 10-20 names". Did not hang (cached). |
| ssvep | ✅ 0.600 | 20 trials; labels still `Stimulus/S155`/`S255` |
| BNCI2014_001 (MOABB) | ✅ 0.734 | |
| Zhou2016 | ✅ 0.733 (chance 0.33) | |
| braindecode BNCI2014_001 | ✅ 0.726 | **was 48/576 trials, labels 1–4, 0.167** — fetcher took only `datasets[0]`; stim channel masked annotation names; no preset (generic −0.2–0.8 s). braindecode annotations sit AT THE CUE (duration 4 s) → window 0–4 s |
| braindecode Schirrmeister2017 | ✅ 0.558 (chance 0.25) | was first run only, 0.234. Slow: ~160–265 s |
| erp_core | ⚠️ 0.261 (chance 0.25) | 6 classes mixing stimulus + response events; CSP recommended because C3/C4 exist. Needs ERP mode + event selection |
| limo | ⚠️ 0.505 (chance 0.50) | Face A/B, CSP inappropriate. Needs ERP mode |
| BNCI2014_002 | ⚠️ 0.463 (chance 0.50) | only basic mode offered → channel names not standard; needs a name mapping |
| kiloword | ❌ crash | 960 classes (one per word) → CV impossible. Needs a real label (e.g. word concreteness from metadata) or removal |

### Group 1 quick fixes (session 2) — subject 1 throughout

| Fix | Result |
|---|---|
| Basic features in µV (`extract_basic_features`) | variance was 3.9e-10 V² → below sklearn's 1e-7 tree threshold. eegbci basic: rf 0.689 |
| SSVEP `annotation_rename` S255→12hz, S155→15hz (MNE SSVEP tutorial, verbatim) | spectral logreg 0.95, lda 0.90 (20 trials, 4/fold — noisy) |
| kiloword: abstract vs concrete by **median Concreteness** (`label_kiloword_by_concreteness`) | was a crash; now lda 0.53–0.58, logreg ≤0.59 vs chance 0.50. EXPLORATORY — the split is our choice |
| Motor imagery gate loosened (2.3's plan): needs events + ≥2 EEG ch, not C3/C4 by name; C3/C4 only drive the *recommendation* | BNCI2014_002 (channels EEG1–EEG15): riemann_ts 0.769, lda 0.738 vs chance 0.50 (was basic-only, 0.463) |
| `recommend_for_task`: a public dataset's metadata task overrides the channel-name guess | BNCI2014_002 → motor imagery; SSVEP → spectral (it has C3/C4). kiloword/erp_core/limo still get CSP — needs ERP mode |

BNCI2014_002 channel POSITIONS remain unknown: MOABB says three Laplacian clusters around
C3/Cz/C4 but not which EEGn is which (UNVERIFIED — backlog). No topomaps for it.

### 13 Nov changes (Claude Code session) — all verified on BNCI2014_001 subject 1

- **Test suite:** `backend/tests/`, 35 fast tests (~9 s, synthetic data) +
  4 slow tests (`python -m pytest -m slow -s`, ~20 s, real cached BNCI2014_001 subject 1).
  The slow test pins riemann_ts 0.833 ± 0.033 and lda 0.734 ± 0.045 as a regression baseline.
- **Class names fixed:** MOABB's `get_data(return_epochs=True)` returns an EpochsArray with
  `event_id={'1': 1, ...}`; the names only exist in the separate `labels` array, which the
  fetcher discarded. `moabb_fetcher.restore_class_names` rebuilds `event_id` from it (only if
  codes ↔ labels are one-to-one). Now `left_hand / right_hand / feet / tongue`, 144 each.
- **PSD + topomaps in every mode:** `visualizer.spectral_summary` — never raises; spectral
  mode reuses its PSD, other modes compute it with identical settings.
- **`preprocessing_applied` is honest for pre-epoched data:** reports
  `pre_epoched: true` plus the filter edges and window recorded on the epochs. For BNCI2014_001
  that's 8–32 Hz, window 2–6 s — previously it claimed the generic 7–30 Hz / −0.2–0.8 s preset.
- **`/analyze` returns 400 when `is_ready is False`.** Upload routes now register
  `is_ready = not needs_event_setup` (previously always `True`, contradicting the response).
- **`get_preset` deepcopy**, **log10 band power** (section 6).

### eegbci subject 1, runs 4/8/12, two-class

lda 0.711, svm 0.800, rf 0.600, riemann_ts 0.711, riemann_mdm 0.622. Only 45 trials, so
every difference is within noise — ~9 trials per fold means one trial moves accuracy 11%.

### ⚠️ Subject variability is enormous

The same code on **subject 5** gives riemann_ts 49.5%, lda 37.5%, eegnet ~25–40%. Published
per-subject results on this dataset span roughly 45–90%. Hours were lost this session
comparing a subject-1 run against a subject-5 run and hunting for a nonexistent code
regression. `dataset_summary` now reports `subjects` and `runs`, and the results page shows
the subject — **keep it that way**.

---

## 5. What's left to finish

**Immediate / in progress**
1. ~~**Spectral visuals in all modes.**~~ ✅ Done 13 Nov (`visualizer.spectral_summary`).
2. **ChooseSection redesign.** Card layout like the dataset picker. Rename modes to describe
   methods rather than tasks. Explain what each computes and what suits it. Show why a mode
   is unavailable rather than just greying it out.
3. ~~**`preprocessing_applied` lies for pre-epoched sources.**~~ ✅ Done 13 Nov
   (`analyze.describe_preprocessing`).
4. ~~**Class names show as `1, 2, 3, 4`.**~~ ✅ Done 13 Nov (`moabb_fetcher.restore_class_names`).
   Not yet checked for braindecode / limo / kiloword sources.

**Then**
5. ~~**BIDS ZIP UI**~~ ✅ Done 13 Nov. The "backend complete" claim was wrong: `zip_handler`
   crashed on every upload (read `.file` off a `Path`, returned a file list not a root), and
   `scan_bids_entities` returned per-entity lists while `BidsSelector` expected flat
   combinations, so every dropdown after Subject was empty. Both rewritten; now uses
   `mne_bids.find_matching_paths` so only combinations with a real file are offered.
   BIDS sessions get no dataset preset, so eegbci-derived BIDS shows raw `T0/T1/T2` labels.
6. **MAT VariablePicker** — currently a dead end. Matters because BCI Competition and DEAP
   ship as `.mat`.
7. **Test basic mode** — has never been run once. **Expect the RF bug again:** its features
   are mean/std/var/min/max/ptp in volts (var ~1e-10 V²), all far below sklearn's 1e-7 tree
   split threshold (section 6). Likely needs scaling to µV or log for var.
8. **Multi-subject / cross-subject evaluation** (section 8).
9. **Background tasks + progress polling** — becomes necessary with multi-subject.
10. ~~**ERP mode**~~ ✅ session 2. `features.extract_erp_features`: mean µV in consecutive 50-ms bins
    per channel (Blankertz et al. 2011, NeuroImage 56:814 — equal bins are our generic version of
    their hand-picked intervals). Available with events + ≥1 EEG ch; recommended only via the
    dataset's task (`TASK_MODE["erp"]`). Step 4 "Recommended" in ERP mode = lda (+lsqr/auto
    shrinkage) / logreg / xdawn_ts. erp_core now labelled **correct vs error** exactly as MNE's
    metadata tutorial does (`label_flankers_responses`). **Balanced accuracy** added to every result
    (`balanced_accuracy_mean/std`); the Results page ranks by it when the majority class exceeds
    1/k + 5 pts. `xdawn_estimator` default scm → oas (pyriemann says it must be regularised; scm
    crashed on rank-deficient erp_core/limo). Subject 1, balanced accuracy:
    erp_core lda+shrinkage **0.782**, logreg 0.753, xdawn_ts 0.716, riemann_ts 0.523 (was 0.26 at 6
    classes); kiloword ~0.60; **limo still at chance (~0.49) in every mode** — see backlog.
11. **Live simulation** — v1 PLAYBACK built (session 2); Nikhil's scope: replay public datasets
    first, uploads next, live decoding later.
    - `backend/core/playback.py` + `routes/playback.py`: `GET /playback/{sid}/info` (channels,
      duration, events with real names) and `GET /playback/{sid}/stream?start=&speed=&channels=`
      (Server-Sent Events, 25 frames/s, µV, events emitted as the playhead passes them; paced
      against the wall clock so it doesn't drift). Pause/seek/speed = reconnect at a new `start`.
    - Sources: continuous MNE + braindecode recordings (eegbci, Sleep-EDF, SSVEP, ERP CORE,
      braindecode BNCI2014_001 / Schirrmeister). MOABB route and limo/kiloword are pre-epoched →
      clear 400 pointing to the braindecode version.
    - Display filter always 0.5–40 Hz (+ preset notch), NOT the analysis band; zero-phase is fine
      for display. Live decoding will need causal filtering (`raw.filter(..., phase="minimum")`).
    - In-memory cache of the 2 most recent prepared recordings: first frame 0.04 s / seek 0.07 s on
      77-min BCI IV 2a (was several seconds per play/seek). Only a cache — statelessness holds.
    - UI: "Play recording" on the Data loaded card → `components/playback/PlaybackView.jsx`:
      canvas traces (10-s window, ≤8 channels, motor first), event markers + legend + recent events,
      play/pause, 0.5–4× speed, seekable timeline with event ticks, gain.
    - Tests: `test_playback.py` (every sample once, event ordering/timing, seek, channel defaults,
      refused sources, display filter + eegbci renames). Measured over HTTP: eegbci 2.55 s of data
      in 3.0 s wall-clock (connection setup included).
    - NOT screenshot-verified: headless Chrome can't capture a page with an open stream (virtual
      time never settles). Check in a real browser.
    - Next: uploaded files (continuous EDF/BDF/FIF/BIDS), then live decoding on the same stream
      (train on early trials, classify each trial as its window completes).
12. **PDF report**, info/manual page, deploy.

---

## 6. Things that broke and why — read before debugging

Every one of these cost real time this session. They're recorded so they don't cost it twice.

### `openneuro-py` poisoned DNS for the entire process
It depends on `niquests`, which depends on **`urllib3-future`** — a fork that installs under
the `urllib3` name and replaces urllib3's DNS resolution with its own. Downloads from
physionet.org, lampx.tugraz.at and others failed with `NameResolutionError` while `nslookup`
and `socket.gethostbyname` resolved the same hosts fine. **Removed `openneuro-py`,
`niquests` and `wassima`.** `openneuro_fetcher.fetch_openneuro_dataset` now raises 501.
It was also installed in system Python and had to be removed there separately.
*If OpenNeuro support comes back, reimplement over plain `requests` against their S3/API
endpoints — do not reinstall `openneuro-py`.*

### MOABB writes to a mangled path on Windows
`moabb.datasets.download.data_dl` runs `_sanitize_path()` over the **full absolute path**
including the drive letter, turning `C:\Users\...` into `C-\Users\...`, which is no longer
absolute and resolves relative to the working directory. So MOABB datasets land in
`<cwd>/C-/Users/nikhi/mne_data/MNE-bnci-data/...` — inside the repo. Reproduce with:
```python
from moabb.datasets.download import _sanitize_path
from pathlib import Path
print(_sanitize_path(Path(r"C:\Users\nikhi\mne_data")))   # -> C-\Users\nikhi\mne_data
```
Upstream bug, no config workaround. `C-/` is gitignored. Note the cache location follows the
working directory, so launching uvicorn from elsewhere re-downloads everything.
MOABB also sets `verify=False` on all its downloads (hence the TLS warnings).

### EEGNet trained at chance for most of a day
Three separate causes, found in sequence:
1. `EEGClassifier(lr=0.01)` **stores the value but never passes it to the optimiser** —
   confirmed with `get_params()["optimizer__lr"] == "not set"`. Needs `optimizer__lr=`.
2. `EEGClassifier` **defaults to SGD**, not Adam. At lr=0.01 SGD barely moves this
   architecture. The EEGNet paper and braindecode examples use AdamW.
3. Passing a **module instance** rather than the class means skorch's clone reuses the same
   object across CV folds, so folds 2–5 inherit fold 1's trained weights.

Diagnostic that would have found all three faster: print `data.shape`, `data.std()`, and the
config values actually arriving at the builder, then watch whether `train_loss` moves off
ln(n_classes) in the first ten epochs. Loss pinned at exactly 1.3863 on a 4-class problem
means the network is not learning at all.

### `isinstance(obj, mne.Epochs)` is wrong
`EpochsFIF` (read from .fif) and `EpochsArray` (from MOABB) are **siblings** of `mne.Epochs`
under `mne.BaseEpochs`, not subclasses. `detect_capabilities` took the Raw branch for
kiloword and crashed on `obj.n_times`. Always use `mne.BaseEpochs`.

### Dataset metadata was wrong in three of six MNE entries
- **erp_core is not BIDS.** MNE ships a single 118 MB `.fif` Raw file: one subject, Flankers
  task (the ERN paradigm), 33 channels at 1024 Hz. Not the 5-subject 7-paradigm OSF release.
  The BIDS round-trip built for it was deleted.
- **kiloword** `data_path()` returns a `Path` in the current version, not a list of one.
- Sizes were guesses and mostly wrong — eegbci is 7 MB not 20, limo 335 not 700, erp_core
  118 MB not 1 GB. `DATASET_NOTES` in `DatasetPicker.jsx` now holds measured values for
  what's been fetched; BNCI2014_002 and Schirrmeister2017 are still estimates.

### Sleep-EDF cannot be downloaded
`sleep_physionet.age.fetch_data` uses MNE's **legacy `_fetch_file`** path rather than pooch —
no progress bar, no retries, no timeout. It opens a connection, receives zero bytes, and
hangs indefinitely. Ctrl+C does not work (the thread is blocked in a C-level socket read);
the process needs SIGKILL. `curl` reaches the same URL instantly and reports
`Content-Length: 51110528`, so the server is fine.
**Workaround:** download `SC4011E0-PSG.edf` and `SC4011EC-Hypnogram.edf` manually from
PhysioNet into `~/mne_data/physionet-sleep-data/` — `fetch_data` checks for existing files
and skips the network.

### `uvicorn --reload` detects changes but keeps serving OLD code (Windows)
Seen twice on 13 Nov: the log says `StatReload detected changes in 'backend\fetchers\mne_fetcher.py'.
Reloading...` but no new `Started server process` line follows, and requests keep hitting the
old worker. A sleep-EDF run produced old-code results (7 channels, 160 epochs) after the fix
was in place. **After editing backend code, confirm the change is live** (e.g.
`curl "http://127.0.0.1:8000/preprocessing/preset?dataset_name=sleep_physionet&mode=spectral"`
shows `event_chunk_duration`) — or just stop uvicorn (`Get-Process python | Stop-Process -Force`)
and start it again. Root cause not investigated.

### A stalled download wedges a worker permanently
Dataset fetches run synchronously inside the request. A hung transfer blocks the worker,
ignores SIGINT, and needs the process killed. **This blocks hosting.** `requests`' timeout is
per-read, not total, so a connection trickling bytes never trips it. The real fix is a
background task with a total-elapsed watchdog and a cancel path.

### Random Forest sat at exactly chance in spectral mode
scikit-learn's trees treat any feature whose range is `<= FEATURE_THRESHOLD = 1e-7` as
constant and never split on it (`sklearn/tree/_partitioner.pxd`, sklearn 1.9.0). Raw band
power is ~1e-12 V²/Hz — on BNCI2014_001 subject 1 the *largest* feature range was 2.2e-11 —
so RF and Extra Trees saw 110 constant features and predicted at chance (24.8%, 24.5%).
HGB was unaffected (51.2%) because it bins features into histograms first. **Fix:**
`extract_band_power_features` returns `log10(band power)`; RF/ET → 46.7%. `raw_psds` stays
linear for the plots. *General lesson: any feature in SI units of volts or V² is a candidate
for this — scale or log it before it reaches a tree.*

### `eegbci.load_data`'s first parameter was renamed
`subject` → `subjects`. Pass positionally to be version-agnostic. Also pass
`update_path=True` — without it, MNE calls `input()` and prompts on stdin, which under
uvicorn means a hung worker. `limo.load_data` has the same issue; the `data_path()` functions
default to `update_path=True` and are safe.

---

## 7. Backlog

### Blocking deployment
- [x] ~~No timeout or cancellation on dataset fetches~~ — session 2: `backend/jobs.py` +
      `routes/jobs.py`. Each fetch/analysis runs in its own **spawned process** (threads can't
      be killed out of a C-level socket read). State on disk at `/tmp/neuroscope/jobs/<id>/state.json`
      (stateless, survives restarts, any worker can cancel). Watchdog thread + lazy check on poll
      enforce timeouts (fetch 600 s, analyze 1800 s; `?timeout_s=` overrides). Cancel kills the
      process tree via psutil. Stages reported by `run_fetch`/`run_analysis` (`progress` callback),
      incl. "Training riemann_ts (2 of 5)". Endpoints: `POST /jobs/fetch`, `POST /jobs/analyze`,
      `GET /jobs/{id}`, `POST /jobs/{id}/cancel`. Old sync routes kept. UI (DatasetPicker, Run
      analysis) uses `api.runJob` (1 s polling) with live stage + Cancel. Tests: `test_jobs.py`
      (completion, timeout kills process, cancel, error surfacing, real eegbci fetch→analyze = 0.711).
      Cost: ~3 s process start-up per job. `pyriemann` and `psutil` added to requirements.txt.
- [ ] **Job directories are never cleaned up** — `/tmp/neuroscope/jobs/` grows like sessions do;
      fold into the session-expiry sweep below.
- [ ] **Session expiry** — `/tmp/neuroscope/` grows forever. Uploaded files are other
      people's data; a sweep deleting sessions older than ~24h is a privacy requirement as
      much as a disk one. `cleanup_session` is only called on failure paths.
- [ ] **Dataset cache cap** — `~/mne_data` has no size limit. Schirrmeister2017 alone is
      several GB and currently exists in two copies (braindecode's and MOABB's). Needs LRU
      eviction and a per-dataset size limit, plus awareness of MOABB's `C-/` path.
- [ ] **Orphaned temp files** — every failed download leaves a `tmp*` file behind.
- [ ] **Summary panel shows one subject's trial count on multi-subject runs** — e.g. "Trials 576"
      beside a 9-subject result (5184 trials). SummaryPanel reads the fetch response, not the
      result. Seen in the online demo, 14 Nov.

### Correctness
- [x] ~~`get_preset` returns the live dict~~ — deepcopy, 13 Nov.
- [x] ~~`preprocessing_applied` reports preset values for pre-epoched sources~~ — 13 Nov.
- [x] ~~Class names unresolved for MOABB~~ — 13 Nov.
- [x] ~~RF at chance in spectral mode~~ — 13 Nov, log10 band power. Cause confirmed (section 6).
- [ ] **SVM is the weakest classifier in spectral mode** (45.0% vs LDA 57.6%, subject 1) even
      with log features and its StandardScaler. Default `C`/`gamma` may be badly matched to
      110 features from 576 trials — worth a small grid check before blaming the features.
- [ ] **Class names for braindecode / limo / kiloword** — unchecked; same symptom possible.
      (kiloword now labelled abstract/concrete; braindecode fixed.)
- [ ] **BNCI2014_002 electrode positions** — EEG1–EEG15 order not documented by MOABB. Find it in
      Steyrl et al. 2016 / BNCI Horizon docs before naming channels. UNVERIFIED until then.
- [x] ~~kiloword, erp_core, limo are recommended CSP~~ — ERP mode is now recommended for them.
- [ ] **limo: Face/A vs Face/B is at chance in every mode.** UNVERIFIED hunch: LIMO's manipulated
      variable is the faces' phase coherence / noise level (MNE's LIMO tutorial models that), not
      face identity — so A vs B may simply carry no signal. Check the tutorial + epochs.metadata
      before relabelling. Also slow (128 ch, 1055 epochs: ~5 min for 4 decoders).
- [ ] erp_core preset has `ica_enabled: True`; MNE's tutorial doesn't use ICA and does use
      `reject={"eeg": 250e-6}` (we have no reject option). Worth comparing.
- [x] ~~Classify step should flag decoders that don't suit the data~~ — 13 Nov: `suitability()` in
      `ClassifySection.jsx` (warn: Riemannian with ≤3 EEG ch, xdawn on non-ERP datasets, EEGNet on
      trials > 8 s; block: FgMDM when c(c+1)/2 < n_classes−1). Plus quick picks (Recommended / All
      classical / Clear). Step 3 now shows the analysis window (`epoch_train_crop`) as editable
      fields. Backend: `UploadResponse.n_eeg_channels` and `.analysis_classes`
      (`presets.analysis_class_names`) so the UI shows left/right hand, not T0/T1/T2.
      Riemannian/deep are no longer disabled in basic mode — they always read the raw trials.
      Original notes:
      Sleep-EDF subject 4 (2 EEG ch, 5 stages, 30-s windows), spectral: classical 0.77–0.80, but
      riemann_ts 0.63, riemann_mdm 0.50 (≈ chance 0.50), xdawn_ts 0.66, eegnet 0.68, and FgMDM can't
      run at all (now a clear 400: tangent space c(c+1)/2 = 3 < n_classes−1 = 4). Rules worth showing:
      Riemannian needs more than ~2 channels; xdawn needs time-locked ERPs; EEGNet is built for
      short (≈1–4 s) trials, not 30-s windows.
- [ ] **`xdawn_ts` is offered for motor imagery** where it's inappropriate (it needs a
      time-locked evoked response). `detector.py` can't distinguish ERP from MI yet.
- [ ] **`PARAM_FIELDS` defaults must mirror backend fallbacks.** Long-term fix: serve
      hyperparameter defaults from the backend, same pattern as `/preprocessing/preset`.
- [ ] **Deep branch `confusion_labels` are LabelEncoder indices (0..n-1)** while other
      categories return raw event codes. Ordering matches so rendering is correct, but the
      integers differ between rows of the same response.
- [x] ~~`/analyze` doesn't check `session["is_ready"]`~~ — 13 Nov (400 when `is_ready is False`).
- [ ] `routes/upload.py` carries ~10 unused imports and an unused `_build_early_response`
      (flagged by Pylance). Cleanup only.

### Data handling
- [ ] MAT loader generates generic `"EEG 1"…` channel names, which breaks topomaps. Needs an
      optional `channel_names` field on `VariableMapping`.
- [ ] `tabular_multi` (several CSV/TSV as one dataset) is a placeholder error.
- [ ] `inspect_csv`/`inspect_tsv` read the whole file for a 3-row preview. Use `nrows=`.
- [ ] `ssvep` annotations are `Stimulus/S255` / `Stimulus/S155`, should rename to `12hz`/`15hz`.
- [x] ~~`zip_handler` should skip `__MACOSX` entries~~ — 13 Nov; also caps ZIPs at 2 GB
      uncompressed (checked before extracting) and deletes the ZIP after extraction.
- [ ] BIDS run dropdown sorts as strings (`12, 4, 8`). Natural sort in `BidsSelector`.
- [ ] ColumnPicker's role heuristic marks EOG/ECG/STI as EEG channels.

### Preprocessing / features
- [ ] `FREQUENCY_BANDS` is hardcoded; PSD `fmin`/`fmax` hardcoded at 0.5/45. Should clamp to
      the data's actual passband — on MOABB data, 3 of 5 bands are filter rolloff and 66 of
      110 features are dead weight.
- [ ] Manual ICA component selection (currently auto EOG-based or nothing).
- [ ] FIR `firwin` only — no IIR, filter length, transition bandwidth or order controls.
- [ ] CSP `log` / `norm_trace` hardcoded.
- [x] ~~CSP leakage~~ — solved with Pipeline-inside-CV. **Keep it that way.**
- [x] ~~Pre-epoched routing~~ — `prepare_epochs` handles it.
- [x] ~~Unconsumed preset keys~~ — `annotation_rename`, `epoch_train_crop`, `montage`,
      `include_events`, `picks` all now applied.

### Classifiers
- [ ] **Cross-subject transfer is near chance** (BNCI s1/4/5: ~0.30 vs 0.25). Standard remedy:
      Riemannian re-centering / alignment of each subject's covariances before pooling (pyriemann
      has transfer-learning tools — UNVERIFIED which API in 0.12; check docs before building).
- [ ] **Cross-subject memory**: pooling all 9 BNCI2014_001 subjects is ~0.9 GB float64 before
      copies. Consider float32 or per-fold loading if larger cohorts are offered.
- [ ] `MAX_EEGNET_EPOCHS = 50` silently caps whatever the user enters, and loss is often
      still descending at 50. Either raise it or warn when training hasn't converged.
- [ ] `SVC(probability=True)` is deprecated in sklearn 1.9, removed in 1.11. Nothing calls
      `predict_proba` yet, but `build_voting_ensemble` with soft voting would need it.
- [ ] `build_voting_ensemble` exists but is never called from anywhere.
- [ ] Expose the optimiser as a config option (AdamW / Adam / SGD) — comparing them is a
      legitimate experiment and would have made the SGD default visible immediately.
- [ ] Per-trial standardisation inside the CV pipeline for deep models, as a sklearn
      transformer rather than the current fixed `* 1e6`.

### Visualization
- [x] ~~PSD / topomaps only in spectral mode~~ — 13 Nov.
- [ ] Per-channel PSD view — the class-averaged curves are nearly identical on motor imagery
      data because averaging across channels destroys the spatial information.
- [ ] CSP pattern topomaps — needs refitting on the full dataset after CV, for visualisation
      only, clearly labelled as such.
- [ ] ERD/ERS maps, time-frequency (`epochs.compute_tfr`), scrolling raw viewer.
- [ ] Topomap colorbar sits inside the figure and steals subplot width, pushing heads left.

### Frontend
- [ ] MAT VariablePicker, Info/manual page. (BIDS tab done 13 Nov.)
- [x] ~~Landing page redesign~~ — 13 Nov. Light indigo/blue glass theme scoped under `.lp` in
      `pages/LandingPage.css` (wizard untouched). Fonts: Plus Jakarta Sans + JetBrains Mono.
      `components/landing/`: `HeadMap` (10-20 electrodes, C3/C4 highlighted), `EegTraces`
      (decorative 1/f waveform), `ResultsPreview` (REAL subject-1 numbers incl. riemann_ts
      confusion matrix — regenerate if the pipeline changes). `Navbar.jsx` / `ProcessDiagram.jsx`
      deleted. The labelled ERD trace (cue → contralateral mu drop) was prototyped and is
      reserved for the live-simulation feature.
- [x] ~~Wizard shell + Upload step redesign~~ — 13 Nov. Shared tokens in `styles/tokens.css`
      (used by `.lp` and `.wz`), shared icons in `components/ui/Icons.jsx`. Shell: sticky glass
      nav with clickable `wizard/Stepper` (reachable steps only), sticky `wizard/SummaryPanel`,
      floating footer. Results become unreachable when settings change after a run
      (`resultsKey` vs `configKey` in `Wizard.jsx`). Upload: route cards (public / own files),
      `upload/Dropzone` infers single / BrainVision set / BIDS from the selection (tabs gone),
      `upload/DetectedCard` with `wizard/ChannelMap` (real `standard_1020` positions exported
      to `data/montage_1020.json`). `DATASET_NOTES` sizes now MEASURED per subject from this
      machine's cache (BNCI2014_001 86 MB, eegbci 8 MB, Schirrmeister2017 490 MB…); unmeasured
      ones are labelled "est." in the UI. `ColumnPicker` now goes through `api.js`.
      `ProgressBar.jsx` deleted; dead shell CSS removed from `App.css` (753 → 420 lines).
      Steps 2–5 still use old components with an interim restyle (`.wz .primary-btn` etc.).
- [x] ~~Step 2 (Choose) + Results redesign~~ — 13 Nov. Choose: three calm cards titled
      Motor imagery / Spectral / Basic with the method underneath; locked cards say why.
      Results: built from shared `components/results/` (StatTile, AccuracyBars, ConfusionTable,
      metrics.js) — the SAME components as the landing preview, so they can't drift. Click a bar
      to switch the confusion matrix; auto-generated one-line reading; band tabs for topomaps.
      Chance level is now the largest-class share (was 1/k). Shared styles in
      `styles/results.css`. Deleted AccuracyChart, ConfusionMatrix, TopoMaps, ResultsSummary.
- [ ] BIDS selection reached via `DatasetPicker` loses `subjects`/`display_name` context
      (BidsSelector's onConfigured returns the raw backend response).
- [ ] ESLint `react/prop-types` fails on every component (168 errors) — nothing uses PropTypes.
      Either turn the rule off in `eslint.config.js` or adopt PropTypes; decide once.
- [ ] `API_BASE` hardcoded in `api.js` → `import.meta.env.VITE_API_BASE`.
- [ ] `react-router-dom` for `/`, `/app`, `/info`, `/results/:id`.
- [ ] One growing `App.css` → CSS modules or Tailwind.
- [ ] Commented-out previous versions still sit at the top of several components. Git has
      the history; delete them.
- [ ] `ResultsSummary.jsx` may be dead code now.

---

## 8. Multi-subject evaluation — BUILT (session 2)

**Status:** `backend/core/multisubject.py` (`run_within_subject`, `run_cross_subject`),
`AnalysisRequest.evaluation` ("single" | "within_subject" | "cross_subject") + `.subjects`.
`run_classifier(groups=...)` switches to `LeaveOneGroupOut`; results gain `fold_groups` and
`per_subject`. Within-subject keeps one subject in memory at a time; cross-subject pools
(BNCI2014_001: ~0.9 GB float64 for 9 subjects — watch memory). Multi-subject jobs default to a
3600 s timeout. UI: Evaluation card in Step 4 (public datasets with >1 subject), decoder × subject
grid on Results (`SubjectGrid`), spread labelled "across subjects". Tests: `test_multisubject.py`
(incl. CSP fit sizes prove the held-out subject never reaches training).

**Verified, BNCI2014_001 subjects 1/4/5, 5-fold, defaults:**
- within-subject: riemann_ts 0.677 ± 0.139 (s1 0.833 = single-subject baseline exactly, s4 0.701,
  s5 0.495 = the "subject 5 ≈ 49.5%" from section 4); lda 0.559 ± 0.147
- cross-subject (train on 2, test on 1): riemann_ts 0.303 ± 0.037, lda 0.328 ± 0.054 — barely above
  chance 0.25. Expected without alignment; see backlog (Riemannian re-centering).

### Original design notes

The most valuable next feature, and the one this session's subject-variability confusion
makes the case for.

**Two distinct experiments:**

**Within-subject across a cohort.** Train and test separately on each subject, report mean ±
std across the group. This is what BCI papers report — "72.3% ± 11.4% across 9 subjects" —
and it's what a single-subject number fundamentally cannot express. Each subject gets its own
CSP filters and its own classifier.

**Cross-subject transfer.** Pool everyone, train on subjects 1–8, test on 9. Much harder,
scores much lower, and it's the genuinely interesting research question: can a BCI work
without per-person calibration?

**What's in the way:**
- `fetch_mne_dataset` does `subject = subjects[0]` and discards the rest.
- MOABB's `paradigm.get_data(subjects=[...])` **already accepts a list** and returns a
  `metadata` DataFrame with a `subject` column — currently thrown away in
  `fetch_moabb_dataset`. That column is what splits pooled epochs back into groups.
- Pooled CV would leak: `StratifiedKFold` puts the same subject in train and test, reporting
  within-subject accuracy while claiming cross-subject. Needs `GroupKFold` /
  `LeaveOneGroupOut` with subject as the group.
- Runtime and download size both multiply. This is where the `.fif` cache and background
  tasks stop being optional.

Shape of the change: `evaluation: "single" | "within_subject" | "cross_subject"` on
`AnalysisRequest`; results grow a per-subject breakdown alongside the aggregate; the results
UI becomes classifier × subject with an expandable detail view and a spread chart.

---

## 9. Ideas beyond the MVP

- **"Fetch from anywhere" (Nikhil's idea, session 2):** fetch datasets the way repositories offer
  them via terminal/code (OpenNeuro, Zenodo, PhysioNet…). **Never execute user-supplied commands
  or code on the server** — once hosted that is remote code execution, same as the pickle warning
  below. Safe shape: an allowlist of source types, each with a fixed downloader the backend owns;
  the user supplies only an identifier (OpenNeuro accession `ds00xxxx`, Zenodo record id, URL on an
  allowed host). Runs as a background job (timeouts/cancel already exist). OpenNeuro must be rebuilt
  on plain `requests` against its API — never reinstall `openneuro-py` (section 6).
- **Helsinki neonatal EEG (seizure detection), session 2:** Nikhil ran LOSO on it elsewhere with much
  higher accuracy than BNCI cross-subject. Different task (seizures transfer across patients far
  better than motor imagery) and heavily imbalanced — report balanced accuracy / sensitivity /
  specificity / AUC, not plain accuracy. Labels live in separate per-expert annotation files, not
  EDF events. Would need a fetcher + "clinical / seizure detection" task. Verify source + format
  against the official dataset page first (details here are from memory).

- **More braindecode models** — ShallowFBCSPNet (Schirrmeister 2017), Deep4Net,
  SleepStagerChambon2018 (pairs with sleep_physionet), EEGConformer, ATCNet. Generalise
  `_build_eegnet` into `_build_braindecode_model(name, ...)`. Verify constructor signatures
  per model first; they differ.
- **Cropped decoding** for EEGNet — braindecode's actual recommendation for BNCI2014_001.
  Each trial yields many overlapping windows, multiplying effective sample count. Likely the
  real fix for deep-learning performance at this dataset size.
- **FBCSP**, more pyriemann pipelines (`MeanField`, `KNearestNeighbor`, `ERPCovariances`).
- **User-defined pipelines via an allowlist** — a JSON spec naming supported components that
  the backend constructs. **Never accept pickled models or uploaded `.py`** — `pickle.load`
  is remote code execution. A safer and arguably more useful alternative: extend
  `ParamsExport` to emit a complete runnable script so users swap in their own estimator
  locally.
- **Experiment history** — save runs to SQLite, compare across runs and datasets.
- **Pre-computed demo runs** so portfolio visitors see results with no wait.
- **Learning curves** (accuracy vs number of trials) — would demonstrate the
  "classical wins on small data, deep wins on big data" point directly.
- **Data quality report** — flat/noisy channels, line-noise level, amplitude outliers.
- **`/dataset/cache` endpoint** + a "cached" badge on dataset cards, so users know which
  fetches are instant. Must check both `~/mne_data` and the MOABB `C-/` path.

---

## 10. UI / frontend improvements

**DECIDED (13 Nov): the real Results page must match the landing page's results preview**
(`frontend/src/components/landing/ResultsPreview.jsx`). Nikhil approved that design
explicitly. Concretely: a window-style card; a row of stat tiles with coloured circular icons
(best accuracy, fold spread, macro F1, chance level); accuracy bars grouped by family colour
(Riemannian violet, classical blue, deep pink) with a dashed chance marker drawn per row;
a confusion matrix as a table, row-normalised %, green diagonal scaled by value, pink tint on
off-diagonal cells ≥5%, an F1 row, and a one-line plain-English interpretation under it;
scalp maps full width with a caption explaining what blue means. Reuse those `rp-*` styles
rather than restyling from scratch.

**Visual direction (13 Nov, approved):** light indigo/blue with frosted-glass cards, Plus
Jakarta Sans + JetBrains Mono for numbers/channel names — see `pages/LandingPage.css`. This
supersedes the "dark surfaces" idea below; the app pages should follow the landing page.

**Visual design (original notes).** Currently white + indigo `#6366f1`, system-ui, 800px column. Worth moving
to a deliberate identity: dark surfaces for signal views (traces read better on dark), one
strong accent, monospace for numbers and channel names, a proper type pairing. The landing
hero could use an animated EEG trace or a pulsing topomap instead of the static box diagram.

**UX**
- Clickable stepper back to completed steps, keeping config.
- Disabled modes and classifiers should explain *why* in a tooltip.
- Inline validation (bandpass high < Nyquist, tmin < tmax) with plain-English "what does this
  do?" popovers.
- A badge showing which preset was applied and where it came from
  ("ERP CORE N400 published pipeline").
- Real progress for long runs with a cancel option.
- Empty/loading/error states everywhere — friendly boxes, not raw `detail` strings.
- Sticky summary sidebar: dataset · subject · mode · filters · classifiers.
- **Warn about subject variability** on the dataset card and results page.
- Responsive layout; signal views should go full width.

**Already done this session:** dataset cards with size/speed badges and documentation links,
subject field with an explanation of what "subject" means, classifier grouping by category
with per-classifier notes, preset-vs-override highlighting in the preprocessing form,
chance-level reference line, fold-spread column, an interpretation note about statistical
resolution, and "Did not run" surfacing for failed classifiers.

---

## 11. Deployment notes

### Online demo — BUILT (14 Nov 2026), static, no backend

Hugging Face Docker Spaces need a paid plan for personal accounts (PRO, $9/month — per
https://huggingface.co/docs/hub/spaces-overview, checked 14 Nov), so the public version is a
**static demo build** that runs with no server and can go on any free static host.

- `npm run build:demo` (in `frontend/`) → `frontend/dist-demo/`, relative paths (`--base ./`),
  so it works from any URL or sub-folder. `npm run dev:demo` runs it locally. `.env.demo` sets
  `VITE_DEMO=1` and `VITE_REPO_URL` (empty hides the "Get the full app" link — fill in once the
  repo is public).
- `src/api.js` exports `DEMO ? demo.x : live_x` for every call; `src/demo/demoApi.js` answers
  from `public/demo/`. Simulated jobs report the same stages as the real ones. Playback replays
  an int16 excerpt through the same frame protocol as `stream_frames`. Uploads, other datasets
  and other subjects give a clear "needs the full app" message.
- What's recorded (`python -m backend.demo.record_demo`, backend running, ~65 min):
  BCI IV 2a within- and cross-subject over subjects 1–9 (MI mode, all decoders except EEGNet);
  best subject by tangent space = **subject 3 (88.5%)** in all 4 modes with all 14 decoders;
  Sleep-EDF subject 1 in all 4 modes; a 30-min Sleep-EDF playback excerpt from 11.25 h
  (19 stage markers). 4.2 MB total. Per-subject tangent space: 83.3 / 63.7 / 88.5 / 70.1 /
  49.5 / 62.3 / 85.4 / 87.7 / 79.2 (S1–S9), mean 74.4 ± 12.9.
- Limits, stated in the UI by `src/demo/DemoNote.jsx`: preprocessing and hyperparameter edits
  don't change results; multi-subject is fixed to all 9 subjects.
- `backend/tests/test_demo_data.py` checks the recording is complete and consistent.
- Re-record after any change that alters results, or the demo shows stale numbers.

### Full hosted backend (later)

- Backend is heavy (torch, MNE, MOABB, pyriemann). Options: Hugging Face Spaces (Docker),
  Render, Fly.io with CPU-only torch. EEGNet on a free CPU tier will be slow — cap epochs or
  offer it only in pre-computed demos.
- Frontend: static Vite build on Vercel/Netlify pointed at the backend URL.
- **Blockers before any public deployment:** the fetch timeout/cancellation problem, session
  expiry, and the dataset cache cap. Any user can currently wedge a worker or fill the disk.
- Pre-fetch datasets into the image or a volume so the demo doesn't download at request time.
- Restrict CORS to the portfolio domain; add upload size limits.
- **Portfolio page:** 1-line pitch, 30–60s GIF (fetch → results), architecture diagram,
  screenshots of the topomap and confusion matrix (the two most visually compelling),
  technical-highlights bullets, live demo + repo links, "what I'd do next".
- **README:** setup, supported-formats table, datasets tested, accuracy table vs literature,
  known limitations from section 7.

---

## 12. Suggested first prompt for a new session

> "Read `NEUROSCOPE_CONTEXT.md`. Scan the repo and tell me which claims in sections 4–7 are
> out of date — especially whether the spectral visuals are now computed in all modes.
> Then set up pytest under `backend/tests/` with tests for: CSP staying inside the CV
> pipeline, pre-epoched routing in `prepare_epochs`, per-classifier failure isolation in
> `run_all_classifiers`, and `detect_capabilities` on both Raw and Epochs inputs. Run them
> and paste the actual output. Cite the docs for every API you use, and verify library
> signatures with `inspect.signature()` before writing code against them."

*(Done 13 Nov — the suite exists. `python -m pytest` for the fast tests,
`python -m pytest -m slow -s` for the real-data regression check.)*
