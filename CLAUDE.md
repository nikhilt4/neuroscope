# CLAUDE.md — working agreement for EEG-ai1

Project state, architecture and backlog live in `NEUROSCOPE_CONTEXT.md`. Read that for
*what* the project is. This file is *how* to work on it.

---

## 0. NEVER run git or GitHub commands

**Under no circumstances run `git`, `gh`, or anything that commits, pushes, branches,
stages, resets, stashes or talks to GitHub.** Not even read-only ones (`git status`,
`git diff`, `git log`) unless Nikhil explicitly asks for that specific command. Nikhil
handles all version control. Edit files on disk, then list what changed so Nikhil can
review and commit.

---

## 1. Show me where you got it from

This is the most important rule. Every library call, parameter, default value, dataset fact
and preprocessing setting needs a source.

- Link the official docs page or tutorial, and say which version it applies to. Check the
  installed version with `pip show <pkg>` first — don't cite `stable` if we're two majors
  behind.
- Put the source in a comment next to the call:
  ```python
  # https://mne.tools/stable/generated/mne.decoding.CSP.html (MNE 1.x)
  csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
  ```
- Preferred sources, in order: official API docs and tutorials (mne.tools,
  moabb.neurotechx.com, braindecode.org, scikit-learn.org, pyriemann.readthedocs.io), then
  the library's source on GitHub, then the original paper. Avoid blog posts and forum
  answers unless nothing else exists — and say so when you use one.
- **If you can't find a source, say "unverified."** Mark it in the code
  (`# UNVERIFIED:` or `"_confidence": "unverified"`) and add it to the backlog. Never
  present a guess as fact.
- When the docs and the existing code disagree, show me both. Don't silently "fix" it.

### Verify signatures before writing against them

Run `inspect.signature()` or `inspect.getsource()` and paste the output *before* writing
code that calls an unfamiliar API. This has caught four real bugs on this project:

```python
import inspect
from mne.datasets import eegbci
print(inspect.signature(eegbci.load_data))     # first arg was renamed subject -> subjects
```

Specifically worth checking: anything in `mne.datasets` (inconsistent return types and
parameter names), skorch/braindecode constructors (`lr=` is accepted and silently ignored —
it must be `optimizer__lr=`), and pyriemann's namespace (reorganised across versions).

---

## 2. Environment and commands

Windows, PowerShell. Repo at `C:\Users\nikhi\Desktop\Projects\EEG-ai1`. Python 3.12 venv at
`./venv`. NVIDIA RTX 4060 available for torch.

```powershell
# activate — always do this first
.\venv\Scripts\Activate.ps1

# backend — MUST run from the repo root, not from backend/
uvicorn backend.main:app --reload          # http://127.0.0.1:8000, docs at /docs

# frontend — separate terminal
cd frontend
npm run dev                                 # http://localhost:5173

# a one-off script that imports backend.*
python -m backend.core.whatever             # from the repo root
```

### Check the interpreter before you start

VS Code has repeatedly launched uvicorn on **system Python** (`C:\Python312`) instead of the
venv, which has a different package set and caused hours of confusion.

```powershell
python -c "import sys; print(sys.executable)"   # must be under EEG-ai1\venv
```

If it isn't: Ctrl+Shift+P → "Python: Select Interpreter" → the `EEG-ai1\venv` entry. Make
sure `EEG-ai1` is the VS Code workspace root, not the parent `Projects` folder.

### When uvicorn won't die

A stalled download inside a synchronous request blocks the worker in a C-level socket read
and ignores Ctrl+C entirely.

```powershell
Get-Process python | Stop-Process -Force
```

---

## 3. Testing

**There are currently no tests. This is the biggest gap in the project.** Every bug found so
far was found by running the app and noticing something looked wrong.

- New modules get pytest tests under `backend/tests/`.
- **Run them and paste the actual output**, not a summary. If something fails or you
  couldn't run it, say so plainly.
- Fast unit tests use small synthetic `RawArray` / `EpochsArray` fixtures. A synthetic test
  where one channel carries 3× the variance for one class is enough to prove a classifier
  pipeline is wired correctly.
- Integration tests use real data: `eegbci` subject 1 runs [4,8,12] (7 MB, fast) or
  `BNCI2014_001` subject 1 (cached). API routes go through FastAPI's `TestClient`.
- Check against published numbers where they exist. CSP+LDA on BNCI 2a subject 1 should land
  around 73%; riemann_ts around 83%. Cite the source for whatever you compare against.
- Priority tests to write first:
  - CSP stays inside the CV pipeline (leakage)
  - `prepare_epochs` routes pre-epoched data past `preprocess_raw`
  - `run_all_classifiers` isolates per-classifier failures
  - `detect_capabilities` handles both `Raw` and `BaseEpochs`

### Always name the subject

Accuracy on BNCI2014_001 ranges roughly 45–85% depending which subject was fetched. A result
without a subject number is not a result. Several hours were lost comparing a subject-1 run
against a subject-5 run and hunting for a code regression that didn't exist.

---

## 4. How to work with me

- **You write the code, end to end.** For anything substantial, give me a short plan first,
  then build it. I review the diff.
- **Small, verifiable steps.** I'd rather apply three changes and test between them than
  receive a 400-line rewrite. When you give me a block of code, say exactly which file it
  goes in and what it replaces — "add this to X" is ambiguous when X already has something
  similar.
- **Explain the *why*, not the what.** I can read the code. Tell me what breaks if we do it
  the other way.
- **Tell me when I'm wrong.** If my suggested approach has a problem, say so before
  implementing it.
- **Flag things for later out loud.** If you spot a problem not worth fixing now, add it to
  the backlog in `NEUROSCOPE_CONTEXT.md` section 7 rather than dropping it.
- **Defaults are automatic, but always overridable.** Every parameter needs a sensible
  sourced default and a way to change it, in both the API and the UI.

---

## 5. Python style

Patterns already established in this codebase — match them.

```python
def extract_csp_features(
    epochs: mne.Epochs,
    labels: np.ndarray,
    n_components: int = 4,
    reg: Optional[Union[str, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    One-line summary.

    Explain the non-obvious constraint, not the obvious mechanics. E.g. "CSP is a
    SUPERVISED transform — fitting it outside the CV folds leaks test data, so the
    caller is responsible for wrapping it in a Pipeline."
    """
```

- **Type hints on function signatures.** Return types too.
- **Docstrings explain constraints and gotchas**, not parameter lists that repeat the
  signature.
- **Inline comments for anything surprising**, with the reason:
  ```python
  # EEGClassifier defaults to SGD, which barely trains this architecture.
  # The EEGNet paper and braindecode's examples use AdamW.
  optimizer=torch.optim.AdamW,
  ```
- **`config.get(key, default)` for every user-settable parameter.** The default here must
  match the frontend's `def` in `PARAM_FIELDS` exactly — see section 7.
- **`HTTPException` with a message a user could act on.** `"Motor imagery needs both C3 and
  C4 — this file has neither"`, not `"invalid channels"`.
- **Catch specific exceptions where the handling differs**, broad `except Exception` only
  where failure is genuinely non-fatal (and comment why):
  ```python
  except Exception:
      pass   # visuals are optional — never lose a five-minute EEGNet run to a plot failure
  ```
- **Never mutate a caller's object.** `raw = raw.copy()` at the top of anything that filters.
- Lazy-import heavy or optional dependencies (torch, braindecode, pyriemann) inside the
  function that needs them, wrapped in try/except with a clear install message.

---

## 6. React / frontend style

- Function components, no TypeScript. Plain CSS classes in `App.css` (single growing
  stylesheet — moving to CSS modules is on the backlog, don't start it unasked).
- **All fetch calls go through `src/api.js`.** No `fetch()` in components, no hardcoded URLs.
- **Shared state lives in `Wizard.jsx`** and comes down as props. Local `useState` only for
  genuinely local UI state (which panel is expanded, which tab is active).
- **Hooks before any early return.** A `useState` below an `if (!x) return` violates the
  rules of hooks and throws.
- Recharts for charts. Server-rendered base64 PNGs only where the computation genuinely
  belongs server-side (scalp topomap interpolation).
- Same comment rule as Python — explain the surprising bit:
  ```jsx
  // Chance depends on class count, not a fixed 50% — a 4-class task is at 25%.
  const chance = 1 / summary.classes.length;
  ```

---

## 7. The override pattern — frontend and backend defaults must match

Both the preprocessing form and the classifier config send **only values the user changed**.
Omitted keys fall through to the backend default.

- `PreprocessingConfig` has every field `Optional = None`; `/analyze` uses
  `model_dump(exclude_unset=True)` and merges over the dataset preset. Sending a full config
  would silently override researched presets with generic values.
- `ClassifySection.jsx` holds a `def` per field in `PARAM_FIELDS`; `setParam` deletes the key
  when the value equals `def`.

**If a frontend `def` and its backend `config.get()` fallback drift apart, the UI displays
one value while the backend runs another, with no error.** This happened with EEGNet's
learning rate and took most of a day to find. When you change a default, change both.

Long-term fix on the backlog: serve hyperparameter defaults from the backend, the same way
`/preprocessing/preset` already works.

---

## 8. File hygiene

- **Delete old code, don't comment it out.** Git has the history. Commented-out previous
  versions have already caused: a duplicated `models.py` (syntax error at the join), a
  duplicated `_build_eegnet` where the second silently won, and a stray `f` inside a comment
  block. Several files still carry 100+ lines of dead code at the top — removing it is
  welcome.
- When telling Nikhil what to commit, flag anything that shouldn't go in: `node_modules`,
  `.env`, dataset caches or the MOABB `C-/` folder. (Nikhil runs git — see section 0.)
- Don't run `npm audit fix --force` — it upgrades across majors and would break the
  deliberate Vite 5 pin.

---

## 9. Things that have bitten us

Full detail with reproduction cases in `NEUROSCOPE_CONTEXT.md` section 6. The short list:

- **`isinstance(obj, mne.Epochs)` is wrong.** `EpochsFIF` and `EpochsArray` are siblings
  under `mne.BaseEpochs`, not subclasses. Always check `BaseEpochs`.
- **`openneuro-py` pulls `urllib3-future`**, which replaces urllib3's DNS resolution and
  broke downloads for every other library in the process. Removed. Don't reinstall it.
- **MOABB mangles Windows paths**, writing datasets to `<cwd>/C-/Users/...` instead of
  `MNE_DATA`. Upstream bug, no workaround, `C-/` is gitignored.
- **Pass `update_path=True`** to MNE dataset fetchers — otherwise they call `input()` and
  hang a uvicorn worker waiting on stdin.
- **Dataset metadata is often wrong** until verified against the actual download. Three of
  six MNE entries had wrong subject counts, structures or sizes.
- **A stalled download wedges a worker permanently** — sync request, no timeout, ignores
  SIGINT. Blocks deployment; fixing it is on the backlog.

---

## 10. Quick orientation

```
backend/
  routes/analyze.py     orchestration — read this first to understand the flow
  core/pipeline.py      reload_data / prepare_epochs / get_labels
  core/classifiers.py   14 classifiers, category dispatch, CV
  core/presets.py       literature-derived preprocessing defaults
frontend/src/
  pages/Wizard.jsx      all shared state
  api.js                every backend call
```

Sessions persist to `/tmp/neuroscope/{session_id}/session.json` — on Windows this resolves
relative to whichever drive uvicorn started on. `/analyze` rebuilds data from scratch on
every call (stateless by design), so a session survives server restarts.