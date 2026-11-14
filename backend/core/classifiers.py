import numpy as np
import mne
from typing import Optional, Union
from fastapi import HTTPException

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder



# ── Individual classifier builders ────────────────────────────────────────────

def _build_lda(config: dict) -> LinearDiscriminantAnalysis:
    solver = config.get("solver", "svd")
    # shrinkage is only valid with 'lsqr'/'eigen' — sklearn raises if paired with 'svd'
    shrinkage = config.get("shrinkage", "auto") if solver != "svd" else None
    return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)


def _build_svm(config: dict) -> SVC:
    return SVC(
        kernel=config.get("kernel", "rbf"),
        C=config.get("C", 1.0),
        gamma=config.get("gamma", "scale"),
        probability=True,  # needed if we ever want confidence scores, small perf cost
    )


def _build_rf(config: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth"),
        max_features=config.get("max_features", "sqrt"),
        random_state=42,
    )


def _build_eegnet(
    config: dict,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float = 250.0,
):
    """
    Builds an EEGNet model + skorch wrapper for sklearn-compatible training.
    Uses the real published architecture from braindecode, not a hand-rolled version.

    sfreq comes from epochs.info, not config — a wrong sampling rate here
    silently mis-sizes the temporal kernels (Sleep-EDF is 100 Hz, several
    MOABB sets are 512).
    """
    try:
        from braindecode.models import EEGNet
        from braindecode.classifier import EEGClassifier
        import torch
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Braindecode/torch not available: {str(e)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = EEGClassifier(
        module=EEGNet,
        module__n_chans=n_chans,
        module__n_outputs=n_outputs,
        module__n_times=n_times,
        module__sfreq=sfreq,
        module__drop_prob=config.get("dropout_rate", 0.25),
        module__pool_mode=config.get("pool_mode", "mean"),
        max_epochs=config.get("epochs", 50),
        optimizer=torch.optim.AdamW,
        optimizer__lr=config.get("learning_rate", 0.01),
        optimizer__weight_decay=config.get("weight_decay", 1e-4),
        batch_size=config.get("batch_size", 32),
        device=device,
        train_split=None,
    )
    return clf


def _build_logreg(config: dict) -> LogisticRegression:
    return LogisticRegression(
        C=config.get("C", 1.0),
        penalty=config.get("penalty", "l2"),
        solver=config.get("solver", "lbfgs"),
        max_iter=config.get("max_iter", 1000),
    )


def _build_extra_trees(config: dict) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth"),
        max_features=config.get("max_features", "sqrt"),
        random_state=42,
    )


def _build_mlp(config: dict) -> MLPClassifier:
    """
    Shallow fully-connected net on extracted features. Deliberately kept as a
    contrast to EEGNet: same 'neural network' label, but operating on
    hand-engineered features rather than raw signal.
    """
    return MLPClassifier(
        hidden_layer_sizes=tuple(config.get("hidden_layer_sizes", [64, 32])),
        activation=config.get("activation", "relu"),
        alpha=config.get("alpha", 1e-4),
        learning_rate_init=config.get("learning_rate_init", 1e-3),
        max_iter=config.get("max_iter", 500),
        early_stopping=config.get("early_stopping", True),
        random_state=42,
    )


def _build_hgb(config: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config.get("max_iter", 100),
        learning_rate=config.get("learning_rate", 0.1),
        max_depth=config.get("max_depth"),
        random_state=42,
    )


def _build_knn(config: dict) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=config.get("n_neighbors", 5),
        weights=config.get("weights", "uniform"),
        metric=config.get("metric", "minkowski"),
    )


def _build_gnb(config: dict) -> GaussianNB:
    return GaussianNB()



# ── Riemann model ──────────────────────────────────────────────────────────
def _import_pyriemann():
    try:
        from pyriemann.estimation import Covariances, XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        from pyriemann.classification import MDM, FgMDM
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"pyriemann not installed — required for Riemannian classifiers: {str(e)}"
        )
    return {
        "Covariances": Covariances,
        "XdawnCovariances": XdawnCovariances,
        "TangentSpace": TangentSpace,
        "MDM": MDM,
        "FgMDM": FgMDM,
    }
    


def _build_riemann_ts(config: dict) -> Pipeline:
    """
    Covariances -> tangent space projection -> logistic regression.
    Projecting SPD matrices into the tangent space makes them euclidean,
    so any standard classifier works downstream. This is the strongest
    general-purpose Riemannian pipeline and the usual MOABB benchmark.
    """
    pr = _import_pyriemann()
    return Pipeline([
        ("cov", pr["Covariances"](estimator=config.get("cov_estimator", "oas"))),
        ("ts", pr["TangentSpace"](metric=config.get("metric", "riemann"))),
        ("lr", LogisticRegression(
            C=config.get("C", 1.0),
            max_iter=config.get("max_iter", 1000),
        )),
    ])


def _build_riemann_mdm(config: dict) -> Pipeline:
    """
    Minimum Distance to Mean — computes a Riemannian mean covariance per class,
    then assigns each trial to the nearest one. No trained classifier at all,
    which makes it very robust with few trials and effectively free to fit.
    """
    pr = _import_pyriemann()
    return Pipeline([
        ("cov", pr["Covariances"](estimator=config.get("cov_estimator", "oas"))),
        ("mdm", pr["MDM"](metric=config.get("metric", "riemann"))),
    ])


def _build_riemann_fgmdm(config: dict) -> Pipeline:
    """
    MDM preceded by Fisher geodesic filtering — a supervised dimensionality
    reduction in tangent space before classification. Usually beats plain MDM.
    """
    pr = _import_pyriemann()
    return Pipeline([
        ("cov", pr["Covariances"](estimator=config.get("cov_estimator", "oas"))),
        ("fgmdm", pr["FgMDM"](metric=config.get("metric", "riemann"))),
    ])


def _build_xdawn_ts(config: dict) -> Pipeline:
    """
    XdawnCovariances + tangent space — the standard pipeline for ERP/P300 data,
    where the discriminative signal is a time-locked evoked response rather than
    a band-power change. Xdawn learns spatial filters that maximise the
    evoked-response SNR before covariances are computed.

    Pairs with your ERP CORE / P300 datasets, not motor imagery.
    """
    pr = _import_pyriemann()
    return Pipeline([
        ("xdawn_cov", pr["XdawnCovariances"](
            nfilter=config.get("nfilter", 4),
            estimator=config.get("cov_estimator", "oas"),
            # pyriemann 0.12 docstring: xdawn_estimator "should be regularized using 'lwf' or
            # 'oas'". The unregularised 'scm' crashed on ERP CORE and LIMO ("leading minor ...
            # is not positive definite"): re-referencing + ICA leave the data rank-deficient.
            # Not exposed in the UI's PARAM_FIELDS, so no frontend default to keep in sync.
            xdawn_estimator=config.get("xdawn_estimator", "oas"),
        )),
        ("ts", pr["TangentSpace"](metric=config.get("metric", "riemann"))),
        ("lr", LogisticRegression(
            C=config.get("C", 1.0),
            max_iter=config.get("max_iter", 1000),
        )),
    ])





FEATURE_CLASSIFIERS = {
    "lda": _build_lda,
    "svm": _build_svm,
    "rf": _build_rf,
    "logreg": _build_logreg,
    "extra_trees": _build_extra_trees,
    "mlp": _build_mlp,
    "hgb": _build_hgb,
    "knn": _build_knn,
    "gnb": _build_gnb,
}

EPOCH_PIPELINES = {
    "riemann_ts": _build_riemann_ts,
    "riemann_mdm": _build_riemann_mdm,
    "riemann_fgmdm": _build_riemann_fgmdm,
    "xdawn_ts": _build_xdawn_ts,
}

DEEP_MODELS = {
    "eegnet": _build_eegnet,
}

SCALE_SENSITIVE = {"svm", "knn", "logreg", "mlp"}


def _get_category(name: str) -> str:
    if name in DEEP_MODELS:
        return "deep"
    if name in EPOCH_PIPELINES:
        return "epoch"
    if name in FEATURE_CLASSIFIERS:
        return "feature"
    raise HTTPException(status_code=400, detail=f"Unknown classifier: {name}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_classifier(
    classifier_name: str,
    mode: str,
    config: dict,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    epochs_obj: Optional[mne.Epochs] = None,
    csp_config: Optional[dict] = None,
    cv_folds: int = 5,
    groups: Optional[np.ndarray] = None,
) -> dict:
    """
    Trains and cross-validates one classifier, returning accuracy metrics.

    Dispatch is by estimator category, not by mode:

      deep    — raw 3D epochs, builder needs shape info (EEGNet, braindecode)
      epoch   — raw 3D epochs, self-contained pipeline (Riemannian).
                Replaces CSP; never has CSP prepended.
      feature — 2D features. In motor_imagery mode, CSP is prepended and refit
                inside each CV fold to avoid leakage. Otherwise trains on the
                pre-extracted X from features.py.
    """
    try:
        category = _get_category(classifier_name)
        # groups = subject id per trial -> leave-one-subject-out (cross-subject transfer).
        # StratifiedKFold on pooled subjects would put the same person in train AND test
        # and report within-subject accuracy while claiming cross-subject (section 8).
        # https://scikit-learn.org/1.9/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
        if groups is not None:
            cv = LeaveOneGroupOut()
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # ── deep ──────────────────────────────────────────────────────────
        if category == "deep":
            if epochs_obj is None or y is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{classifier_name}' trains on raw epoch data — requires epochs_obj and labels."
                )
            data = epochs_obj.get_data(copy=False).astype(np.float32)
            data = data * 1e6
            
            # MAX_SAMPLES = 512
            # if data.shape[2] > MAX_SAMPLES:
            #     factor = int(np.ceil(data.shape[2] / MAX_SAMPLES))
            #     data = data[:, :, ::factor]
            #     sfreq_eff = float(epochs_obj.info["sfreq"]) / factor
            # else:
            #     sfreq_eff = float(epochs_obj.info["sfreq"])
                
            _, n_chans, n_times = data.shape

            # torch needs contiguous int64 class indices; sklearn tolerates
            # string labels but skorch does not
            y_fit = LabelEncoder().fit_transform(y).astype(np.int64)

            estimator = DEEP_MODELS[classifier_name](
                config,
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=len(np.unique(y_fit)),
                sfreq=float(epochs_obj.info["sfreq"]),
            )
            y_true = y_fit
            y_pred = cross_val_predict(estimator, data, y_fit, cv=cv, groups=groups, n_jobs=1)

        # ── epoch-level pipelines (Riemannian) ────────────────────────────
        elif category == "epoch":
            if epochs_obj is None or y is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{classifier_name}' builds covariance matrices from raw epochs — requires epochs_obj and labels."
                )
            estimator = EPOCH_PIPELINES[classifier_name](config)
            data = epochs_obj.get_data(copy=False)

            # FgMDM's Fisher geodesic filter fits an LDA with n_components = n_classes - 1
            # (hardcoded in pyriemann 0.12, FGDA._fit_lda) on the tangent space of the
            # channel covariance, which has c(c+1)/2 features. With too few channels for the
            # class count sklearn rejects it with an opaque message; say what's actually wrong.
            if classifier_name == "riemann_fgmdm":
                n_chans = data.shape[1]
                n_ts = n_chans * (n_chans + 1) // 2
                n_classes = len(np.unique(y))
                if n_ts < n_classes - 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"FgMDM can't run on {n_chans} channels with {n_classes} classes: its "
                               f"covariance features number {n_ts}, and it needs at least {n_classes - 1}. "
                               f"Use riemann_mdm or a classical decoder, or data with more channels."
                    )
            y_true = y
            y_pred = cross_val_predict(
                estimator, data, y, cv=cv, groups=groups, n_jobs=1
            )

        # ── feature classifiers ───────────────────────────────────────────
        else:
            clf = FEATURE_CLASSIFIERS[classifier_name](config)
            steps = []

            if mode == "motor_imagery":
                if epochs_obj is None or y is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Motor imagery classification requires epochs_obj and labels."
                    )
                csp_config = csp_config or {}
                steps.append(("CSP", CSP(
                    n_components=csp_config.get("n_components", 4),
                    reg=csp_config.get("reg"),
                    log=True,
                    norm_trace=False,
                )))
                data = epochs_obj.get_data(copy=False)
            else:
                if X is None or y is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{mode} mode classification requires pre-extracted features (X) and labels (y)."
                    )
                data = X

            # scaler sits INSIDE the pipeline so it refits per fold — same
            # leakage reasoning as CSP
            if classifier_name in SCALE_SENSITIVE:
                steps.append(("scaler", StandardScaler()))

            steps.append((classifier_name, clf))
            estimator = Pipeline(steps)
            
            y_true = y
            y_pred = cross_val_predict(estimator, data, y, cv=cv, groups=groups, n_jobs=1)

        splits = list(cv.split(data, y_true, groups))
        fold_scores = [float(accuracy_score(y_true[test_idx], y_pred[test_idx])) for _, test_idx in splits]
        scores = np.array(fold_scores)
        # Balanced accuracy = mean recall per class; chance is 1/k however lopsided the
        # classes are. Plain accuracy rewards always guessing the majority class: on ERP CORE
        # Flankers (~87% correct responses) a do-nothing model would score ~87%.
        # https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.balanced_accuracy_score.html
        bal_scores = np.array([float(balanced_accuracy_score(y_true[test_idx], y_pred[test_idx]))
                               for _, test_idx in splits])

        labels_sorted = sorted(set(y_true.tolist()))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

        return {
            "classifier": classifier_name,
            "category": category,
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "balanced_accuracy_mean": float(bal_scores.mean()),
            "balanced_accuracy_std": float(bal_scores.std()),
            "fold_scores": fold_scores,
            "n_folds": len(splits),
            "confusion_matrix": cm.tolist(),
            "confusion_labels": [int(l) for l in labels_sorted],
            # cross-subject: the subject held out in each fold, aligned with fold_scores
            **({"fold_groups": [int(groups[test_idx][0]) for _, test_idx in splits]}
               if groups is not None else {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classifier '{classifier_name}' failed: {str(e)}"
        )


# ── Run multiple classifiers for comparison ───────────────────────────────────

def run_all_classifiers(
    selected_classifiers: list,
    mode: str,
    configs: dict,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    epochs_obj: Optional[mne.Epochs] = None,
    csp_config: Optional[dict] = None,
    cv_folds: int = 5,
    progress=None,
    groups: Optional[np.ndarray] = None,
) -> list:
    """
    Runs each selected classifier and returns a list of results for comparison.
    Continues past individual classifier failures rather than failing the whole batch —
    one bad classifier shouldn't block seeing results from the others.
    `progress(msg)` (optional) is told which decoder is training, for job status.
    """
    results = []
    for i, clf_name in enumerate(selected_classifiers, start=1):
        if progress:
            progress(f"Training {clf_name} ({i} of {len(selected_classifiers)})")
        try:
            clf_config = configs.get(clf_name, {})
            result = run_classifier(
                classifier_name=clf_name,
                mode=mode,
                config=clf_config,
                X=X,
                y=y,
                epochs_obj=epochs_obj,
                csp_config=csp_config,
                cv_folds=cv_folds,
                groups=groups,
            )
            results.append(result)
        except HTTPException as e:
            results.append({
                "classifier": clf_name,
                "error": e.detail,
                "accuracy_mean": None,
            })

    return results




def build_voting_ensemble(selected: list, configs: dict, voting: str = "soft"):
    """
    Combines the user's selected FEATURE classifiers into a single voting model.
    Epoch/deep models are excluded — they consume 3D input and can't share a
    pipeline with feature classifiers.

    'soft' voting averages predicted probabilities and usually beats 'hard'
    majority voting, but requires every member to expose predict_proba.
    """
    members = []
    for name in selected:
        if name not in FEATURE_CLASSIFIERS:
            continue
        clf = FEATURE_CLASSIFIERS[name](configs.get(name, {}))
        if voting == "soft" and not hasattr(clf, "predict_proba"):
            continue
        members.append((name, clf))

    if len(members) < 2:
        raise HTTPException(
            status_code=400,
            detail="Voting ensemble needs at least two compatible feature classifiers."
        )
    return VotingClassifier(estimators=members, voting=voting)
















# import numpy as np
# import mne
# from typing import Optional, Union
# from fastapi import HTTPException

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from mne.decoding import CSP

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     ExtraTreesClassifier,
#     HistGradientBoostingClassifier,
#     VotingClassifier,
# )
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler, LabelEncoder



# # ── Individual classifier builders ────────────────────────────────────────────

# def _build_lda(config: dict) -> LinearDiscriminantAnalysis:
#     solver = config.get("solver", "svd")
#     # shrinkage is only valid with 'lsqr'/'eigen' — sklearn raises if paired with 'svd'
#     shrinkage = config.get("shrinkage", "auto") if solver != "svd" else None
#     return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)


# def _build_svm(config: dict) -> SVC:
#     return SVC(
#         kernel=config.get("kernel", "rbf"),
#         C=config.get("C", 1.0),
#         gamma=config.get("gamma", "scale"),
#         probability=True,  # needed if we ever want confidence scores, small perf cost
#     )


# def _build_rf(config: dict) -> RandomForestClassifier:
#     return RandomForestClassifier(
#         n_estimators=config.get("n_estimators", 100),
#         max_depth=config.get("max_depth"),
#         max_features=config.get("max_features", "sqrt"),
#         random_state=42,
#     )


# def _build_eegnet(
#     config: dict,
#     n_chans: int,
#     n_times: int,
#     n_outputs: int,
#     sfreq: float = 250.0,
# ):
#     """
#     Builds an EEGNet model + skorch wrapper for sklearn-compatible training.
#     Uses the real published architecture from braindecode, not a hand-rolled version.

#     sfreq comes from epochs.info, not config — a wrong sampling rate here
#     silently mis-sizes the temporal kernels (Sleep-EDF is 100 Hz, several
#     MOABB sets are 512).
#     """
#     try:
#         from braindecode.models import EEGNet
#         from braindecode.classifier import EEGClassifier
#         import torch
#     except ImportError as e:
#         raise HTTPException(status_code=500, detail=f"Braindecode/torch not available: {str(e)}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     clf = EEGClassifier(
#         module=EEGNet,
#         module__n_chans=n_chans,
#         module__n_outputs=n_outputs,
#         module__n_times=n_times,
#         module__sfreq=sfreq,
#         module__drop_prob=config.get("dropout_rate", 0.25),
#         module__pool_mode=config.get("pool_mode", "mean"),
#         max_epochs=config.get("epochs", 50),
#         optimizer=torch.optim.AdamW,
#         optimizer__lr=config.get("learning_rate", 0.01),
#         optimizer__weight_decay=config.get("weight_decay", 1e-4),
#         batch_size=config.get("batch_size", 32),
#         device=device,
#         train_split=None,
#     )
#     return clf


# def _build_logreg(config: dict) -> LogisticRegression:
#     return LogisticRegression(
#         C=config.get("C", 1.0),
#         penalty=config.get("penalty", "l2"),
#         solver=config.get("solver", "lbfgs"),
#         max_iter=config.get("max_iter", 1000),
#     )


# def _build_extra_trees(config: dict) -> ExtraTreesClassifier:
#     return ExtraTreesClassifier(
#         n_estimators=config.get("n_estimators", 100),
#         max_depth=config.get("max_depth"),
#         max_features=config.get("max_features", "sqrt"),
#         random_state=42,
#     )


# def _build_mlp(config: dict) -> MLPClassifier:
#     """
#     Shallow fully-connected net on extracted features. Deliberately kept as a
#     contrast to EEGNet: same 'neural network' label, but operating on
#     hand-engineered features rather than raw signal.
#     """
#     return MLPClassifier(
#         hidden_layer_sizes=tuple(config.get("hidden_layer_sizes", [64, 32])),
#         activation=config.get("activation", "relu"),
#         alpha=config.get("alpha", 1e-4),
#         learning_rate_init=config.get("learning_rate_init", 1e-3),
#         max_iter=config.get("max_iter", 500),
#         early_stopping=config.get("early_stopping", True),
#         random_state=42,
#     )


# def _build_hgb(config: dict) -> HistGradientBoostingClassifier:
#     return HistGradientBoostingClassifier(
#         max_iter=config.get("max_iter", 100),
#         learning_rate=config.get("learning_rate", 0.1),
#         max_depth=config.get("max_depth"),
#         random_state=42,
#     )


# def _build_knn(config: dict) -> KNeighborsClassifier:
#     return KNeighborsClassifier(
#         n_neighbors=config.get("n_neighbors", 5),
#         weights=config.get("weights", "uniform"),
#         metric=config.get("metric", "minkowski"),
#     )


# def _build_gnb(config: dict) -> GaussianNB:
#     return GaussianNB()



# # ── Riemann model ──────────────────────────────────────────────────────────
# def _import_pyriemann():
#     try:
#         from pyriemann.estimation import Covariances, XdawnCovariances
#         from pyriemann.tangentspace import TangentSpace
#         from pyriemann.classification import MDM, FgMDM
#     except ImportError as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"pyriemann not installed — required for Riemannian classifiers: {str(e)}"
#         )
#     return {
#         "Covariances": Covariances,
#         "XdawnCovariances": XdawnCovariances,
#         "TangentSpace": TangentSpace,
#         "MDM": MDM,
#         "FgMDM": FgMDM,
#     }
    


# def _build_riemann_ts(config: dict) -> Pipeline:
#     """
#     Covariances -> tangent space projection -> logistic regression.
#     Projecting SPD matrices into the tangent space makes them euclidean,
#     so any standard classifier works downstream. This is the strongest
#     general-purpose Riemannian pipeline and the usual MOABB benchmark.
#     """
#     pr = _import_pyriemann()
#     return Pipeline([
#         ("cov", pr["Covariances"](estimator=config.get("cov_estimator", "oas"))),
#         ("ts", pr["TangentSpace"](metric=config.get("metric", "riemann"))),
#         ("lr", LogisticRegression(
#             C=config.get("C", 1.0),
#             max_iter=config.get("max_iter", 1000),
#         )),
#     ])


# def _build_riemann_mdm(config: dict) -> Pipeline:
#     """
#     Minimum Distance to Mean — computes a Riemannian mean covariance per class,
#     then assigns each trial to the nearest one. No trained classifier at all,
#     which makes it very robust with few trials and effectively free to fit.
#     """
#     pr = _import_pyriemann()
#     return Pipeline([
#         ("cov", pr["Covariances"](estimator=config.get("cov_estimator", "oas"))),
#         ("mdm", pr["MDM"](metric=config.get("metric", "riemann"))),
#     ])


# def _build_riemann_fgmdm(config: dict) -> Pipeline:
#     """
#     MDM preceded by Fisher geodesic filtering — a supervised dimensionality
#     reduction in tangent space before classification. Usually beats plain MDM.
#     """
#     pr = _import_pyriemann()
#     return Pipeline([
#         ("cov", pr["Covariances"](estimator=config.get("cov_estimator", "oas"))),
#         ("fgmdm", pr["FgMDM"](metric=config.get("metric", "riemann"))),
#     ])


# def _build_xdawn_ts(config: dict) -> Pipeline:
#     """
#     XdawnCovariances + tangent space — the standard pipeline for ERP/P300 data,
#     where the discriminative signal is a time-locked evoked response rather than
#     a band-power change. Xdawn learns spatial filters that maximise the
#     evoked-response SNR before covariances are computed.

#     Pairs with your ERP CORE / P300 datasets, not motor imagery.
#     """
#     pr = _import_pyriemann()
#     return Pipeline([
#         ("xdawn_cov", pr["XdawnCovariances"](
#             nfilter=config.get("nfilter", 4),
#             estimator=config.get("cov_estimator", "oas"),
#             xdawn_estimator=config.get("xdawn_estimator", "scm"),
#         )),
#         ("ts", pr["TangentSpace"](metric=config.get("metric", "riemann"))),
#         ("lr", LogisticRegression(
#             C=config.get("C", 1.0),
#             max_iter=config.get("max_iter", 1000),
#         )),
#     ])





# FEATURE_CLASSIFIERS = {
#     "lda": _build_lda,
#     "svm": _build_svm,
#     "rf": _build_rf,
#     "logreg": _build_logreg,
#     "extra_trees": _build_extra_trees,
#     "mlp": _build_mlp,
#     "hgb": _build_hgb,
#     "knn": _build_knn,
#     "gnb": _build_gnb,
# }

# EPOCH_PIPELINES = {
#     "riemann_ts": _build_riemann_ts,
#     "riemann_mdm": _build_riemann_mdm,
#     "riemann_fgmdm": _build_riemann_fgmdm,
#     "xdawn_ts": _build_xdawn_ts,
# }

# DEEP_MODELS = {
#     "eegnet": _build_eegnet,
# }

# SCALE_SENSITIVE = {"svm", "knn", "logreg", "mlp"}


# def _get_category(name: str) -> str:
#     if name in DEEP_MODELS:
#         return "deep"
#     if name in EPOCH_PIPELINES:
#         return "epoch"
#     if name in FEATURE_CLASSIFIERS:
#         return "feature"
#     raise HTTPException(status_code=400, detail=f"Unknown classifier: {name}")


# # ── Main entry point ──────────────────────────────────────────────────────────

# def run_classifier(
#     classifier_name: str,
#     mode: str,
#     config: dict,
#     X: Optional[np.ndarray] = None,
#     y: Optional[np.ndarray] = None,
#     epochs_obj: Optional[mne.Epochs] = None,
#     csp_config: Optional[dict] = None,
#     cv_folds: int = 5,
# ) -> dict:
#     """
#     Trains and cross-validates one classifier, returning accuracy metrics.

#     Dispatch is by estimator category, not by mode:

#       deep    — raw 3D epochs, builder needs shape info (EEGNet, braindecode)
#       epoch   — raw 3D epochs, self-contained pipeline (Riemannian).
#                 Replaces CSP; never has CSP prepended.
#       feature — 2D features. In motor_imagery mode, CSP is prepended and refit
#                 inside each CV fold to avoid leakage. Otherwise trains on the
#                 pre-extracted X from features.py.
#     """
#     try:
#         category = _get_category(classifier_name)
#         cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

#         # ── deep ──────────────────────────────────────────────────────────
#         if category == "deep":
#             if epochs_obj is None or y is None:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"'{classifier_name}' trains on raw epoch data — requires epochs_obj and labels."
#                 )
#             data = epochs_obj.get_data(copy=False).astype(np.float32)
#             data = data * 1e6
            
#             MAX_SAMPLES = 512
#             if data.shape[2] > MAX_SAMPLES:
#                 factor = int(np.ceil(data.shape[2] / MAX_SAMPLES))
#                 data = data[:, :, ::factor]
#                 sfreq_eff = float(epochs_obj.info["sfreq"]) / factor
#             else:
#                 sfreq_eff = float(epochs_obj.info["sfreq"])
            
#             _, n_chans, n_times = data.shape
            
            
#             print(f"[eegnet] shape={data.shape} std={data.std():.3f} "
#                 f"lr={config.get('learning_rate', 'DEFAULT')} "
#                 f"epochs={config.get('epochs', 'DEFAULT')}")

#             # torch needs contiguous int64 class indices; sklearn tolerates
#             # string labels but skorch does not
#             y_fit = LabelEncoder().fit_transform(y).astype(np.int64)

#             estimator = DEEP_MODELS[classifier_name](
#                 config,
#                 n_chans=n_chans,
#                 n_times=n_times,
#                 n_outputs=len(np.unique(y_fit)),
#                 sfreq=float(epochs_obj.info["sfreq"]),
#             )
#             scores = cross_val_score(estimator, data, y_fit, cv=cv, n_jobs=1)

#         # ── epoch-level pipelines (Riemannian) ────────────────────────────
#         elif category == "epoch":
#             if epochs_obj is None or y is None:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"'{classifier_name}' builds covariance matrices from raw epochs — requires epochs_obj and labels."
#                 )
#             estimator = EPOCH_PIPELINES[classifier_name](config)
#             scores = cross_val_score(
#                 estimator, epochs_obj.get_data(copy=False), y, cv=cv, n_jobs=1
#             )

#         # ── feature classifiers ───────────────────────────────────────────
#         else:
#             clf = FEATURE_CLASSIFIERS[classifier_name](config)
#             steps = []

#             if mode == "motor_imagery":
#                 if epochs_obj is None or y is None:
#                     raise HTTPException(
#                         status_code=400,
#                         detail="Motor imagery classification requires epochs_obj and labels."
#                     )
#                 csp_config = csp_config or {}
#                 steps.append(("CSP", CSP(
#                     n_components=csp_config.get("n_components", 4),
#                     reg=csp_config.get("reg"),
#                     log=True,
#                     norm_trace=False,
#                 )))
#                 data = epochs_obj.get_data(copy=False)
#             else:
#                 if X is None or y is None:
#                     raise HTTPException(
#                         status_code=400,
#                         detail=f"{mode} mode classification requires pre-extracted features (X) and labels (y)."
#                     )
#                 data = X

#             # scaler sits INSIDE the pipeline so it refits per fold — same
#             # leakage reasoning as CSP
#             if classifier_name in SCALE_SENSITIVE:
#                 steps.append(("scaler", StandardScaler()))

#             steps.append((classifier_name, clf))
#             estimator = Pipeline(steps)
#             scores = cross_val_score(estimator, data, y, cv=cv, n_jobs=1)

#         return {
#             "classifier": classifier_name,
#             "category": category,
#             "accuracy_mean": float(scores.mean()),
#             "accuracy_std": float(scores.std()),
#             "fold_scores": [float(s) for s in scores],
#             "n_folds": cv_folds,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Classifier '{classifier_name}' failed: {str(e)}"
#         )


# # ── Run multiple classifiers for comparison ───────────────────────────────────

# def run_all_classifiers(
#     selected_classifiers: list,
#     mode: str,
#     configs: dict,
#     X: Optional[np.ndarray] = None,
#     y: Optional[np.ndarray] = None,
#     epochs_obj: Optional[mne.Epochs] = None,
#     csp_config: Optional[dict] = None,
#     cv_folds: int = 5,
# ) -> list:
#     """
#     Runs each selected classifier and returns a list of results for comparison.
#     Continues past individual classifier failures rather than failing the whole batch —
#     one bad classifier shouldn't block seeing results from the others.
#     """
#     results = []
#     for clf_name in selected_classifiers:
#         try:
#             clf_config = configs.get(clf_name, {})
#             result = run_classifier(
#                 classifier_name=clf_name,
#                 mode=mode,
#                 config=clf_config,
#                 X=X,
#                 y=y,
#                 epochs_obj=epochs_obj,
#                 csp_config=csp_config,
#                 cv_folds=cv_folds,
#             )
#             results.append(result)
#         except HTTPException as e:
#             results.append({
#                 "classifier": clf_name,
#                 "error": e.detail,
#                 "accuracy_mean": None,
#             })

#     return results




# def build_voting_ensemble(selected: list, configs: dict, voting: str = "soft"):
#     """
#     Combines the user's selected FEATURE classifiers into a single voting model.
#     Epoch/deep models are excluded — they consume 3D input and can't share a
#     pipeline with feature classifiers.

#     'soft' voting averages predicted probabilities and usually beats 'hard'
#     majority voting, but requires every member to expose predict_proba.
#     """
#     members = []
#     for name in selected:
#         if name not in FEATURE_CLASSIFIERS:
#             continue
#         clf = FEATURE_CLASSIFIERS[name](configs.get(name, {}))
#         if voting == "soft" and not hasattr(clf, "predict_proba"):
#             continue
#         members.append((name, clf))

#     if len(members) < 2:
#         raise HTTPException(
#             status_code=400,
#             detail="Voting ensemble needs at least two compatible feature classifiers."
#         )
#     return VotingClassifier(estimators=members, voting=voting)





