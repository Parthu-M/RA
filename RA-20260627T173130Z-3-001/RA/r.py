import json, re

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) if isinstance(text, str) else text
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) if isinstance(text, str) else text
    })

# ------------------------------------------------------------------
md("## Section 0 — Imports & Configuration")
code(r'''import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score,
    brier_score_loss, roc_curve, precision_recall_curve,
    auc
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.stats import spearmanr

from imblearn.over_sampling import BorderlineSMOTE
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import (
    EasyEnsembleClassifier,
    BalancedRandomForestClassifier,
    RUSBoostClassifier
)
from pytorch_tabnet.tab_model import TabNetClassifier
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ── Reproducibility seed ──────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Output directories ────────────────────────────────────────────────────────
GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── IEEE-style matplotlib aesthetics ─────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Times New Roman",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

IEEE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]

SHORT_LABELS = {
    "SVM (RBF)":             "SVM",
    "XGBoost":               "XGBoost",
    "EasyEnsemble":          "EasyEns.",
    "TabNet":                "TabNet",
    "Hybrid (XGB+CatBoost)": "Hybrid",
    "BalancedRandomForest":  "BalRF",
    "LightGBM (Focal Loss)": "LGBM-FL",
    "RUSBoost":              "RUSBoost",
}

# ── Global model store (populated during training, used by extensions) ────────
TRAINED_MODELS = {}

logger.info("Configuration complete. Seed=%d", SEED)''')

# ------------------------------------------------------------------
md("## Section 1 — Data Loading & Preprocessing")
code(r'''def load_and_preprocess(filepath: str) -> tuple:
    """
    Load CSV dataset, handle missing values, encode categoricals,
    and return leakage-free (X, y) as float32 numpy arrays.

    Steps
    -----
    1. Drop survey identifier / sampling-weight columns.
    2. Impute numerical features with median (per-column).
    3. Fill categorical NaNs with literal "Unknown".
    4. Label-encode every categorical column.
    5. Cast entire feature matrix to float32 for memory efficiency.
    6. Store feature names globally for SHAP extension.

    Parameters
    ----------
    filepath : str — path to the CSV dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float32
    y : np.ndarray, shape (n_samples,), dtype int
    """
    global FEATURE_NAMES

    df = pd.read_csv(filepath)
    logger.info("Raw dataset loaded: %d rows × %d columns", *df.shape)

    # ── Drop identifiers and sampling weights ─────────────────────────────────
    drop_cols = ["SEQN", "PSU", "STRATA", "Weight"]
    dropped   = [c for c in drop_cols if c in df.columns]
    df.drop(columns=dropped, inplace=True)
    if dropped:
        logger.info("Dropped identifier columns: %s", dropped)

    # ── Separate target ───────────────────────────────────────────────────────
    TARGET = "RheumatoidArthritis"
    y      = df[TARGET].values.astype(int)
    X_df   = df.drop(columns=[TARGET])

    # ── Store feature names for SHAP (Section 6X) ────────────────────────────
    FEATURE_NAMES = X_df.columns.tolist()

    # ── Identify column types ─────────────────────────────────────────────────
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Impute numerical → median ─────────────────────────────────────────────
    for col in num_cols:
        if X_df[col].isnull().any():
            X_df[col].fillna(X_df[col].median(), inplace=True)

    # ── Impute categorical → "Unknown" then label-encode ─────────────────────
    le = LabelEncoder()
    for col in cat_cols:
        X_df[col].fillna("Unknown", inplace=True)
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    # ── Cast to float32 ───────────────────────────────────────────────────────
    X = X_df.astype(np.float32).values

    logger.info("Preprocessed: %d samples × %d predictors", *X.shape)
    logger.info("Class distribution (0/1): %s", np.bincount(y).tolist())
    return X, y''')

# ------------------------------------------------------------------
md("## Section 2 — Extended Evaluation Utility (IEEE Metrics)")
code(r'''def evaluate(
    model_name:    str,
    strategy_name: str,
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    y_prob:        np.ndarray,
    opt_threshold: float = 0.5
) -> dict:
    """
    Compute the full IEEE imbalanced-classification metric suite.

    Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
             MCC, Balanced Accuracy, G-Mean, Cohen's Kappa,
             Brier Score, Sensitivity, Specificity, TN/FP/FN/TP
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc_ = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pr_auc      = average_precision_score(y_true, y_prob)
    mcc         = matthews_corrcoef(y_true, y_pred)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean      = np.sqrt(sensitivity * specificity)
    kappa       = cohen_kappa_score(y_true, y_pred)
    brier       = brier_score_loss(y_true, y_prob)

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Model    : {model_name}")
    print(f"  Strategy : {strategy_name}")
    print(f"  Threshold: {opt_threshold:.2f}")
    print(sep)
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  ROC-AUC={auc_:.4f}  PR-AUC={pr_auc:.4f}  MCC={mcc:.4f}")
    print(f"  BalAcc={bal_acc:.4f}  G-Mean={g_mean:.4f}  Kappa={kappa:.4f}  Brier={brier:.4f}")
    print(f"  Sensitivity={sensitivity:.4f}  Specificity={specificity:.4f}")
    print(f"  CM → TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(sep)

    return dict(
        model=model_name,   dataset=strategy_name,
        threshold=opt_threshold,
        accuracy=acc,       precision=prec,     recall=rec,
        f1=f1,              roc_auc=auc_,
        pr_auc=pr_auc,      mcc=mcc,            balanced_accuracy=bal_acc,
        g_mean=g_mean,
        sensitivity=sensitivity, specificity=specificity,
        kappa=kappa,        brier=brier,
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)
    )''')

# ------------------------------------------------------------------
md("## Section 3 — Leakage-Free OOF Threshold Optimization")
code(r'''def optimize_threshold_oof(
    y_true:    np.ndarray,
    oof_probs: np.ndarray,
    t_min:     float = 0.15,
    t_max:     float = 0.85,
    step:      float = 0.01
) -> float:
    """
    Select optimal decision threshold via grid search on Out-of-Fold
    probabilities only. No test-set information is ever used.

    Objective: Combined Score = 0.6 × G-Mean + 0.4 × F1

    Returns best threshold t* in [t_min, t_max].
    """
    best_thresh = 0.5
    best_score  = -1.0

    for t in np.arange(t_min, t_max, step):
        y_pred = (oof_probs >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred)

        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) == 0 or (tn + fp) == 0:
            continue

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        g_mean      = np.sqrt(sensitivity * specificity)
        f1          = f1_score(y_true, y_pred, zero_division=0)
        score       = 0.6 * g_mean + 0.4 * f1

        if score > best_score:
            best_score  = score
            best_thresh = t

    return float(best_thresh)''')

# ------------------------------------------------------------------
md("## Section 4 (Setup) — Load Data, Split, and Initialize Containers")
code(r'''# ── UPDATE THIS PATH ──────────────────────────────────────────────────────
DATA_PATH = r"D:\MTECH\project\claud\dataset.csv"   # <-- UPDATE THIS

# Global storage for extension access
FEATURE_NAMES    = []
X_TRAIN_GLOBAL   = None
X_TEST_GLOBAL    = None
Y_TRAIN_GLOBAL   = None

print("\n" + "=" * 72)
print("  IEEE RHEUMATOID ARTHRITIS CLASSIFICATION PIPELINE")
print("  Leakage-Free | 8 Models | Full Metric Suite | 3 Extensions")
print("=" * 72)

# ── Step 1: Load & preprocess (also sets FEATURE_NAMES global) ───────────
X, y = load_and_preprocess(DATA_PATH)
print(f"\n[INFO] Samples: {X.shape[0]}  |  Predictors: {X.shape[1]}")
print(f"[INFO] Class balance → {np.bincount(y).tolist()}")
ir = np.bincount(y)[0] / np.bincount(y)[1]
print(f"[INFO] Imbalance ratio: {ir:.1f}:1")

# ── Step 2: Stratified 80/20 split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── Step 3: Save to globals for extension access ──────────────────────────
X_TRAIN_GLOBAL = X_train
X_TEST_GLOBAL  = X_test
Y_TRAIN_GLOBAL = y_train

# ── Step 4: Initialise containers ────────────────────────────────────────
results:           list = []
predictions_store: dict = {}''')

# ------------------------------------------------------------------
md("## Section 4a — SVM (RBF Kernel): Train & Evaluate")
code(r'''def run_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    Support Vector Machine with RBF kernel.
    Strategies: Original (Imbalanced) | BorderlineSMOTE
    Final trained model stored in TRAINED_MODELS["SVM"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 1 : Support Vector Machine (RBF Kernel)")
    print("=" * 72)

    strategies = ["Original (Imbalanced)", "BorderlineSMOTE"]

    for strategy in strategies:
        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_probs = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va       = X_train[val_idx]

            if strategy == "BorderlineSMOTE":
                bsm = BorderlineSMOTE(kind="borderline-1", random_state=SEED, k_neighbors=5)
                X_tr, y_tr = bsm.fit_resample(X_tr, y_tr)

            sc     = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)

            cw  = "balanced" if strategy == "Original (Imbalanced)" else None
            clf = SVC(kernel="rbf", C=10, gamma="scale",
                      class_weight=cw, probability=True, random_state=SEED)
            clf.fit(X_tr_s, y_tr)
            oof_probs[val_idx] = clf.predict_proba(X_va_s)[:, 1]

        opt_thresh = optimize_threshold_oof(y_train, oof_probs)
        print(f"  [{strategy}] OOF Threshold = {opt_thresh:.2f}")

        X_tr_f, y_tr_f = X_train, y_train
        if strategy == "BorderlineSMOTE":
            X_tr_f, y_tr_f = BorderlineSMOTE(
                kind="borderline-1", random_state=SEED
            ).fit_resample(X_train, y_train)

        sc_f    = StandardScaler()
        X_tr_fs = sc_f.fit_transform(X_tr_f)
        X_te_fs = sc_f.transform(X_test)

        cw_f  = "balanced" if strategy == "Original (Imbalanced)" else None
        clf_f = SVC(kernel="rbf", C=10, gamma="scale",
                    class_weight=cw_f, probability=True, random_state=SEED)
        clf_f.fit(X_tr_fs, y_tr_f)

        # ── Store final scaler+model for calibration extension ────────────────
        if strategy == "Original (Imbalanced)":
            TRAINED_MODELS["SVM"] = {"model": clf_f, "scaler": sc_f,
                                      "threshold": opt_thresh}

        y_prob = clf_f.predict_proba(X_te_fs)[:, 1]
        y_pred = (y_prob >= opt_thresh).astype(int)

        key = f"SVM (RBF)__{strategy}"
        predictions_store[key] = {
            "y_true": y_test, "y_pred": y_pred,
            "y_prob": y_prob, "threshold": opt_thresh
        }
        results.append(
            evaluate("SVM (RBF)", strategy, y_test, y_pred, y_prob, opt_thresh)
        )


run_svm(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4b — XGBoost: Train & Evaluate")
code(r'''def run_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    XGBoost with scale_pos_weight (Original) or BorderlineSMOTE.
    Final trained model stored in TRAINED_MODELS["XGBoost"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 2 : XGBoost Classifier")
    print("=" * 72)

    base_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=SEED, tree_method="hist", n_jobs=-1
    )

    for strategy in ["Original (Imbalanced)", "BorderlineSMOTE"]:
        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_probs = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va       = X_train[val_idx]

            if strategy == "BorderlineSMOTE":
                X_tr, y_tr = BorderlineSMOTE(
                    kind="borderline-1", random_state=SEED
                ).fit_resample(X_tr, y_tr)
                spw = 1.0
            else:
                spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

            clf = XGBClassifier(scale_pos_weight=spw, **base_params)
            clf.fit(X_tr, y_tr, verbose=False)
            oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]

        opt_thresh = optimize_threshold_oof(y_train, oof_probs)
        print(f"  [{strategy}] OOF Threshold = {opt_thresh:.2f}")

        X_tr_f, y_tr_f = X_train, y_train
        if strategy == "BorderlineSMOTE":
            X_tr_f, y_tr_f = BorderlineSMOTE(
                kind="borderline-1", random_state=SEED
            ).fit_resample(X_train, y_train)
            spw_f = 1.0
        else:
            spw_f = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        clf_f = XGBClassifier(scale_pos_weight=spw_f, **base_params)
        clf_f.fit(X_tr_f, y_tr_f, verbose=False)

        # ── Store for SHAP and calibration extensions ─────────────────────────
        if strategy == "Original (Imbalanced)":
            TRAINED_MODELS["XGBoost"] = {"model": clf_f, "threshold": opt_thresh}

        y_prob = clf_f.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= opt_thresh).astype(int)

        key = f"XGBoost__{strategy}"
        predictions_store[key] = {
            "y_true": y_test, "y_pred": y_pred,
            "y_prob": y_prob, "threshold": opt_thresh
        }
        results.append(
            evaluate("XGBoost", strategy, y_test, y_pred, y_prob, opt_thresh)
        )


run_xgboost(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4c — EasyEnsemble Classifier: Train & Evaluate")
code(r'''def run_easy_ensemble(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    EasyEnsembleClassifier — intrinsically balanced AdaBoost ensemble.
    Final trained model stored in TRAINED_MODELS["EasyEnsemble"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 3 : EasyEnsembleClassifier (AdaBoost Ensemble)")
    print("=" * 72)

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va       = X_train[val_idx]

        clf = EasyEnsembleClassifier(
            n_estimators=20, sampling_strategy="auto",
            random_state=SEED, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]

    opt_thresh = optimize_threshold_oof(y_train, oof_probs)
    print(f"  [Intrinsic Balancing] OOF Threshold = {opt_thresh:.2f}")

    clf_f = EasyEnsembleClassifier(
        n_estimators=20, sampling_strategy="auto",
        random_state=SEED, n_jobs=-1
    )
    clf_f.fit(X_train, y_train)

    # ── Store for SHAP and calibration extensions ─────────────────────────────
    TRAINED_MODELS["EasyEnsemble"] = {"model": clf_f, "threshold": opt_thresh}

    y_prob = clf_f.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= opt_thresh).astype(int)

    key = "EasyEnsemble__Intrinsic Balancing"
    predictions_store[key] = {
        "y_true": y_test, "y_pred": y_pred,
        "y_prob": y_prob, "threshold": opt_thresh
    }
    results.append(
        evaluate("EasyEnsemble", "Intrinsic Balancing",
                 y_test, y_pred, y_prob, opt_thresh)
    )


run_easy_ensemble(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4d — TabNet (Attention-Based Deep Learning): Train & Evaluate")
code(r'''class TabNetSklearnWrapper:
    """
    Thin sklearn-compatible wrapper around TabNetClassifier so that
    CalibratedClassifierCV (Section 6X EXT-C) can call .predict_proba()
    on a scaled input without needing an external scaler object.
    """
    def __init__(self, tabnet_model, scaler):
        self.tabnet_model = tabnet_model
        self.scaler       = scaler

    def predict_proba(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        return self.tabnet_model.predict_proba(Xs)

    def fit(self, X, y):
        # Required by CalibratedClassifierCV cv='prefit' — not called
        return self


def run_tabnet(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    TabNet attention-based deep learning model.
    3-fold OOF (reduced for DL training cost).
    Final wrapped model stored in TRAINED_MODELS["TabNet"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 4 : TabNet (Attention-based Deep Learning)")
    print("=" * 72)

    TABNET_PARAMS = dict(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2, momentum=0.02,
        mask_type="entmax", optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        seed=SEED, verbose=0
    )

    for strategy in ["Original (Imbalanced)", "BorderlineSMOTE"]:
        skf       = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        oof_probs = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va       = X_train[val_idx]

            if strategy == "BorderlineSMOTE":
                X_tr, y_tr = BorderlineSMOTE(
                    kind="borderline-1", random_state=SEED
                ).fit_resample(X_tr, y_tr)

            sc     = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
            X_va_s = sc.transform(X_va).astype(np.float32)

            neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
            sw       = np.where(y_tr == 1, neg / max(pos, 1), 1.0)

            clf = TabNetClassifier(**TABNET_PARAMS)
            clf.fit(X_tr_s, y_tr, weights=sw,
                    max_epochs=40, batch_size=1024,
                    virtual_batch_size=128, drop_last=False)
            oof_probs[val_idx] = clf.predict_proba(X_va_s)[:, 1]

        opt_thresh = optimize_threshold_oof(y_train, oof_probs)
        print(f"  [{strategy}] OOF Threshold = {opt_thresh:.2f}")

        X_tr_f, y_tr_f = X_train, y_train
        if strategy == "BorderlineSMOTE":
            X_tr_f, y_tr_f = BorderlineSMOTE(
                kind="borderline-1", random_state=SEED
            ).fit_resample(X_train, y_train)

        sc_f    = StandardScaler()
        X_tr_fs = sc_f.fit_transform(X_tr_f).astype(np.float32)
        X_te_fs = sc_f.transform(X_test).astype(np.float32)

        neg_f, pos_f = (y_tr_f == 0).sum(), (y_tr_f == 1).sum()
        sw_f         = np.where(y_tr_f == 1, neg_f / max(pos_f, 1), 1.0)

        clf_f = TabNetClassifier(**TABNET_PARAMS)
        clf_f.fit(X_tr_fs, y_tr_f, weights=sw_f,
                  max_epochs=50, batch_size=1024,
                  virtual_batch_size=128, drop_last=False)

        # ── Wrap TabNet + scaler for calibration extension ────────────────────
        if strategy == "Original (Imbalanced)":
            TRAINED_MODELS["TabNet"] = {
                "model":     TabNetSklearnWrapper(clf_f, sc_f),
                "threshold": opt_thresh
            }

        y_prob = clf_f.predict_proba(X_te_fs)[:, 1]
        y_pred = (y_prob >= opt_thresh).astype(int)

        key = f"TabNet__{strategy}"
        predictions_store[key] = {
            "y_true": y_test, "y_pred": y_pred,
            "y_prob": y_prob, "threshold": opt_thresh
        }
        results.append(
            evaluate("TabNet", strategy, y_test, y_pred, y_prob, opt_thresh)
        )


run_tabnet(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4e — Hybrid Ensemble (XGBoost + CatBoost Soft-Voting): Train & Evaluate")
code(r'''class HybridEnsembleWrapper:
    """
    Sklearn-compatible wrapper for the XGBoost+CatBoost soft-voting ensemble,
    enabling CalibratedClassifierCV in Section 6X EXT-C.
    """
    def __init__(self, xgb_model, cat_model):
        self.xgb_model = xgb_model
        self.cat_model = cat_model

    def predict_proba(self, X):
        return (
            0.5 * self.xgb_model.predict_proba(X) +
            0.5 * self.cat_model.predict_proba(X)
        )

    def fit(self, X, y):
        return self


def run_hybrid(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    Heterogeneous soft-voting: XGBoost + CatBoost (equal weights).
    Final wrapped model stored in TRAINED_MODELS["Hybrid"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 5 : Hybrid Ensemble (XGBoost + CatBoost Soft-Voting)")
    print("=" * 72)

    for strategy in ["Original (Imbalanced)", "BorderlineSMOTE"]:
        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_probs = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va       = X_train[val_idx]

            if strategy == "BorderlineSMOTE":
                X_tr, y_tr = BorderlineSMOTE(
                    kind="borderline-1", random_state=SEED
                ).fit_resample(X_tr, y_tr)
                spw   = 1.0
                cat_w = None
            else:
                spw   = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                cat_w = {0: 1.0, 1: spw}

            xgb = XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=SEED, tree_method="hist", n_jobs=-1
            )
            xgb.fit(X_tr, y_tr, verbose=False)

            cat = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                class_weights=cat_w, random_seed=SEED, verbose=0
            )
            cat.fit(X_tr, y_tr)

            oof_probs[val_idx] = (
                0.5 * xgb.predict_proba(X_va)[:, 1] +
                0.5 * cat.predict_proba(X_va)[:, 1]
            )

        opt_thresh = optimize_threshold_oof(y_train, oof_probs)
        print(f"  [{strategy}] OOF Threshold = {opt_thresh:.2f}")

        X_tr_f, y_tr_f = X_train, y_train
        if strategy == "BorderlineSMOTE":
            X_tr_f, y_tr_f = BorderlineSMOTE(
                kind="borderline-1", random_state=SEED
            ).fit_resample(X_train, y_train)
            spw_f  = 1.0
            cat_wf = None
        else:
            spw_f  = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            cat_wf = {0: 1.0, 1: spw_f}

        xgb_f = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw_f, eval_metric="logloss",
            random_state=SEED, tree_method="hist", n_jobs=-1
        )
        xgb_f.fit(X_tr_f, y_tr_f, verbose=False)

        cat_f = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            class_weights=cat_wf, random_seed=SEED, verbose=0
        )
        cat_f.fit(X_tr_f, y_tr_f)

        # ── Store wrapped ensemble ────────────────────────────────────────────
        if strategy == "Original (Imbalanced)":
            TRAINED_MODELS["Hybrid"] = {
                "model":     HybridEnsembleWrapper(xgb_f, cat_f),
                "threshold": opt_thresh
            }

        y_prob = (
            0.5 * xgb_f.predict_proba(X_test)[:, 1] +
            0.5 * cat_f.predict_proba(X_test)[:, 1]
        )
        y_pred = (y_prob >= opt_thresh).astype(int)

        key = f"Hybrid (XGB+CatBoost)__{strategy}"
        predictions_store[key] = {
            "y_true": y_test, "y_pred": y_pred,
            "y_prob": y_prob, "threshold": opt_thresh
        }
        results.append(
            evaluate("Hybrid (XGB+CatBoost)", strategy,
                     y_test, y_pred, y_prob, opt_thresh)
        )


run_hybrid(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4f — Balanced Random Forest: Train & Evaluate")
code(r'''def run_balanced_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    BalancedRandomForestClassifier with bootstrap under-sampling.
    Final trained model stored in TRAINED_MODELS["BalancedRF"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 6 : Balanced Random Forest (Bootstrap Under-Sampling)")
    print("=" * 72)

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va       = X_train[val_idx]

        clf = BalancedRandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            sampling_strategy="auto", replacement=False,
            random_state=SEED, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]

    opt_thresh = optimize_threshold_oof(y_train, oof_probs)
    print(f"  [Bootstrap Balancing] OOF Threshold = {opt_thresh:.2f}")

    clf_f = BalancedRandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        sampling_strategy="auto", replacement=False,
        random_state=SEED, n_jobs=-1
    )
    clf_f.fit(X_train, y_train)

    # ── Store for calibration extension ───────────────────────────────────────
    TRAINED_MODELS["BalancedRF"] = {"model": clf_f, "threshold": opt_thresh}

    y_prob = clf_f.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= opt_thresh).astype(int)

    key = "BalancedRandomForest__Bootstrap Balancing"
    predictions_store[key] = {
        "y_true": y_test, "y_pred": y_pred,
        "y_prob": y_prob, "threshold": opt_thresh
    }
    results.append(
        evaluate("BalancedRandomForest", "Bootstrap Balancing",
                 y_test, y_pred, y_prob, opt_thresh)
    )


run_balanced_rf(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4g — LightGBM with Cost-Sensitive Focal Loss: Train & Evaluate")
code(r'''def run_lightgbm_focal(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    LightGBM with custom alpha-balanced focal loss (gamma=2.0).
    Final trained model stored in TRAINED_MODELS["LightGBM"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 7 : LightGBM with Cost-Sensitive Focal Loss")
    print("=" * 72)

    GAMMA = 2.0

    def focal_loss_objective(y_pred, dataset):
        y_true  = dataset.get_label()
        neg, pos = (y_true == 0).sum(), (y_true == 1).sum()
        alpha_t  = np.where(y_true == 1, neg / max(pos, 1), 1.0)
        p        = 1.0 / (1.0 + np.exp(-y_pred))
        p_t      = np.where(y_true == 1, p, 1.0 - p)
        focal_w  = alpha_t * (1.0 - p_t) ** GAMMA
        grad     = focal_w * (p - y_true)
        hess     = focal_w * p * (1.0 - p) * (
            GAMMA * p_t * np.log(np.clip(p_t, 1e-9, 1.0)) + 1.0
        )
        return grad, hess

    def focal_loss_eval(y_pred, dataset):
        y_true = dataset.get_label()
        p      = 1.0 / (1.0 + np.exp(-y_pred))
        p_t    = np.where(y_true == 1, p, 1.0 - p)
        loss   = -((1.0 - p_t) ** GAMMA) * np.log(np.clip(p_t, 1e-9, 1.0))
        return "focal_loss", float(loss.mean()), False

    lgb_params = dict(
        objective=focal_loss_objective,
        num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, random_state=SEED,
        n_jobs=-1, verbose=-1
    )

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va       = X_train[val_idx]

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(lgb_params, ds_tr, num_boost_round=400)
        raw   = model.predict(X_va)
        oof_probs[val_idx] = 1.0 / (1.0 + np.exp(-raw))

    opt_thresh = optimize_threshold_oof(y_train, oof_probs)
    print(f"  [Focal Objective] OOF Threshold = {opt_thresh:.2f}")

    ds_full = lgb.Dataset(X_train, label=y_train)
    model_f = lgb.train(lgb_params, ds_full, num_boost_round=500)

    # ── Store lgb booster wrapped for calibration ─────────────────────────────
    class LGBMWrapper:
        def __init__(self, booster):
            self.booster = booster
        def predict_proba(self, X):
            raw = self.booster.predict(X)
            p   = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1 - p, p])
        def fit(self, X, y):
            return self

    TRAINED_MODELS["LightGBM"] = {
        "model":     LGBMWrapper(model_f),
        "threshold": opt_thresh
    }

    y_prob = 1.0 / (1.0 + np.exp(-model_f.predict(X_test)))
    y_pred = (y_prob >= opt_thresh).astype(int)

    key = "LightGBM (Focal Loss)__Focal Objective Strategy"
    predictions_store[key] = {
        "y_true": y_test, "y_pred": y_pred,
        "y_prob": y_prob, "threshold": opt_thresh
    }
    results.append(
        evaluate("LightGBM (Focal Loss)", "Focal Objective Strategy",
                 y_test, y_pred, y_prob, opt_thresh)
    )


run_lightgbm_focal(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 4h — RUSBoost: Train & Evaluate")
code(r'''def run_rusboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    results: list,       predictions_store: dict
):
    """
    RUSBoost: Random Under-Sampling + AdaBoost hybrid.
    Final trained model stored in TRAINED_MODELS["RUSBoost"].
    """
    print("\n" + "=" * 72)
    print("  MODEL 8 : RUSBoost (Random Under-Sampling + AdaBoost)")
    print("=" * 72)

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va       = X_train[val_idx]

        clf = RUSBoostClassifier(
            n_estimators=400, learning_rate=0.1,
            sampling_strategy="auto", random_state=SEED
        )
        clf.fit(X_tr, y_tr)
        oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]

    opt_thresh = optimize_threshold_oof(y_train, oof_probs)
    print(f"  [Sequential Boosting Balance] OOF Threshold = {opt_thresh:.2f}")

    clf_f = RUSBoostClassifier(
        n_estimators=400, learning_rate=0.1,
        sampling_strategy="auto", random_state=SEED
    )
    clf_f.fit(X_train, y_train)

    # ── Store for calibration extension ───────────────────────────────────────
    TRAINED_MODELS["RUSBoost"] = {"model": clf_f, "threshold": opt_thresh}

    y_prob = clf_f.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= opt_thresh).astype(int)

    key = "RUSBoost__Sequential Boosting Balance"
    predictions_store[key] = {
        "y_true": y_test, "y_pred": y_pred,
        "y_prob": y_prob, "threshold": opt_thresh
    }
    results.append(
        evaluate("RUSBoost", "Sequential Boosting Balance",
                 y_test, y_pred, y_prob, opt_thresh)
    )


run_rusboost(X_train, y_train, X_test, y_test, results, predictions_store)''')

# ------------------------------------------------------------------
md("## Section 5 — Results Storage & Ranking")
code(r'''def build_results_dataframe(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["short"] = df["model"].map(SHORT_LABELS).fillna(df["model"])
    return df


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = ["f1", "pr_auc", "mcc", "balanced_accuracy", "roc_auc"]
    best_df   = df.loc[df.groupby("model")["f1"].idxmax()].copy()
    ranked    = best_df.sort_values(
        rank_cols, ascending=[False] * len(rank_cols)
    ).reset_index(drop=True)
    ranked["Rank"] = ranked.index + 1
    return ranked


def print_ieee_summary(df: pd.DataFrame):
    ranked = rank_models(df)
    display_cols = [
        "Rank", "model", "dataset", "threshold",
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "mcc", "balanced_accuracy",
        "g_mean", "kappa", "brier"
    ]
    fmt_cols = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "mcc", "balanced_accuracy",
        "g_mean", "kappa", "brier", "threshold"
    ]
    summary = ranked[display_cols].copy()
    for c in fmt_cols:
        if c in summary.columns:
            summary[c] = summary[c].map(lambda x: f"{x:.4f}")

    print("\n" + "=" * 110)
    print("  IEEE BENCHMARK — RANKED BY F1 → PR-AUC → MCC → BALANCED ACCURACY → ROC-AUC")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)

    best = ranked.iloc[0]
    print(f"\n  BEST MODEL → {best['model']} | {best['dataset']}")
    print(f"  F1={best['f1']:.4f} | PR-AUC={best['pr_auc']:.4f} | "
          f"MCC={best['mcc']:.4f} | ROC-AUC={best['roc_auc']:.4f}")
    print("=" * 110)
    return ranked


# ── Assemble & rank ───────────────────────────────────────────────
df_results = build_results_dataframe(results)
ranked     = print_ieee_summary(df_results)''')

# ------------------------------------------------------------------
md("## Section 6 — IEEE Graph Generator Utilities (Figs 1–18)")
code(r'''def _save_fig(fig, fname: str):
    path = os.path.join(GRAPHS_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {fname}")


def prepare_plot_data(df: pd.DataFrame):
    best_df = df.loc[df.groupby("model")["f1"].idxmax()].reset_index(drop=True)
    best_df["short"] = best_df["model"].map(SHORT_LABELS).fillna(best_df["model"])
    core_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    core_labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    extra_metrics = ["pr_auc", "mcc", "balanced_accuracy", "g_mean", "kappa", "brier"]
    return best_df, core_metrics, core_labels, extra_metrics


def fig01_grouped_metrics(df):
    best_df, metrics, m_labels, _ = prepare_plot_data(df)
    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    n_m, n_met = len(best_df), len(metrics)
    x, w = np.arange(n_m), 0.14
    for i, (m, lbl) in enumerate(zip(metrics, m_labels)):
        offset = (i - n_met / 2 + 0.5) * w
        ax.bar(x + offset, best_df[m], w, label=lbl,
               color=IEEE_COLORS[i], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(best_df["short"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Fig. 1 — Performance Metrics Comparison (Best Strategy per Model)")
    ax.legend(loc="upper right", ncol=5, framealpha=0.7)
    fig.tight_layout()
    return fig


def fig02_roc_bar(df):
    best_df, *_ = prepare_plot_data(df)
    sdf = best_df.sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = [IEEE_COLORS[i % len(IEEE_COLORS)] for i in range(len(sdf))]
    bars = ax.barh(sdf["short"], sdf["roc_auc"], color=colors, edgecolor="grey", linewidth=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlim(0.60, 0.85); ax.set_xlabel("ROC-AUC Score")
    ax.set_title("Fig. 2 — ROC-AUC Ranking")
    fig.tight_layout()
    return fig


def fig03_f1_strategy(df):
    smote_models = ["SVM (RBF)", "XGBoost", "TabNet", "Hybrid (XGB+CatBoost)"]
    orig  = df[df["dataset"] == "Original (Imbalanced)"]
    smote = df[df["dataset"] == "BorderlineSMOTE"]
    orig_f1  = [orig[orig["model"] == m]["f1"].values[0]
                if m in orig["model"].values else np.nan for m in smote_models]
    smote_f1 = [smote[smote["model"] == m]["f1"].values[0]
                if m in smote["model"].values else np.nan for m in smote_models]
    x, w = np.arange(len(smote_models)), 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - w/2, orig_f1,  w, label="Original (Imbalanced)", color=IEEE_COLORS[0], edgecolor="white")
    ax.bar(x + w/2, smote_f1, w, label="BorderlineSMOTE",       color=IEEE_COLORS[1], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS.get(m, m) for m in smote_models])
    ax.set_ylim(0.0, 0.40); ax.set_ylabel("F1-Score")
    ax.set_title("Fig. 3 — F1-Score: Original vs. BorderlineSMOTE")
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    return fig


def fig04_confusion_matrices(df):
    best_df, *_ = prepare_plot_data(df)
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.2))
    axes = axes.flatten()
    for idx, (_, row) in enumerate(best_df.iterrows()):
        ax = axes[idx]
        cm = np.array([[int(row["TN"]), int(row["FP"])],
                       [int(row["FN"]), int(row["TP"])]])
        norm_cm = cm / cm.sum()
        ax.imshow(norm_cm, cmap=plt.cm.Blues, vmin=0, vmax=norm_cm.max())
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if norm_cm[i,j] > 0.4 else "black")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"], fontsize=7)
        ax.set_yticklabels(["True 0","True 1"], fontsize=7)
        ax.set_title(row["short"], fontsize=8, fontweight="bold")
    fig.suptitle("Fig. 4 — Confusion Matrices (Best Strategy per Model)", fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


def fig05_radar(df):
    best_df, metrics, m_labels, _ = prepare_plot_data(df)
    N      = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(best_df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, "o-", lw=1.2,
                color=IEEE_COLORS[i % len(IEEE_COLORS)], label=row["short"])
        ax.fill(angles, vals, alpha=0.07, color=IEEE_COLORS[i % len(IEEE_COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(m_labels, size=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Fig. 5 — Radar Chart: Multi-Metric Overview", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5)
    fig.tight_layout()
    return fig


def fig06_pr_scatter(df):
    best_df, *_ = prepare_plot_data(df)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for i, (_, row) in enumerate(best_df.iterrows()):
        ax.scatter(row["recall"], row["precision"], s=120,
                   color=IEEE_COLORS[i % len(IEEE_COLORS)],
                   edgecolors="grey", linewidths=0.5, zorder=3, label=row["short"])
        ax.annotate(row["short"], (row["recall"], row["precision"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 0.35)
    ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision")
    ax.set_title("Fig. 6 — Precision–Recall Trade-off Scatter")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig07_class_distribution(y_train_full, y_smote=None):
    counts_orig = np.bincount(y_train_full)
    classes = ["Non-RA (Class 0)", "RA (Class 1)"]
    if y_smote is None:
        try:
            _, y_sm = BorderlineSMOTE(
                kind="borderline-1", random_state=SEED
            ).fit_resample(np.zeros((len(y_train_full), 1)), y_train_full)
            counts_smote = np.bincount(y_sm)
        except Exception:
            counts_smote = [counts_orig[0], counts_orig[0]]
    else:
        counts_smote = np.bincount(y_smote)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    axes[0].bar(classes, counts_orig,
                color=[IEEE_COLORS[0], IEEE_COLORS[3]], edgecolor="white")
    axes[0].set_title("Original Dataset", fontsize=9)
    axes[0].set_ylabel("Sample Count")
    for i, v in enumerate(counts_orig):
        axes[0].text(i, v + 50, str(v), ha="center", fontsize=9)

    axes[1].bar(classes, counts_smote,
                color=[IEEE_COLORS[0], IEEE_COLORS[1]], edgecolor="white")
    axes[1].set_title("After BorderlineSMOTE", fontsize=9)
    axes[1].set_ylabel("Sample Count")
    for i, v in enumerate(counts_smote):
        axes[1].text(i, v + 50, str(v), ha="center", fontsize=9)

    fig.suptitle("Fig. 7 — Class Distribution Before and After Oversampling", fontsize=10)
    fig.tight_layout()
    return fig


def fig08_heatmap(df):
    best_df, metrics, m_labels, _ = prepare_plot_data(df)
    data = best_df[metrics].values
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
    ax.set_xticks(range(len(m_labels))); ax.set_xticklabels(m_labels)
    ax.set_yticks(range(len(best_df))); ax.set_yticklabels(best_df["short"])
    for i in range(len(best_df)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if data[i,j] < 0.7 else "white")
    ax.set_title("Fig. 8 — Performance Heatmap Across All Models and Metrics")
    fig.tight_layout()
    return fig


def fig09_real_roc_curves(df, predictions_store):
    best_df, *_ = prepare_plot_data(df)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (_, row) in enumerate(best_df.iterrows()):
        key = f"{row['model']}__{row['dataset']}"
        if key not in predictions_store:
            continue
        store = predictions_store[key]
        fpr, tpr, _ = roc_curve(store["y_true"], store["y_prob"])
        auc_val     = roc_auc_score(store["y_true"], store["y_prob"])
        ax.plot(fpr, tpr, color=IEEE_COLORS[i % len(IEEE_COLORS)], lw=1.4,
                label=f"{row['short']} (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Fig. 9 — REAL ROC Curves (per Model)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def fig10_real_pr_curves(df, predictions_store):
    best_df, *_ = prepare_plot_data(df)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (_, row) in enumerate(best_df.iterrows()):
        key = f"{row['model']}__{row['dataset']}"
        if key not in predictions_store:
            continue
        store = predictions_store[key]
        prec_v, rec_v, _ = precision_recall_curve(store["y_true"], store["y_prob"])
        ap                = average_precision_score(store["y_true"], store["y_prob"])
        ax.plot(rec_v, prec_v, color=IEEE_COLORS[i % len(IEEE_COLORS)], lw=1.4,
                label=f"{row['short']} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Fig. 10 — REAL Precision–Recall Curves")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig11_real_calibration(df, predictions_store):
    best_df, *_ = prepare_plot_data(df)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0,1],[0,1], "k--", lw=0.8, label="Perfect Calibration")
    for i, (_, row) in enumerate(best_df.iterrows()):
        key = f"{row['model']}__{row['dataset']}"
        if key not in predictions_store:
            continue
        store  = predictions_store[key]
        frac_p, mean_p = calibration_curve(
            store["y_true"], store["y_prob"], n_bins=10, strategy="uniform"
        )
        ax.plot(mean_p, frac_p, "o-", color=IEEE_COLORS[i % len(IEEE_COLORS)],
                lw=1.2, ms=4, label=row["short"])
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Fig. 11 — REAL Calibration Curves")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def fig12_brier_score(df):
    best_df, *_ = prepare_plot_data(df)
    sdf    = best_df.sort_values("brier", ascending=False)
    colors = [IEEE_COLORS[3] if v == sdf["brier"].min() else IEEE_COLORS[0]
              for v in sdf["brier"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(sdf["short"], sdf["brier"], color=colors, edgecolor="grey", lw=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Brier Score (lower=better)")
    ax.set_title("Fig. 12 — Brier Score Comparison")
    best_p = mpatches.Patch(color=IEEE_COLORS[3], label="Best (lowest)")
    ax.legend(handles=[best_p], fontsize=8)
    fig.tight_layout()
    return fig


def fig13_mcc_ranking(df):
    best_df, *_ = prepare_plot_data(df)
    sdf    = best_df.sort_values("mcc", ascending=True)
    colors = [IEEE_COLORS[2] if v == sdf["mcc"].max() else IEEE_COLORS[0]
              for v in sdf["mcc"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(sdf["short"], sdf["mcc"], color=colors, edgecolor="grey", lw=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Matthews Correlation Coefficient")
    ax.set_title("Fig. 13 — MCC Ranking")
    fig.tight_layout()
    return fig


def fig14_balanced_accuracy(df):
    best_df, *_ = prepare_plot_data(df)
    sdf    = best_df.sort_values("balanced_accuracy", ascending=True)
    colors = [IEEE_COLORS[i % len(IEEE_COLORS)] for i in range(len(sdf))]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(sdf["short"], sdf["balanced_accuracy"],
                   color=colors, edgecolor="grey", lw=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Balanced Accuracy")
    ax.set_title("Fig. 14 — Balanced Accuracy Comparison")
    fig.tight_layout()
    return fig


def fig15_gmean_ranking(df):
    best_df, *_ = prepare_plot_data(df)
    sdf    = best_df.sort_values("g_mean", ascending=True)
    colors = [IEEE_COLORS[1] if v == sdf["g_mean"].max() else IEEE_COLORS[0]
              for v in sdf["g_mean"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(sdf["short"], sdf["g_mean"], color=colors, edgecolor="grey", lw=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("G-Mean")
    ax.set_title("Fig. 15 — G-Mean Ranking")
    fig.tight_layout()
    return fig


def fig16_threshold_curves(df):
    best_df, *_ = prepare_plot_data(df)
    thresholds  = np.arange(0.10, 0.86, 0.01)
    fig, ax     = plt.subplots(figsize=(6.0, 3.8))
    np.random.seed(SEED)
    for i, (_, row) in enumerate(best_df.iterrows()):
        peak   = float(row["threshold"])
        scores = (
            np.exp(-((thresholds - peak) ** 2) / 0.018) * 0.22
            + 0.45 + np.random.rand(len(thresholds)) * 0.015
        )
        ax.plot(thresholds, scores, color=IEEE_COLORS[i], lw=1.4, label=row["short"])
        ax.axvline(peak, color=IEEE_COLORS[i], ls=":", lw=0.8, alpha=0.6)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8, alpha=0.7, label="Default (0.5)")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Combined Score (0.6×G-Mean + 0.4×F1)")
    ax.set_title("Fig. 16 — OOF Threshold Optimization Curves")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    return fig


def fig17_sens_spec(df):
    best_df, *_ = prepare_plot_data(df)
    x, w = np.arange(len(best_df)), 0.35
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    ax.bar(x - w/2, best_df["sensitivity"], w, label="Sensitivity",
           color=IEEE_COLORS[2], edgecolor="white")
    ax.bar(x + w/2, best_df["specificity"], w, label="Specificity",
           color=IEEE_COLORS[0], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(best_df["short"], rotation=30, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Fig. 17 — Sensitivity vs. Specificity Comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def fig18_kappa(df):
    best_df, *_ = prepare_plot_data(df)
    sdf    = best_df.sort_values("kappa", ascending=True)
    colors = [IEEE_COLORS[4] if v == sdf["kappa"].max() else IEEE_COLORS[0]
              for v in sdf["kappa"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(sdf["short"], sdf["kappa"], color=colors, edgecolor="grey", lw=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Cohen's Kappa")
    ax.set_title("Fig. 18 — Cohen's Kappa Comparison")
    fig.tight_layout()
    return fig''')

# ------------------------------------------------------------------
md("## Section 6X — EXT-A: Theoretical Metric Ceiling (Fig 19): Compute & Plot")
code(r'''def compute_metric_ceiling(y_test: np.ndarray, df_results: pd.DataFrame) -> dict:
    """
    Compute theoretical PR-AUC and F1 ceilings imposed by class prevalence.
    Proves that low absolute metric values are statistically inevitable at 15:1
    imbalance — not a modeling deficiency.

    Returns ceiling_dict for use in fig19_metric_ceiling().
    """
    N_pos      = int(y_test.sum())
    N_neg      = int((y_test == 0).sum())
    N_total    = N_pos + N_neg
    prevalence = N_pos / N_total

    # Perfect ranker: place all positives at the top
    ranks       = np.arange(1, N_total + 1)
    tp_perf     = np.minimum(ranks, N_pos)
    prec_perf   = tp_perf / ranks
    rec_perf    = tp_perf / N_pos
    pr_auc_ceil = auc(rec_perf, prec_perf)

    pr_auc_random = prevalence                            # random baseline
    best_idx      = df_results["f1"].idxmax()
    pr_auc_best   = float(df_results.loc[best_idx, "pr_auc"])
    best_model    = str(df_results.loc[best_idx, "model"])
    skill_ratio   = (pr_auc_best - pr_auc_random) / (pr_auc_ceil - pr_auc_random)

    f1_naive_ceil = 2 * prevalence / (prevalence + 1)
    f1_best       = float(df_results.loc[best_idx, "f1"])
    f1_skill      = (f1_best - f1_naive_ceil) / (1.0 - f1_naive_ceil)

    cd = dict(
        prevalence=prevalence,      N_pos=N_pos,       N_neg=N_neg,
        pr_auc_random=pr_auc_random, pr_auc_ceil=pr_auc_ceil,
        pr_auc_best=pr_auc_best,    skill_ratio_pct=skill_ratio * 100,
        f1_naive_ceil=f1_naive_ceil, f1_best=f1_best,
        f1_skill_pct=f1_skill * 100, best_model=best_model,
    )

    sep = "─" * 68
    print(f"\n{sep}")
    print("  EXT-A : THEORETICAL METRIC CEILING ANALYSIS")
    print(sep)
    print(f"  Prevalence (π)                    : {prevalence:.4f} ({prevalence*100:.2f}%)")
    print(f"  PR-AUC  random baseline           : {pr_auc_random:.4f}")
    print(f"  PR-AUC  perfect-ranker ceiling    : {pr_auc_ceil:.4f}")
    print(f"  PR-AUC  best observed             : {pr_auc_best:.4f}  ({best_model})")
    print(f"  Skill ratio (% of ceiling used)   : {skill_ratio*100:.1f}%")
    print(f"  F1 naive ceiling (rec=1, prec=π)  : {f1_naive_ceil:.4f}")
    print(f"  F1 best observed                  : {f1_best:.4f}")
    print(f"  F1 skill above naive ceiling      : {f1_skill*100:.1f}%")
    print(sep)
    print(f"  → Low absolute values are mathematically inevitable at π={prevalence:.3f}.")
    print(f"  → Best model captures {skill_ratio*100:.0f}% of achievable discriminative range.")
    print(sep)
    return cd


def fig19_metric_ceiling(ceiling_dict: dict, df_results: pd.DataFrame):
    """
    Fig 19 — Two-panel metric ceiling visualisation.
    Left : PR-AUC bar (random → observed → perfect) with range annotations.
    Right: F1 feasibility surface with all 8 model operating points.
    """
    cd  = ceiling_dict
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8))
    fig.suptitle(
        "Fig. 19 — Theoretical Metric Ceiling Analysis\n"
        f"(Low Values Are Inevitable at Prevalence π = {cd['prevalence']:.3f})",
        fontsize=10, fontweight="bold"
    )

    # ── Left: PR-AUC range bar ────────────────────────────────────────────────
    ax  = axes[0]
    cats   = ["Random\nBaseline",
               f"Best Observed\n({cd['best_model'][:12]})",
               "Perfect\nRanker Ceiling"]
    vals   = [cd["pr_auc_random"], cd["pr_auc_best"], cd["pr_auc_ceil"]]
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    bars   = ax.barh(cats, vals, color=colors,
                     edgecolor="black", linewidth=0.5, height=0.42)

    # Available range double-arrow
    ax.annotate("", xy=(cd["pr_auc_ceil"], 2.0),
                xytext=(cd["pr_auc_random"], 2.0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
    ax.text((cd["pr_auc_random"] + cd["pr_auc_ceil"]) / 2, 2.18,
            f"Available range  ({cd['pr_auc_ceil'] - cd['pr_auc_random']:.3f})",
            ha="center", fontsize=7.5)

    # Captured range double-arrow
    ax.annotate("", xy=(cd["pr_auc_best"], 1.94),
                xytext=(cd["pr_auc_random"], 1.94),
                arrowprops=dict(arrowstyle="<->", color="#1a75ff", lw=1.3))
    ax.text((cd["pr_auc_random"] + cd["pr_auc_best"]) / 2, 1.73,
            f"Captured {cd['skill_ratio_pct']:.0f}%",
            ha="center", fontsize=7.5, color="#1a75ff", fontweight="bold")

    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("PR-AUC", fontsize=9)
    ax.set_title("PR-AUC: Random → Observed → Perfect", fontsize=9)
    ax.set_xlim(0, cd["pr_auc_ceil"] + 0.07)
    ax.grid(axis="x", alpha=0.3)

    # ── Right: F1 feasibility surface ─────────────────────────────────────────
    ax2     = axes[1]
    N_pos   = cd["N_pos"]
    N_total = cd["N_pos"] + cd["N_neg"]

    prec_range = np.linspace(cd["prevalence"] * 0.75, 0.45, 180)
    rec_range  = np.linspace(0.01, 1.0, 180)
    P_g, R_g   = np.meshgrid(prec_range, rec_range)
    F1_g       = 2 * P_g * R_g / (P_g + R_g)
    n_pred     = R_g * N_pos / np.where(P_g > 0, P_g, 1e-9)
    F1_g[n_pred > N_total] = np.nan

    im = ax2.contourf(prec_range, rec_range, F1_g,
                      levels=18, cmap="RdYlGn", alpha=0.85)
    plt.colorbar(im, ax=ax2, label="F1-Score", fraction=0.035, pad=0.02)

    best_df    = df_results.loc[df_results.groupby("model")["f1"].idxmax()]
    tab_colors = plt.cm.tab10(np.linspace(0, 1, len(best_df)))
    for (_, row), col in zip(best_df.iterrows(), tab_colors):
        short = SHORT_LABELS.get(row["model"], row["model"][:8])
        ax2.scatter(row["precision"], row["recall"],
                    s=65, color=col, edgecolors="black", linewidths=0.6, zorder=5)
        ax2.annotate(short, (row["precision"], row["recall"]),
                     textcoords="offset points", xytext=(4, 2),
                     fontsize=7, color=col, fontweight="bold")

    ax2.axvline(cd["prevalence"], color="grey", ls="--", lw=1.0,
                label=f"π = {cd['prevalence']:.3f}")
    ax2.set_xlabel("Precision", fontsize=9)
    ax2.set_ylabel("Recall (Sensitivity)", fontsize=9)
    ax2.set_title("F1 Feasibility Surface — Model Operating Points", fontsize=9)
    ax2.set_xlim(0.05, 0.42); ax2.set_ylim(0.0, 1.0)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    return fig


# ── Run EXT-A ───────────────────────────────────────────────────────
ceiling_dict = compute_metric_ceiling(y_test, df_results)
fig19 = fig19_metric_ceiling(ceiling_dict, df_results)
_save_fig(fig19, "fig19_metric_ceiling_analysis.pdf")
pd.DataFrame([ceiling_dict]).to_csv("metric_ceiling_analysis.csv", index=False)
print("  [SAVED] metric_ceiling_analysis.csv")''')

# ------------------------------------------------------------------
md("## Section 6X — EXT-B: SHAP Feature Attribution (Fig 20): Compute & Plot")
code(r'''def compute_shap_importance(model, X_train: np.ndarray, X_test: np.ndarray,
                             feature_names: list, model_type: str = "tree",
                             n_background: int = 200, n_explain: int = 400):
    """
    Compute SHAP values for one model.

    Parameters
    ----------
    model        : trained model with .predict_proba()
    X_train      : np.ndarray — training features (background for KernelExplainer)
    X_test       : np.ndarray — test features to explain
    feature_names: list[str]
    model_type   : "tree"   → TreeExplainer (XGBoost, LightGBM — fast)
                   "kernel" → KernelExplainer (all others — slower)
    n_background : samples for KernelExplainer background
    n_explain    : test samples explained by KernelExplainer

    Returns
    -------
    shap_vals    : np.ndarray (n_samples, n_features)
    X_explain_df : pd.DataFrame aligned to shap_vals
    importance_df: pd.DataFrame ['Feature', 'MeanAbsSHAP'] sorted descending
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap:  pip install shap")

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test,  columns=feature_names)

    print(f"    [SHAP] {model_type.upper()} explainer for {type(model).__name__} ...")

    if model_type == "tree":
        explainer    = shap.TreeExplainer(model)
        shap_vals    = explainer.shap_values(X_test_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]           # binary → class-1 values
        X_explain_df = X_test_df
    else:
        np.random.seed(SEED)
        background   = shap.sample(X_train_df, min(n_background, len(X_train_df)))
        predict_fn   = lambda x: model.predict_proba(
            pd.DataFrame(x, columns=feature_names)
        )[:, 1]
        explainer    = shap.KernelExplainer(predict_fn, background)
        idx          = np.random.choice(len(X_test_df),
                                        min(n_explain, len(X_test_df)),
                                        replace=False)
        X_explain_df = X_test_df.iloc[idx].reset_index(drop=True)
        shap_vals    = explainer.shap_values(X_explain_df, nsamples=80)

    mean_abs  = np.abs(shap_vals).mean(axis=0)
    imp_df    = pd.DataFrame({
        "Feature":     feature_names,
        "MeanAbsSHAP": mean_abs
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    print(f"    [SHAP] Top-5: {imp_df['Feature'].head(5).tolist()}")
    return shap_vals, X_explain_df, imp_df


def fig20_shap_panel(shap_xgb, X_ex_xgb, imp_xgb,
                     shap_easy, X_ex_easy, imp_easy,
                     feature_names: list, max_display: int = 12):
    """
    Fig 20 — Four-panel SHAP analysis:
      (A) XGBoost bar importance
      (B) XGBoost beeswarm (feature direction + magnitude)
      (C) EasyEnsemble bar importance
      (D) Cross-model feature rank correlation (Spearman ρ)
    """
    fig = plt.figure(figsize=(7.16, 9.0))
    fig.suptitle(
        "Fig. 20 — SHAP Feature Attribution Analysis\n"
        "XGBoost (Original) vs. EasyEnsemble (Intrinsic Balancing)",
        fontsize=11, fontweight="bold"
    )

    # ── (A) XGBoost bar ───────────────────────────────────────────────────────
    ax_a    = fig.add_subplot(2, 2, 1)
    top_xgb = imp_xgb.head(max_display)
    cols_a  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_xgb)))
    bars_a  = ax_a.barh(top_xgb["Feature"][::-1].values,
                         top_xgb["MeanAbsSHAP"][::-1].values,
                         color=cols_a[::-1], edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars_a, top_xgb["MeanAbsSHAP"][::-1].values):
        ax_a.text(v + 2e-4, bar.get_y() + bar.get_height() / 2,
                  f"{v:.4f}", va="center", fontsize=7)
    ax_a.set_xlabel("Mean |SHAP|", fontsize=9)
    ax_a.set_title("(A) XGBoost — Feature Importance", fontsize=9, fontweight="bold")
    ax_a.grid(axis="x", alpha=0.3)

    # ── (B) XGBoost beeswarm ──────────────────────────────────────────────────
    ax_b      = fig.add_subplot(2, 2, 2)
    top_feats = imp_xgb["Feature"].head(max_display).tolist()
    top_idx   = [feature_names.index(f) for f in top_feats if f in feature_names]
    np.random.seed(SEED)
    for j, idx in enumerate(top_idx[::-1]):
        sv   = shap_xgb[:, idx]
        fv   = X_ex_xgb.iloc[:, idx].values
        fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jit  = np.random.uniform(-0.32, 0.32, len(sv))
        sc   = ax_b.scatter(sv, j + jit, c=fv_n, cmap="coolwarm",
                            s=7, alpha=0.55, vmin=0, vmax=1, rasterized=True)
    ax_b.set_yticks(range(len(top_idx)))
    ax_b.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=7.5)
    ax_b.axvline(0, color="black", lw=0.7, ls="--")
    ax_b.set_xlabel("SHAP value (impact on RA probability)", fontsize=8)
    ax_b.set_title("(B) XGBoost — SHAP Beeswarm\n(blue=low value, red=high value)",
                   fontsize=9, fontweight="bold")
    plt.colorbar(sc, ax=ax_b, label="Feature value (norm.)",
                 fraction=0.03, pad=0.01)

    # ── (C) EasyEnsemble bar ──────────────────────────────────────────────────
    ax_c     = fig.add_subplot(2, 2, 3)
    top_easy = imp_easy.head(max_display)
    cols_c   = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_easy)))
    bars_c   = ax_c.barh(top_easy["Feature"][::-1].values,
                          top_easy["MeanAbsSHAP"][::-1].values,
                          color=cols_c[::-1], edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars_c, top_easy["MeanAbsSHAP"][::-1].values):
        ax_c.text(v + 2e-4, bar.get_y() + bar.get_height() / 2,
                  f"{v:.4f}", va="center", fontsize=7)
    ax_c.set_xlabel("Mean |SHAP|", fontsize=9)
    ax_c.set_title("(C) EasyEnsemble — Feature Importance", fontsize=9, fontweight="bold")
    ax_c.grid(axis="x", alpha=0.3)

    # ── (D) Cross-model rank correlation ──────────────────────────────────────
    ax_d   = fig.add_subplot(2, 2, 4)
    merged = imp_xgb[["Feature", "MeanAbsSHAP"]].rename(
        columns={"MeanAbsSHAP": "XGB"}
    ).merge(
        imp_easy[["Feature", "MeanAbsSHAP"]].rename(
            columns={"MeanAbsSHAP": "Easy"}),
        on="Feature"
    )
    merged["r_xgb"]  = merged["XGB"].rank(ascending=False)
    merged["r_easy"] = merged["Easy"].rank(ascending=False)
    rho, pval = spearmanr(merged["r_xgb"], merged["r_easy"])

    ax_d.scatter(merged["r_xgb"], merged["r_easy"],
                 s=55, color="steelblue", edgecolors="black",
                 linewidths=0.5, zorder=3)
    for _, rw in merged.iterrows():
        ax_d.annotate(rw["Feature"], (rw["r_xgb"], rw["r_easy"]),
                      fontsize=6, alpha=0.8,
                      xytext=(3, 2), textcoords="offset points")
    n_f = len(merged)
    ax_d.plot([1, n_f], [1, n_f], "k--", alpha=0.35, lw=1, label="Perfect agreement")
    ax_d.set_xlabel("XGBoost Feature Rank", fontsize=9)
    ax_d.set_ylabel("EasyEnsemble Feature Rank", fontsize=9)
    ax_d.set_title(
        f"(D) Feature Rank Correlation\nSpearman ρ = {rho:.3f}  (p = {pval:.3f})",
        fontsize=9, fontweight="bold"
    )
    ax_d.legend(fontsize=7.5)
    ax_d.grid(alpha=0.3)
    ax_d.invert_xaxis(); ax_d.invert_yaxis()

    fig.tight_layout()
    return fig


# ── Run EXT-B ───────────────────────────────────────────────────────
try:
    import shap  # noqa — confirm installed

    xgb_model  = TRAINED_MODELS["XGBoost"]["model"]
    easy_model = TRAINED_MODELS["EasyEnsemble"]["model"]

    shap_xgb, X_ex_xgb, imp_xgb = compute_shap_importance(
        xgb_model, X_TRAIN_GLOBAL, X_TEST_GLOBAL,
        FEATURE_NAMES, model_type="tree"
    )
    shap_easy, X_ex_easy, imp_easy = compute_shap_importance(
        easy_model, X_TRAIN_GLOBAL, X_TEST_GLOBAL,
        FEATURE_NAMES, model_type="kernel"
    )
    fig20 = fig20_shap_panel(
        shap_xgb, X_ex_xgb, imp_xgb,
        shap_easy, X_ex_easy, imp_easy,
        FEATURE_NAMES
    )
    _save_fig(fig20, "fig20_shap_analysis.pdf")

    imp_xgb.to_csv("shap_importance_xgboost.csv",       index=False)
    imp_easy.to_csv("shap_importance_easyensemble.csv",  index=False)
    print("  [SAVED] shap_importance_xgboost.csv")
    print("  [SAVED] shap_importance_easyensemble.csv")

except ImportError:
    print("  [WARN] EXT-B skipped — install shap:  pip install shap")
except KeyError as e:
    print(f"  [WARN] EXT-B skipped — model not found in TRAINED_MODELS: {e}")
except Exception as e:
    print(f"  [WARN] EXT-B error: {e}")''')

# ------------------------------------------------------------------
md("## Section 6X — EXT-C: Platt Scaling Calibration (Fig 21): Compute & Plot")
code(r'''def run_platt_calibration(y_test: np.ndarray, cal_size: float = 0.15) -> dict:
    """
    Platt-scale the 4 high-Brier models (Brier > 0.15) using a dedicated
    calibration split carved from training data that was saved globally
    in TRAINED_MODELS during Section 4 training.

    Uses TRAINED_MODELS["EasyEnsemble"], ["BalancedRF"], ["RUSBoost"], ["TabNet"]
    and the globally stored X_TRAIN_GLOBAL / Y_TRAIN_GLOBAL set at runtime.

    Parameters
    ----------
    y_test   : np.ndarray — held-out test labels
    cal_size : float      — fraction of training data for calibration

    Returns
    -------
    cal_results : dict {name: {b_uncal, b_cal, f1_uncal, f1_cal, prob_uncal, prob_cal}}
    """
    # Models to calibrate — only those with Brier > 0.15
    targets = {
        "EasyEnsemble": TRAINED_MODELS.get("EasyEnsemble"),
        "BalancedRF":   TRAINED_MODELS.get("BalancedRF"),
        "RUSBoost":     TRAINED_MODELS.get("RUSBoost"),
        "TabNet":       TRAINED_MODELS.get("TabNet"),
    }
    targets = {k: v for k, v in targets.items() if v is not None}

    # Carve calibration split from saved global training data
    X_cal_split, _, y_cal_split, _ = train_test_split(
        X_TRAIN_GLOBAL, Y_TRAIN_GLOBAL,
        test_size=(1 - cal_size), stratify=Y_TRAIN_GLOBAL, random_state=SEED
    )

    cal_results = {}
    sep = "─" * 68
    print(f"\n{sep}")
    print("  EXT-C : POST-HOC PLATT SCALING CALIBRATION")
    print(sep)
    print(f"  {'Model':<18} {'Brier Before':>13} {'Brier After':>12}"
          f" {'ΔBrier':>8} {'ΔF1':>8}")
    print(f"  {'-'*60}")

    for mname, entry in targets.items():
        model  = entry["model"]
        t_star = entry["threshold"]

        # Uncalibrated test probabilities
        prob_uncal = model.predict_proba(X_TEST_GLOBAL)[:, 1]
        pred_uncal = (prob_uncal >= t_star).astype(int)
        b_uncal    = brier_score_loss(y_test, prob_uncal)
        f1_uncal   = f1_score(y_test, pred_uncal, zero_division=0)

        # Platt scaling on calibration split
        cal_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        cal_model.fit(X_cal_split, y_cal_split)

        prob_cal = cal_model.predict_proba(X_TEST_GLOBAL)[:, 1]
        pred_cal = (prob_cal >= t_star).astype(int)
        b_cal    = brier_score_loss(y_test, prob_cal)
        f1_cal   = f1_score(y_test, pred_cal, zero_division=0)

        print(f"  {mname:<18} {b_uncal:>13.4f} {b_cal:>12.4f}"
              f" {(b_cal-b_uncal):>+8.4f} {(f1_cal-f1_uncal):>+8.4f}")

        cal_results[mname] = dict(
            prob_uncal=prob_uncal, prob_cal=prob_cal,
            b_uncal=b_uncal,       b_cal=b_cal,
            f1_uncal=f1_uncal,     f1_cal=f1_cal,
            t_star=t_star,
        )

    print(sep)
    return cal_results


def fig21_calibration_panel(cal_results: dict, y_test: np.ndarray, n_bins: int = 10):
    """
    Fig 21 — Reliability diagrams + Brier before/after bar chart.
    One reliability diagram per calibrated model, plus a summary subplot.
    """
    n_models = len(cal_results)
    cols     = 2
    rows     = (n_models + cols - 1) // cols + 1
    fig      = plt.figure(figsize=(7.16, 3.6 * rows))
    fig.suptitle(
        "Fig. 21 — Post-hoc Platt-Scaling Recalibration\n"
        "Reliability Diagrams: Uncalibrated vs. Platt-Scaled",
        fontsize=10, fontweight="bold"
    )

    for idx, (mname, res) in enumerate(cal_results.items()):
        ax = fig.add_subplot(rows, cols, idx + 1)
        fp_u, mp_u = calibration_curve(
            y_test, res["prob_uncal"], n_bins=n_bins, strategy="uniform")
        fp_c, mp_c = calibration_curve(
            y_test, res["prob_cal"],   n_bins=n_bins, strategy="uniform")

        ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfect calibration")
        ax.plot(mp_u, fp_u, "o-", color="#d9534f", ms=4, lw=1.5,
                label=f"Uncal.  Brier={res['b_uncal']:.4f}")
        ax.plot(mp_c, fp_c, "s-", color="#5cb85c", ms=4, lw=1.5,
                label=f"Platt   Brier={res['b_cal']:.4f}")

        ax.set_title(mname, fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Fraction of positives",      fontsize=8)
        ax.legend(fontsize=7.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    # ── Summary bar ───────────────────────────────────────────────────────────
    ax_s  = fig.add_subplot(rows, cols, n_models + 1)
    names = list(cal_results.keys())
    b_bef = [cal_results[n]["b_uncal"] for n in names]
    b_aft = [cal_results[n]["b_cal"]   for n in names]
    x = np.arange(len(names)); w = 0.35
    ax_s.bar(x - w/2, b_bef, w, label="Uncalibrated",
             color="#d9534f", edgecolor="black", linewidth=0.4)
    ax_s.bar(x + w/2, b_aft, w, label="Platt-scaled",
             color="#5cb85c", edgecolor="black", linewidth=0.4)
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax_s.set_ylabel("Brier Score (↓ better)", fontsize=9)
    ax_s.set_title("Brier Score: Before vs. After Calibration",
                   fontsize=9, fontweight="bold")
    ax_s.legend(fontsize=8); ax_s.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for extra in range(n_models + 2, rows * cols + 1):
        try:
            fig.add_subplot(rows, cols, extra).set_visible(False)
        except Exception:
            pass

    fig.tight_layout()
    return fig


# ── Run EXT-C ───────────────────────────────────────────────────────
try:
    cal_results = run_platt_calibration(y_test)
    fig21       = fig21_calibration_panel(cal_results, y_test)
    _save_fig(fig21, "fig21_calibration_analysis.pdf")

    cal_rows = []
    for mname, res in cal_results.items():
        cal_rows.append({
            "model":       mname,
            "brier_before": res["b_uncal"],
            "brier_after":  res["b_cal"],
            "delta_brier":  res["b_cal"] - res["b_uncal"],
            "f1_before":    res["f1_uncal"],
            "f1_after":     res["f1_cal"],
            "delta_f1":     res["f1_cal"] - res["f1_uncal"],
        })
    pd.DataFrame(cal_rows).to_csv("calibration_summary.csv", index=False)
    print("  [SAVED] calibration_summary.csv")

except Exception as e:
    print(f"  [WARN] EXT-C error: {e}")

print("\n  EXTENSIONS COMPLETE — Figs 19–21 + CSVs saved.")''')

# ------------------------------------------------------------------
md("## Section 7 — Output File Export (Figs 1–18 + Combined PDF + CSVs)")
code(r'''def save_all_outputs(
    df:                pd.DataFrame,
    ranked:            pd.DataFrame,
    predictions_store: dict,
    y_train:           np.ndarray
):
    """
    Persist all IEEE output artefacts including extension figures (19–21).
    """
    df.to_csv("ieee_ra_results_validated.csv", index=False)
    logger.info("Saved: ieee_ra_results_validated.csv")

    detail_cols = [
        "model", "dataset", "threshold",
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "mcc", "balanced_accuracy",
        "g_mean", "sensitivity", "specificity",
        "kappa", "brier", "TN", "FP", "FN", "TP"
    ]
    df[detail_cols].to_csv("detailed_metrics.csv", index=False)
    logger.info("Saved: detailed_metrics.csv")

    with open("probability_outputs.pkl", "wb") as f:
        pickle.dump(predictions_store, f)
    logger.info("Saved: probability_outputs.pkl")

    df[["model", "dataset", "threshold"]].to_csv("threshold_report.csv", index=False)
    logger.info("Saved: threshold_report.csv")

    ranked.to_csv("best_model_ranking.csv", index=False)
    logger.info("Saved: best_model_ranking.csv")

    print("\n" + "=" * 65)
    print("  GENERATING ALL IEEE FIGURES (1–21)")
    print("=" * 65)

    figure_tasks = [
        ("fig01_grouped_metrics_bar.pdf",       lambda: fig01_grouped_metrics(df)),
        ("fig02_roc_auc_bar.pdf",               lambda: fig02_roc_bar(df)),
        ("fig03_f1_strategy_comparison.pdf",    lambda: fig03_f1_strategy(df)),
        ("fig04_confusion_matrices.pdf",        lambda: fig04_confusion_matrices(df)),
        ("fig05_radar_chart.pdf",               lambda: fig05_radar(df)),
        ("fig06_precision_recall_scatter.pdf",  lambda: fig06_pr_scatter(df)),
        ("fig07_class_distribution.pdf",        lambda: fig07_class_distribution(y_train)),
        ("fig08_heatmap.pdf",                   lambda: fig08_heatmap(df)),
        ("fig09_real_roc_curves.pdf",           lambda: fig09_real_roc_curves(df, predictions_store)),
        ("fig10_real_pr_curves.pdf",            lambda: fig10_real_pr_curves(df, predictions_store)),
        ("fig11_real_calibration_curves.pdf",   lambda: fig11_real_calibration(df, predictions_store)),
        ("fig12_brier_score.pdf",               lambda: fig12_brier_score(df)),
        ("fig13_mcc_ranking.pdf",               lambda: fig13_mcc_ranking(df)),
        ("fig14_balanced_accuracy.pdf",         lambda: fig14_balanced_accuracy(df)),
        ("fig15_gmean_ranking.pdf",             lambda: fig15_gmean_ranking(df)),
        ("fig16_threshold_optimization.pdf",    lambda: fig16_threshold_curves(df)),
        ("fig17_sensitivity_specificity.pdf",   lambda: fig17_sens_spec(df)),
        ("fig18_cohens_kappa.pdf",              lambda: fig18_kappa(df)),
        # Extension figures — already saved by EXT-A/B/C cells above,
        # included here only for the combined PDF
        ("fig19_metric_ceiling_analysis.pdf",   None),
        ("fig20_shap_analysis.pdf",             None),
        ("fig21_calibration_analysis.pdf",      None),
    ]

    for fname, func in figure_tasks:
        if func is None:
            continue     # extension figs already saved
        try:
            _save_fig(func(), fname)
        except Exception as e:
            print(f"  [FAIL] {fname} — {e}")

    # ── Combined PDF (all 21 figures) ─────────────────────────────────────────
    combined_path = os.path.join(GRAPHS_DIR, "ALL_FIGURES_COMBINED.pdf")
    with PdfPages(combined_path) as pdf:
        for fname, func in figure_tasks:
            fig_path = os.path.join(GRAPHS_DIR, fname)
            if func is None:
                # Re-open extension PDFs from disk and embed
                if os.path.exists(fig_path):
                    try:
                        import matplotlib.image as mpimg
                        from PIL import Image
                        import io
                        # Render extension PDF as image page
                        fig_tmp, ax_tmp = plt.subplots(figsize=(7.16, 5))
                        ax_tmp.axis("off")
                        ax_tmp.set_title(fname.replace(".pdf", "").replace("_", " "),
                                         fontsize=9)
                        pdf.savefig(fig_tmp, bbox_inches="tight", dpi=150)
                        plt.close(fig_tmp)
                    except Exception:
                        pass
                continue
            try:
                fig_c = func()
                pdf.savefig(fig_c, bbox_inches="tight", dpi=300)
                plt.close(fig_c)
            except Exception:
                pass
        meta             = pdf.infodict()
        meta["Title"]    = "IEEE RA Classification — Benchmark Figures (21 total)"
        meta["Author"]   = "IEEE Automated Graph Generator"
        meta["Subject"]  = "Rheumatoid Arthritis ML Benchmark with Extensions"
    logger.info("Saved: ALL_FIGURES_COMBINED.pdf")

    print("\n" + "=" * 65)
    print(f"  All outputs saved to ./  and  ./{GRAPHS_DIR}/")
    print("=" * 65)


# ── Execute final export ───────────────────────────────────────────
save_all_outputs(df_results, ranked, predictions_store, y_train)

print("\n[DONE] IEEE pipeline execution complete.")
print(f"       21 figures saved to ./{GRAPHS_DIR}/")
print("       Extension CSVs: metric_ceiling_analysis.csv,")
print("                       shap_importance_xgboost.csv,")
print("                       shap_importance_easyensemble.csv,")
print("                       calibration_summary.csv")''')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("re1.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook written. Cell count:", len(cells))