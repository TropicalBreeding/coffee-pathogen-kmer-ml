#!/usr/bin/env python3

"""
Two-Stage Pathogen Detection and Infection-Rate Prediction (Step 5)
===================================================================

Description
-----------
Implements a two-model framework for pathogen detection and quantification
from k-mer feature matrices:

  **Model A — Detector** (Logistic Regression, elastic-net penalty)
    Binary classifier that distinguishes healthy from infected samples.
    Optionally trained on a restricted set of low infection rates
    (``--detector-train-rates``) to sharpen the decision boundary for
    faint pathogen signals.  The detection threshold is selected on an
    inner validation split at a user-specified false-positive rate
    (``--target-fpr``, default 0.05).

  **Model B — Quantifier** (Ridge regression on log10 rate)
    Predicts the continuous infection rate for samples flagged as infected.
    Always trained on all available infection rates to capture the full
    dynamic range.  Predictions can be evaluated on infected samples only,
    healthy samples only, or both (``--rate-test-set``).

Training and evaluation use an outer stratified k-fold cross-validation
(default 4 folds) with an inner train/validation split for threshold
calibration.  The procedure is repeated across multiple simulation
replicates for statistical robustness.

Dependencies
------------
- Python >= 3.8
- NumPy, pandas, scikit-learn

Inputs
------
- ``metadata.csv`` with columns: ``sample_id``, ``label``, and an
  infection-rate column (``target_rate``, ``targetrate``, ``rate``, or
  ``target_rate_fraction``)
- ``sample_ids.txt`` — ordered list matching matrix row order
- ``matrix.npy`` — dense k-mer count matrix (samples x 4^k)

Input scale
-----------
``--input-scale counts``
    Matrix contains raw k-mer counts; transformed via ``--norm``.

``--input-scale log1pcpm``
    Matrix is already log1p(CPM)-transformed; used as-is.

Outputs
-------
- ``step5_*_detection_*.csv``        — per-fold detection metrics
  (accuracy, specificity, sensitivity, balanced accuracy, FPR)
- ``step5_*_rate_predictions_*.csv`` — per-sample rate predictions
- ``step5_*_rate_summary_*.csv``     — MAE summary (overall + per rate)

Examples
--------
1. Internal CV with detector trained on five lowest rates::

     python 5_two_stage_detection.py \\
       --mode internal \\
       --metadata metadata.csv \\
       --matrix results_kmer/matrix_k7.npy \\
       --ids results_kmer/sample_ids.txt \\
       --outdir results_step5 \\
       --detector-train-rates 0.00005,0.0001,0.0005,0.001,0.005

2. Internal CV using all rates for both models (baseline)::

     python 5_two_stage_detection.py \\
       --mode internal \\
       --metadata metadata.csv \\
       --matrix matrix_k7.npy \\
       --ids sample_ids.txt \\
       --outdir results_step5_allrates

3. External validation (train on one cultivar, test on another)::

     python 5_two_stage_detection.py \\
       --mode external \\
       --metadata metadata_train.csv \\
       --train-matrix matrix_train_k7.npy \\
       --train-ids train_ids.txt \\
       --test-metadata metadata_test.csv \\
       --test-matrix matrix_test_k7.npy \\
       --test-ids test_ids.txt \\
       --outdir results_step5_external \\
       --detector-train-rates 0.00005,0.0001,0.0005,0.001,0.005

4. Predict rates for healthy test samples (false-positive analysis)::

     python 5_two_stage_detection.py \\
       --mode internal \\
       --metadata metadata.csv \\
       --matrix matrix_k7.npy \\
       --ids sample_ids.txt \\
       --outdir results_step5_healthy \\
       --rate-test-set healthy_only
"""

import sys
import gc
import copy
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_curve


SEED0 = 42
N_SPLITS = 4
N_SIM_REPEATS = 10
BUFFER_SIZE = 50


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 5 v6: detection + rate prediction — two-model architecture"
    )
    p.add_argument("--outdir", required=True)
    p.add_argument("--mode", choices=["internal", "external"], default="internal")

    # internal mode inputs
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--matrix", type=str, default=None)
    p.add_argument("--ids", type=str, default=None)

    # external mode inputs
    p.add_argument("--test-metadata", type=str, default=None)
    p.add_argument("--train-matrix", type=str, default=None)
    p.add_argument("--train-ids", type=str, default=None)
    p.add_argument("--test-matrix", type=str, default=None)
    p.add_argument("--test-ids", type=str, default=None)

    p.add_argument("--fixed-n", type=int, default=None,
                   help="Optional cap for infected samples PER RATE in each sim_rep.")
    p.add_argument("--target-fpr", type=float, default=0.05,
                   help="Target FPR for threshold selection on inner validation data.")
    p.add_argument("--ridge-alpha", type=float, default=10.0)

    p.add_argument("--norm", choices=["log1p_cpm", "log1p_raw"], default="log1p_cpm",
                   help="Used only when --input-scale counts.")
    p.add_argument("--input-scale", choices=["counts", "log1pcpm"], default="counts",
                   help="Whether input matrix is raw counts or already log1pCPM-transformed.")

    p.add_argument("--logreg-c", type=float, default=0.1,
                   help="Inverse regularization strength for logistic regression.")
    p.add_argument("--logreg-l1-ratio", type=float, default=0.3,
                   help="Elastic-net mixing parameter for logistic regression.")

    p.add_argument("--inner-val-frac", type=float, default=0.2,
                   help="Fraction of outer-train used as inner validation for threshold selection.")

    p.add_argument("--weight-mode", choices=["none", "pow025", "sqrt"], default="pow025",
                   help="How strongly to emphasize lower infected rates in Model A.")
    p.add_argument("--max-infected-weight", type=float, default=3.0,
                   help="Maximum infected sample weight for Model A after rate-based weighting.")

    # --- rate filtering ---
    p.add_argument("--detector-train-rates", type=str, default=None,
                   help=(
                       "Comma-separated infected rates used to train Model A (detector) ONLY. "
                       "Example: '0.00005,0.0001,0.0005,0.001,0.005'. "
                       "Healthy samples are always kept. "
                       "If not set, falls back to --train-rates. "
                       "Model B (regressor) always uses ALL infected rates regardless of this flag."
                   ))
    p.add_argument("--train-rates", type=str, default=None,
                   help=(
                       "Fallback comma-separated infected rates for Model A when "
                       "--detector-train-rates is not set. "
                       "Kept for backward compatibility with v5. "
                       "Model B ignores this flag."
                   ))

    p.add_argument("--rate-tol", type=float, default=1e-12,
                   help="Tolerance used when matching rates to metadata rates.")
    p.add_argument("--rate-test-set", choices=["infected_only", "healthy_only", "all"],
                   default="infected_only",
                   help=(
                       "Which test samples should receive Model B rate predictions. "
                       "Training remains infected-only. "
                       "infected_only = infected test samples only; "
                       "healthy_only = healthy test samples only; "
                       "all = both infected and healthy test samples."
                   ))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers — rate list parsing
# ---------------------------------------------------------------------------

def parse_rate_list(s):
    if s is None or str(s).strip() == "":
        return None
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return sorted(vals) if vals else None


def effective_detector_rates(args):
    """
    Return the rate list that Model A (detector) should be restricted to.
    Priority: --detector-train-rates > --train-rates > None (all rates).
    """
    det = parse_rate_list(args.detector_train_rates)
    if det is not None:
        return det
    return parse_rate_list(args.train_rates)


def detector_rates_label(args):
    """Human-readable label for the detection CSV column."""
    det = parse_rate_list(args.detector_train_rates)
    if det is not None:
        return ",".join(str(r) for r in det)
    tr = parse_rate_list(args.train_rates)
    if tr is not None:
        return ",".join(str(r) for r in tr)
    return "all"


def training_suffix_from_rates(rate_list):
    """
    Convert an actual training-rate list to a filename suffix.

    None means all rates are used.
    A list of length 2 becomes training_2rates.
    Any other explicit list becomes training_<N>rates.
    """
    if rate_list is None:
        return "training_allRates"
    return f"training_{len(rate_list)}rates"


def detector_training_suffix(args):
    """Filename suffix for Model A outputs based on actual detector training rates."""
    return training_suffix_from_rates(effective_detector_rates(args))


def regressor_training_suffix():
    """
    Filename suffix for Model B outputs.

    The regressor always trains on all infected rates and is independent of
    the detector-rate restriction used by Model A, so label these outputs as
    shared rather than case-specific.
    """
    return "allRates"


def regressor_detector_label():
    """Label used in Model B outputs when detector rates are not applicable."""
    return "not_used_by_regressor"


def rate_test_set_label(args):
    return args.rate_test_set


def make_args_copy(args, detector_rates_str=None, train_rates_str=None):
    new_args = copy.deepcopy(args)
    new_args.detector_train_rates = detector_rates_str
    new_args.train_rates = train_rates_str
    return new_args


def detector_case_label(rate_list):
    if rate_list is None:
        return "allrates"
    return f"{len(rate_list)}.lowrates"


def build_detector_arg_sets(args):
    arg_sets = [(detector_case_label(None), make_args_copy(args))]

    low_rates = effective_detector_rates(args)
    if low_rates is not None:
        low_rate_text = ",".join(str(r) for r in low_rates)
        arg_sets.append((
            detector_case_label(low_rates),
            make_args_copy(args, detector_rates_str=low_rate_text, train_rates_str=None),
        ))

    return arg_sets
def select_rate_test_indices(y, mode):
    y = np.asarray(y)
    if mode == "infected_only":
        return np.where(y == 1)[0]
    if mode == "healthy_only":
        return np.where(y == 0)[0]
    if mode == "all":
        return np.arange(len(y))
    raise ValueError(f"Unsupported --rate-test-set: {mode}")


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def read_id_list(fp: Path):
    with open(fp, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_matrix(fp: Path):
    return np.load(fp)


def detect_rate_col(df):
    for c in ["target_rate", "targetrate", "rate", "target_rate_fraction"]:
        if c in df.columns:
            return c
    raise ValueError("Metadata missing infection-rate column.")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def transform_log1p_cpm(X, scale=1e6, eps=1e-12):
    X = X.astype(np.float32, copy=False)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps).astype(np.float32)
    X = X / row_sum
    X = X * np.float32(scale)
    return np.log1p(X).astype(np.float32, copy=False)


def transform_log1p_raw(X):
    return np.log1p(X.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def preprocess_matrix(X, norm_method, input_scale):
    X = X.astype(np.float32, copy=False)
    if input_scale == "log1pcpm":
        return X
    if norm_method == "log1p_cpm":
        return transform_log1p_cpm(X)
    if norm_method == "log1p_raw":
        return transform_log1p_raw(X)
    raise ValueError(f"Unknown norm method: {norm_method}")


def load_dataset(matrix_fp, ids_fp, meta_fp, norm_method, input_scale):
    X = load_matrix(Path(matrix_fp))
    ids = read_id_list(Path(ids_fp))
    if X.shape[0] != len(ids):
        raise RuntimeError("Matrix row count does not match ID count.")

    df = pd.read_csv(meta_fp, dtype=str)
    rate_col = detect_rate_col(df)
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)

    df = df.set_index("sample_id").reindex(ids).reset_index()
    if df["label"].isna().any():
        missing = df["sample_id"][df["label"].isna()].head(3).tolist()
        raise RuntimeError(f"Metadata missing labels after reindexing. Example IDs: {missing}")

    X = preprocess_matrix(X, norm_method, input_scale)
    return X, df, rate_col


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def choose_threshold_at_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, thr = roc_curve(y_true, y_score)

    finite_mask = np.isfinite(thr)
    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thr = thr[finite_mask]

    if len(thr) == 0:
        return 0.5

    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        best = int(np.argmin(fpr))
        return float(thr[best])

    best = ok[np.argmax(tpr[ok])]
    return float(thr[best])


def compute_binary_metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    is_h = (y_true == 0)
    is_i = (y_true == 1)
    spec = float((y_pred[is_h] == 0).mean()) if np.any(is_h) else np.nan
    sens = float((y_pred[is_i] == 1).mean()) if np.any(is_i) else np.nan
    bal_acc = float(np.nanmean([spec, sens]))
    return acc, spec, sens, bal_acc


def compute_healthy_only_metrics(y_true, y_pred):
    """
    Healthy-only evaluation — acts as a disease-free (0% infection) test.
    Returns false-positive behaviour statistics.
    """
    is_h = (y_true == 0)
    n_h = int(np.sum(is_h))
    if n_h == 0:
        return None

    y_pred_h = y_pred[is_h]
    n_true_negative = int(np.sum(y_pred_h == 0))
    n_false_positive = int(np.sum(y_pred_h == 1))
    specificity = float(n_true_negative / n_h)
    false_positive_rate = float(n_false_positive / n_h)

    return {
        "accuracy": specificity,
        "healthy_acc": specificity,
        "infected_acc": np.nan,
        "balanced_accuracy": np.nan,
        "false_positive_rate": false_positive_rate,
        "n_false_positive": n_false_positive,
        "n_true_negative": n_true_negative,
        "n": n_h,
    }


# ---------------------------------------------------------------------------
# Sample weighting (Model A only)
# ---------------------------------------------------------------------------

def compute_rate_weights(y, rates, mode="pow025", max_infected_weight=3.0, eps=1e-12):
    w = np.ones_like(y, dtype=float)
    mask = (y == 1)

    if not np.any(mask):
        return w

    r = np.clip(rates[mask], eps, None)

    if mode == "none":
        inf_w = np.ones_like(r, dtype=float)
    elif mode == "pow025":
        inf_w = 1.0 / np.power(r, 0.25)
    elif mode == "sqrt":
        inf_w = 1.0 / np.sqrt(r)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    inf_w = np.minimum(inf_w, float(max_infected_weight))
    w[mask] = inf_w
    w /= w.mean()
    return w


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_presence_model(C=0.1, l1_ratio=0.3):
    """Model A — logistic regression detector."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            C=C,
            max_iter=4000,
            class_weight=None,
            random_state=SEED0,
        ),
    )


def build_rate_model(alpha):
    """Model B — ridge regression quantifier on log10(rate)."""
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, random_state=SEED0),
    )


# ---------------------------------------------------------------------------
# Buffered CSV writer
# ---------------------------------------------------------------------------

class BufferedSaver:
    def __init__(self, path, buffer_size=50):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.buffer = []
        self.header_written = self.path.exists() and self.path.stat().st_size > 0

    def add(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        pd.DataFrame(self.buffer).to_csv(
            self.path, mode="a", header=not self.header_written, index=False
        )
        self.header_written = True
        self.buffer = []


# ---------------------------------------------------------------------------
# Data sampling helpers
def sort_rate_prediction_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    required_cols = {"sim_rep", "fold", "true_rate_percent"}
    if not required_cols.issubset(df.columns):
        return

    df = df.assign(
        _sim_num=pd.to_numeric(df["sim_rep"], errors="coerce"),
        _fold_num=pd.to_numeric(df["fold"], errors="coerce"),
        _true_rate_num=pd.to_numeric(df["true_rate_percent"], errors="coerce"),
    )

    sort_cols = ["_sim_num", "sim_rep", "_fold_num", "fold", "_true_rate_num", "true_rate_percent"]
    if "sample_id" in df.columns:
        sort_cols.append("sample_id")

    df = df.sort_values(sort_cols, kind="stable", na_position="last")
    df = df.drop(columns=["_sim_num", "_fold_num", "_true_rate_num"])
    df.to_csv(path, index=False)

# ---------------------------------------------------------------------------

def get_balanced_rate_indices(df, rate_col, rng, fixed_n=None):
    """
    Build one balanced mixed-rate dataset for one sim_rep.

    Infected : equal n per infection rate (across all represented rates).
    Healthy  : total healthy count matched to total infected count.
    """
    y = (df["label"] == "infected").astype(int).values
    rates = df[rate_col].astype(float).values

    idx_h = np.where(y == 0)[0]
    idx_i = np.where(y == 1)[0]
    unique_rates = sorted(np.unique(rates[idx_i]))

    rate_to_idx = {}
    min_rate_n = None
    for rr in unique_rates:
        idx_rr = np.where((y == 1) & np.isclose(rates, rr))[0]
        if len(idx_rr) == 0:
            continue
        rate_to_idx[rr] = idx_rr
        min_rate_n = len(idx_rr) if min_rate_n is None else min(min_rate_n, len(idx_rr))

    if not rate_to_idx or len(idx_h) == 0:
        return np.array([], dtype=int), [], 0

    per_rate_n = min_rate_n
    if fixed_n is not None:
        per_rate_n = min(per_rate_n, int(fixed_n))

    if per_rate_n <= 0:
        return np.array([], dtype=int), [], 0

    represented_rates = sorted(rate_to_idx.keys())
    inf_sel = []
    for rr in represented_rates:
        inf_sel.append(rng.choice(rate_to_idx[rr], size=per_rate_n, replace=False))
    inf_sel = np.concatenate(inf_sel)

    total_inf = len(inf_sel)
    total_h = min(total_inf, len(idx_h))
    hea_sel = rng.choice(idx_h, size=total_h, replace=False)

    selected = np.concatenate([inf_sel, hea_sel])
    return selected, represented_rates, per_rate_n


def filter_training_indices_by_rates(y, rates, train_idx, allowed_rates=None, tol=1e-12):
    """
    Keep ALL healthy training samples.
    For infected training samples, keep only those with rates in allowed_rates.
    If allowed_rates is None, return all training indices unchanged.
    """
    if allowed_rates is None:
        return np.array(train_idx, dtype=int)

    train_idx = np.array(train_idx, dtype=int)
    y_tr = y[train_idx]
    r_tr = rates[train_idx]

    keep = np.zeros(len(train_idx), dtype=bool)
    keep[y_tr == 0] = True  # always keep healthy

    inf_pos = np.where(y_tr == 1)[0]
    if len(inf_pos) > 0:
        inf_rates = r_tr[inf_pos]
        match = np.zeros(len(inf_rates), dtype=bool)
        for rr in allowed_rates:
            match |= np.isclose(inf_rates, rr, atol=tol, rtol=0.0)
        keep[inf_pos] = match

    return train_idx[keep]


def make_strata(y, rates):
    return np.array([
        "healthy" if yy == 0 else f"infected_{rr:.8g}"
        for yy, rr in zip(y, rates)
    ])


# ---------------------------------------------------------------------------
# Core training function — TWO-MODEL split
# ---------------------------------------------------------------------------

def fit_models_with_threshold(Xs, ys, rs, train_idx, args, rng_seed):
    """
    Fit Model A (detector) and prepare the index set for Model B (quantifier).

    Steps
    -----
    1. Split outer-train into inner-fit and inner-val (for threshold selection).
    2. Filter inner-fit to detector_rates for Model A.
       Inner-fit with ALL rates is kept separately for Model B.
    3. Train Model A on the filtered inner-fit set with rate-based sample weights.
    4. Choose detection threshold on inner-val at target FPR.
    5. Return clf (Model A), threshold, tr_fit_A (Model A training indices),
       tr_fit_all (all-rate training indices for Model B), and tr_val.

    Model B is built by the caller using tr_fit_all to keep the function
    single-responsibility.
    """
    train_idx = np.array(train_idx, dtype=int)
    strata_train = make_strata(ys[train_idx], rs[train_idx])

    # --- inner train/val split ---
    use_fallback = False
    strata_counts = pd.Series(strata_train).value_counts()
    if np.any(strata_counts < 2):
        use_fallback = True

    if not use_fallback:
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.inner_val_frac,
            random_state=rng_seed,
        )
        fit_rel, val_rel = next(inner.split(Xs[train_idx], strata_train))
        tr_fit = train_idx[fit_rel]   # inner-fit (all rates, for Model B later)
        tr_val = train_idx[val_rel]
    else:
        tr_fit = train_idx
        tr_val = train_idx

    # --- Model A: filter inner-fit to detector rates only ---
    det_rates = effective_detector_rates(args)
    tr_fit_A = filter_training_indices_by_rates(
        y=ys,
        rates=rs,
        train_idx=tr_fit,
        allowed_rates=det_rates,
        tol=args.rate_tol,
    )

    # Safety: if filtering removed all infected samples, fall back to full tr_fit
    if len(np.unique(ys[tr_fit_A])) < 2:
        print(
            f"  [WARN] detector-train-rates filter left no infected samples; "
            f"falling back to all rates for Model A."
        )
        tr_fit_A = np.array(tr_fit, dtype=int)

    # --- Train Model A ---
    sw_fit = compute_rate_weights(
        ys[tr_fit_A],
        rs[tr_fit_A],
        mode=args.weight_mode,
        max_infected_weight=args.max_infected_weight,
    )

    clf = build_presence_model(C=args.logreg_c, l1_ratio=args.logreg_l1_ratio)
    clf.fit(Xs[tr_fit_A], ys[tr_fit_A], logisticregression__sample_weight=sw_fit)

    # --- Threshold on inner-val ---
    p_val = clf.predict_proba(Xs[tr_val])[:, 1]
    thr = choose_threshold_at_fpr(ys[tr_val], p_val, args.target_fpr)

    # tr_fit (unrestricted inner-fit) is returned for Model B
    return clf, thr, tr_fit_A, tr_fit, tr_val


# ---------------------------------------------------------------------------
# Output row builders  (schema identical to v5, plus two new label columns)
# ---------------------------------------------------------------------------

def _detection_base(sim, fold, args, thr, n_train_fit_A, n_train_val, n_test):
    """Shared fields for every detection output row."""
    return {
        "sim_rep": sim,
        "fold": fold,
        "input_scale": args.input_scale,
        "norm_method": args.norm if args.input_scale == "counts" else "precomputed_log1pcpm",
        "threshold": thr,
        "target_fpr": args.target_fpr,
        "n_train_fit": int(n_train_fit_A),
        "n_train_val": int(n_train_val),
        "n_test": int(n_test),
        # v5 backward-compat column — reports what Model A was trained on
        "train_rates": detector_rates_label(args),
        # new v6 columns
        "detector_train_rates": detector_rates_label(args),
        "regressor_train_rates": "all",
        "weight_mode": args.weight_mode,
        "max_infected_weight": args.max_infected_weight,
        "logreg_c": args.logreg_c,
    }


def add_detection_rows(out1, sim, fold, args, ys_te, rs_te, yhat, thr,
                       represented_rates, n_train_fit_A, n_train_val, n_test,
                       rng):
    """
    Save detection results:
      - overall_detection        : metrics across the full test fold
      - healthy_only_detection   : false-positive behaviour on healthy samples only
      - balanced_detection_by_rate : per infection-rate balanced evaluation
    """
    base = _detection_base(sim, fold, args, thr, n_train_fit_A, n_train_val, n_test)

    # --- overall detection ---
    acc, spec, sens, bal_acc = compute_binary_metrics(ys_te, yhat)
    overall_fp = int(np.sum((ys_te == 0) & (yhat == 1)))
    overall_tn = int(np.sum((ys_te == 0) & (yhat == 0)))
    overall_fpr = float(overall_fp / np.sum(ys_te == 0)) if np.any(ys_te == 0) else np.nan

    out1.add({
        **base,
        "metric": "overall_detection",
        "rate_percent": "all",
        "accuracy": acc,
        "healthy_acc": spec,
        "infected_acc": sens,
        "balanced_accuracy": bal_acc,
        "false_positive_rate": overall_fpr,
        "n_false_positive": overall_fp,
        "n_true_negative": overall_tn,
        "n": int(len(ys_te)),
    })

    # --- healthy-only evaluation (0% infection, disease-free test) ---
    hm = compute_healthy_only_metrics(ys_te, yhat)
    if hm is not None:
        out1.add({
            **base,
            "metric": "healthy_only_detection",
            "rate_percent": 0.0,
            "accuracy": hm["accuracy"],
            "healthy_acc": hm["healthy_acc"],
            "infected_acc": hm["infected_acc"],
            "balanced_accuracy": hm["balanced_accuracy"],
            "false_positive_rate": hm["false_positive_rate"],
            "n_false_positive": hm["n_false_positive"],
            "n_true_negative": hm["n_true_negative"],
            "n": hm["n"],
        })

    # --- per-rate balanced detection ---
    idx_hea = np.where(ys_te == 0)[0]
    idx_inf = np.where(ys_te == 1)[0]
    if len(idx_hea) == 0 or len(idx_inf) == 0:
        return

    for rr in represented_rates:
        idx_inf_rr = idx_inf[np.isclose(rs_te[idx_inf], rr)]
        if len(idx_inf_rr) == 0:
            continue

        n = min(len(idx_inf_rr), len(idx_hea))
        if n == 0:
            continue

        if len(idx_inf_rr) <= len(idx_hea):
            inf_sel = np.array(idx_inf_rr, dtype=int)
            hea_sel = rng.choice(idx_hea, size=len(inf_sel), replace=False)
        else:
            inf_sel = rng.choice(idx_inf_rr, size=n, replace=False)
            hea_sel = np.array(idx_hea, dtype=int)

        eval_idx = np.concatenate([inf_sel, hea_sel])
        y_eval = ys_te[eval_idx]
        yhat_eval = yhat[eval_idx]
        acc_rr, spec_rr, sens_rr, bal_acc_rr = compute_binary_metrics(y_eval, yhat_eval)

        fp_rr = int(np.sum((y_eval == 0) & (yhat_eval == 1)))
        tn_rr = int(np.sum((y_eval == 0) & (yhat_eval == 0)))
        fpr_rr = float(fp_rr / np.sum(y_eval == 0)) if np.any(y_eval == 0) else np.nan

        out1.add({
            **base,
            "metric": "balanced_detection_by_rate",
            "rate_percent": float(rr) * 100.0,
            "accuracy": acc_rr,
            "healthy_acc": spec_rr,
            "infected_acc": sens_rr,
            "balanced_accuracy": bal_acc_rr,
            "false_positive_rate": fpr_rr,
            "n_false_positive": fp_rr,
            "n_true_negative": tn_rr,
            "n": int(len(eval_idx)),
        })


def add_rate_prediction_rows(out_pred, sim, fold, args, sample_ids,
                              rs_true, yreg_pred, n_train_fit_reg, n_test):
    """Save per-sample rate predictions (Model B output)."""
    true_log = np.log10(np.clip(rs_true, 1e-12, None))
    pred_rate = np.power(10.0, yreg_pred)

    for sid, rr_true, rr_true_log, rr_pred_log, rr_pred in zip(
        sample_ids, rs_true, true_log, yreg_pred, pred_rate
    ):
        out_pred.add({
            "sim_rep": sim,
            "fold": fold,
            "sample_id": sid,
            "input_scale": args.input_scale,
            "norm_method": args.norm if args.input_scale == "counts" else "precomputed_log1pcpm",
            "metric": f"predicted_log10_rate_test_{rate_test_set_label(args)}",
            "rate_percent": float(rr_true) * 100.0,
            "true_rate": float(rr_true),
            "true_rate_percent": float(rr_true) * 100.0,
            "true_log10_rate": float(rr_true_log),
            "pred_log10_rate": float(rr_pred_log),
            "pred_rate": float(rr_pred),
            "pred_rate_percent": float(rr_pred) * 100.0,
            "abs_error_log10": float(abs(rr_true_log - rr_pred_log)),
            "abs_error_rate": float(abs(rr_true - rr_pred)),
            "n_train_fit": int(n_train_fit_reg),
            "n_test": int(n_test),
            # v5 compat: regressor always uses all rates
            "train_rates": "all",
            "detector_train_rates": regressor_detector_label(),
            "regressor_train_rates": "all",
            "rate_test_set": rate_test_set_label(args),
        })


def add_rate_mae_rows(out_mae, sim, fold, args, rs_true, yreg_pred,
                      represented_rates, n_train_fit_reg, n_test):
    """Save overall + per-rate MAE summary for Model B predictions."""
    true_log = np.log10(np.clip(rs_true, 1e-12, None))
    pred_rate = np.power(10.0, yreg_pred)

    mae_log10_all = float(np.mean(np.abs(true_log - yreg_pred)))
    mae_rate_all = float(np.mean(np.abs(rs_true - pred_rate)))

    base_reg = {
        "sim_rep": sim,
        "fold": fold,
        "input_scale": args.input_scale,
        "norm_method": args.norm if args.input_scale == "counts" else "precomputed_log1pcpm",
        "metric": f"MAE_rate_test_{rate_test_set_label(args)}",
        "n_train_fit": int(n_train_fit_reg),
        "n_test": int(n_test),
        "train_rates": "all",
        "detector_train_rates": regressor_detector_label(),
        "regressor_train_rates": "all",
        "rate_test_set": rate_test_set_label(args),
    }

    out_mae.add({
        **base_reg,
        "rate_percent": "all",
        "mae_log10_rate": mae_log10_all,
        "mae_rate": mae_rate_all,
        "n": int(len(rs_true)),
    })

    for rr in represented_rates:
        mask_rr = np.isclose(rs_true, rr)
        if np.sum(mask_rr) == 0:
            continue

        rr_true_arr = rs_true[mask_rr]
        rr_true_log = true_log[mask_rr]
        rr_pred_log = yreg_pred[mask_rr]
        rr_pred_rate = pred_rate[mask_rr]

        out_mae.add({
            **base_reg,
            "rate_percent": float(rr) * 100.0,
            "mae_log10_rate": float(np.mean(np.abs(rr_true_log - rr_pred_log))),
            "mae_rate": float(np.mean(np.abs(rr_true_arr - rr_pred_rate))),
            "n": int(np.sum(mask_rr)),
            "n_test": int(np.sum(mask_rr)),
        })


# ---------------------------------------------------------------------------
# Internal CV mode
# ---------------------------------------------------------------------------

def run_internal(args, X, df, rate_col):
    detector_arg_sets = build_detector_arg_sets(args)
    reg_suffix = regressor_training_suffix()

    detection_outputs = {
        label: BufferedSaver(
            Path(args.outdir) / f"step5_internal_detection_{label}.csv",
            BUFFER_SIZE,
        )
        for label, _ in detector_arg_sets
    }
    out2_pred = BufferedSaver(
        Path(args.outdir) / f"step5_internal_rate_predictions_{reg_suffix}.csv",
        BUFFER_SIZE,
    )
    out2_mae = BufferedSaver(
        Path(args.outdir) / f"step5_internal_rate_summary_{reg_suffix}.csv",
        BUFFER_SIZE,
    )

    y_all = (df["label"] == "infected").astype(int).values
    rates_all = df[rate_col].astype(float).values
    sample_ids_all = df["sample_id"].astype(str).values

    print("\n[INFO] Model A (detector) variants:")
    for label, det_args in detector_arg_sets:
        print(f"  - {label}: {detector_rates_label(det_args)}")
    print("[INFO] Model B (regressor) training rates: all infected in each outer fold")

    for sim in range(N_SIM_REPEATS):
        rng = np.random.RandomState(SEED0 + sim)

        selected_idx, represented_rates, per_rate_n = get_balanced_rate_indices(
            df=df,
            rate_col=rate_col,
            rng=rng,
            fixed_n=args.fixed_n,
        )
        if len(selected_idx) == 0:
            continue

        Xs = X[selected_idx]
        ys = y_all[selected_idx]
        rs = rates_all[selected_idx]
        ids_s = sample_ids_all[selected_idx]

        strata = make_strata(ys, rs)

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED0 + sim,
        )

        for fold, (tr, te) in enumerate(cv.split(Xs, strata)):
            shared_tr_fit = None

            for variant_idx, (label, det_args) in enumerate(detector_arg_sets):
                clf, thr, tr_fit_A, tr_fit, tr_val = fit_models_with_threshold(
                    Xs=Xs,
                    ys=ys,
                    rs=rs,
                    train_idx=tr,
                    args=det_args,
                    rng_seed=SEED0 + 10000 * sim + fold,
                )

                p_te = clf.predict_proba(Xs[te])[:, 1]
                yhat = (p_te >= thr).astype(int)

                fold_rng = np.random.RandomState(SEED0 + 100000 * sim + fold + variant_idx)

                add_detection_rows(
                    out1=detection_outputs[label],
                    sim=sim,
                    fold=fold,
                    args=det_args,
                    ys_te=ys[te],
                    rs_te=rs[te],
                    yhat=yhat,
                    thr=thr,
                    represented_rates=represented_rates,
                    n_train_fit_A=len(tr_fit_A),
                    n_train_val=len(tr_val),
                    n_test=len(te),
                    rng=fold_rng,
                )

                if shared_tr_fit is None:
                    shared_tr_fit = tr_fit

            tr_inf_all = shared_tr_fit[ys[shared_tr_fit] == 1]
            te_sel_local = select_rate_test_indices(ys[te], args.rate_test_set)
            te_sel = te[te_sel_local]

            if len(tr_inf_all) >= 5 and len(te_sel) > 0:
                reg = build_rate_model(args.ridge_alpha)
                yreg_tr = np.log10(np.clip(rs[tr_inf_all], 1e-12, None))
                reg.fit(Xs[tr_inf_all], yreg_tr)

                yreg_pred = reg.predict(Xs[te_sel])

                add_rate_prediction_rows(
                    out_pred=out2_pred,
                    sim=sim,
                    fold=fold,
                    args=args,
                    sample_ids=ids_s[te_sel],
                    rs_true=rs[te_sel],
                    yreg_pred=yreg_pred,
                    n_train_fit_reg=len(tr_inf_all),
                    n_test=len(te_sel),
                )

                add_rate_mae_rows(
                    out_mae=out2_mae,
                    sim=sim,
                    fold=fold,
                    args=args,
                    rs_true=rs[te_sel],
                    yreg_pred=yreg_pred,
                    represented_rates=sorted(np.unique(rs[te_sel])),
                    n_train_fit_reg=len(tr_inf_all),
                    n_test=len(te_sel),
                )

        del Xs, ys, rs, ids_s
        gc.collect()

    for saver in detection_outputs.values():
        saver.flush()
    out2_pred.flush()
    sort_rate_prediction_csv(out2_pred.path)
    out2_mae.flush()
    print(f"\n[DONE] Internal results saved to: {args.outdir}")


# ---------------------------------------------------------------------------
# External validation mode
# ---------------------------------------------------------------------------

def run_external(args, Xtr, dftr, rate_col_tr, Xte, dfte, rate_col_te):
    detector_arg_sets = build_detector_arg_sets(args)
    reg_suffix = regressor_training_suffix()

    detection_outputs = {
        label: BufferedSaver(
            Path(args.outdir) / f"step5_external_detection_{label}.csv",
            BUFFER_SIZE,
        )
        for label, _ in detector_arg_sets
    }
    out2_pred = BufferedSaver(
        Path(args.outdir) / f"step5_external_rate_predictions_{reg_suffix}.csv",
        BUFFER_SIZE,
    )
    out2_mae = BufferedSaver(
        Path(args.outdir) / f"step5_external_rate_summary_{reg_suffix}.csv",
        BUFFER_SIZE,
    )

    ytr = (dftr["label"] == "infected").astype(int).values
    rtr = dftr[rate_col_tr].astype(float).values
    yte = (dfte["label"] == "infected").astype(int).values
    rte = dfte[rate_col_te].astype(float).values
    ids_te = dfte["sample_id"].astype(str).values

    print("\n[INFO] Model A (detector) variants:")
    for label, det_args in detector_arg_sets:
        print(f"  - {label}: {detector_rates_label(det_args)}")
    print("[INFO] Model B (regressor) training rates: all infected in training set")

    train_idx_all = np.arange(len(ytr), dtype=int)
    shared_tr_fit = None

    for variant_idx, (label, det_args) in enumerate(detector_arg_sets):
        clf, thr, tr_fit_A, tr_fit, tr_val = fit_models_with_threshold(
            Xs=Xtr,
            ys=ytr,
            rs=rtr,
            train_idx=train_idx_all,
            args=det_args,
            rng_seed=SEED0 + 999,
        )

        p_te = clf.predict_proba(Xte)[:, 1]
        yhat = (p_te >= thr).astype(int)

        ext_rng = np.random.RandomState(SEED0 + 999999 + variant_idx)

        add_detection_rows(
            out1=detection_outputs[label],
            sim="external",
            fold="external",
            args=det_args,
            ys_te=yte,
            rs_te=rte,
            yhat=yhat,
            thr=thr,
            represented_rates=sorted(np.unique(rte[yte == 1])),
            n_train_fit_A=len(tr_fit_A),
            n_train_val=len(tr_val),
            n_test=len(yte),
            rng=ext_rng,
        )

        if shared_tr_fit is None:
            shared_tr_fit = tr_fit

    tr_inf_all = shared_tr_fit[ytr[shared_tr_fit] == 1]
    te_sel = select_rate_test_indices(yte, args.rate_test_set)

    if len(tr_inf_all) >= 5 and len(te_sel) > 0:
        reg = build_rate_model(args.ridge_alpha)
        yreg_tr = np.log10(np.clip(rtr[tr_inf_all], 1e-12, None))
        reg.fit(Xtr[tr_inf_all], yreg_tr)

        yreg_pred = reg.predict(Xte[te_sel])

        add_rate_prediction_rows(
            out_pred=out2_pred,
            sim="external",
            fold="external",
            args=args,
            sample_ids=ids_te[te_sel],
            rs_true=rte[te_sel],
            yreg_pred=yreg_pred,
            n_train_fit_reg=len(tr_inf_all),
            n_test=len(te_sel),
        )

        add_rate_mae_rows(
            out_mae=out2_mae,
            sim="external",
            fold="external",
            args=args,
            rs_true=rte[te_sel],
            yreg_pred=yreg_pred,
            represented_rates=sorted(np.unique(rte[te_sel])),
            n_train_fit_reg=len(tr_inf_all),
            n_test=len(te_sel),
        )

    for saver in detection_outputs.values():
        saver.flush()
    out2_pred.flush()
    sort_rate_prediction_csv(out2_pred.path)
    out2_mae.flush()
    print(f"\n[DONE] External results saved to: {args.outdir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("\n==========================================")
    print(" Step 5 v6 — Two-model detection + rate  ")
    print("==========================================")

    try:
        if args.mode == "internal":
            if not args.metadata or not args.matrix or not args.ids:
                raise ValueError("Internal mode requires --metadata --matrix --ids")
            X, df, c = load_dataset(
                args.matrix, args.ids, args.metadata, args.norm, args.input_scale
            )
            run_internal(args, X, df, c)

        elif args.mode == "external":
            required = [
                args.metadata,
                args.train_matrix,
                args.train_ids,
                args.test_metadata,
                args.test_matrix,
                args.test_ids,
            ]
            if not all(required):
                raise ValueError(
                    "External mode requires --metadata, --train-matrix, --train-ids, "
                    "--test-metadata, --test-matrix, --test-ids."
                )

            Xtr, dftr, ctr = load_dataset(
                args.train_matrix,
                args.train_ids,
                args.metadata,
                args.norm,
                args.input_scale,
            )
            Xte, dfte, cte = load_dataset(
                args.test_matrix,
                args.test_ids,
                args.test_metadata,
                args.norm,
                args.input_scale,
            )
            run_external(args, Xtr, dftr, ctr, Xte, dfte, cte)

        print(f"\n[DONE] All results saved to: {args.outdir}")
        print("==========================================\n")

    except Exception as e:
        traceback.print_exc()
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


