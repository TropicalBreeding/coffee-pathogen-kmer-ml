#!/usr/bin/env python3

"""
K-mer Machine-Learning Benchmark (Steps 1–4)
=============================================

Description
-----------
Benchmarks ten classification algorithms on k-mer feature matrices for
pathogen detection across four experimental scenarios:

  **Step 1** — Same-rate internal cross-validation.  For each infection rate,
  samples are balanced (equal healthy / infected) and evaluated with
  repeated stratified 10-fold CV.

  **Step 2** — Raw versus host-depleted comparison.  Both undepleted and
  KrakenUniq-depleted k-mer matrices can be supplied; each is evaluated
  under the same CV protocol.

  **Step 3** — Cross-rate external validation.  Models are trained on one or
  more infection rates and tested on all remaining rates using an
  independent held-out set.

  **Step 4** — Cultivar / pathogen transferability.  An external train/test
  split across biological conditions (e.g., train on ET-39 Rust, test on
  Typica Wilt) quantifies classifier generalisability.

Classifiers evaluated: Logistic Regression, Gradient Boosting, Random
Forest, Extra Trees, AdaBoost, MLP, SVM (RBF and linear kernels),
k-Nearest Neighbours, Gaussian Naive Bayes, and a 1-D CNN (TensorFlow).

Dependencies
------------
- Python >= 3.8
- NumPy, pandas, scikit-learn, TensorFlow >= 2.x

Inputs
------
- ``metadata.csv`` with columns: ``sample_id``, ``label``, and an
  infection-rate column (``target_rate``, ``targetrate``, ``rate``, or
  ``target_rate_fraction``)
- ``sample_ids.txt`` — ordered list of sample identifiers matching matrix
  row order
- ``matrix.npy`` — dense k-mer count matrix (samples x 4^k)

Input scale
-----------
``--input-scale counts``
    Matrix contains raw k-mer counts; the transform specified by ``--norm``
    (``log1p_cpm`` or ``log1p_raw``) is applied before modelling.

``--input-scale log1pcpm``
    Matrix is already log1p(CPM)-transformed; used as-is.

Outputs
-------
- ``benchmark_internal_long_*.csv``  — per-fold accuracy, healthy accuracy,
  and infected accuracy for internal CV
- ``benchmark_external_long_*.csv``  — same metrics for external validation
- ``kmer_importance_*.csv``          — top-N discriminative k-mers per model
  (optional, ``--export-kmers``)

Examples
--------
1. Internal 10-fold CV on 7-mer raw counts (all models)::

     python 4_ml_benchmark.py \\
       --outdir results_step1 \\
       --k 7 \\
       --mode internal \\
       --metadata metadata.csv \\
       --raw-matrix results_kmer/matrix_k7.npy \\
       --raw-ids results_kmer/sample_ids.txt \\
       --input-scale counts \\
       --norm log1p_cpm

2. External cross-rate validation — train on 20 % rate, test on all rates::

     python 4_ml_benchmark.py \\
       --outdir results_step3 \\
       --k 7 \\
       --mode external \\
       --metadata metadata_train.csv \\
       --test-metadata metadata_test.csv \\
       --train-matrix matrix_train_k7.npy \\
       --train-ids train_ids.txt \\
       --test-matrix matrix_test_k7.npy \\
       --test-ids test_ids.txt \\
       --ext-train-rates 0.20

3. Skip CNN and slow models for a quick benchmark::

     python 4_ml_benchmark.py \\
       --outdir results_quick \\
       --k 7 \\
       --mode internal \\
       --metadata metadata.csv \\
       --raw-matrix matrix_k7.npy \\
       --raw-ids sample_ids.txt \\
       --no-cnn \\
       --no-slow-models

4. Export top-50 discriminative k-mers per model::

     python 4_ml_benchmark.py \\
       --outdir results_kmers \\
       --k 7 \\
       --mode internal \\
       --metadata metadata.csv \\
       --raw-matrix matrix_k7.npy \\
       --raw-ids sample_ids.txt \\
       --export-kmers \\
       --kmer-top-n 50
"""

import os
import sys
import gc
import argparse
import traceback
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


RATES = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]

SEED0 = 42
N_SPLITS = 10
N_CV_REPEATS = 1
N_SIM_REPEATS = 10

CNN_EPOCHS = 10
CNN_BATCH = 256
BUFFER_SIZE = 50

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
tf.get_logger().setLevel("ERROR")


def parse_args():
    p = argparse.ArgumentParser(description="K-mer ML benchmark for steps 1-4")

    p.add_argument("--outdir", required=True)
    p.add_argument("--k", type=int, choices=[7, 11], required=True)
    p.add_argument("--taxid", type=int, default=33090)
    p.add_argument("--mode", choices=["internal", "external"], default="internal")

    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--raw-matrix", type=str, default=None)
    p.add_argument("--raw-ids", type=str, default=None)
    p.add_argument("--dep-matrix", type=str, default=None)
    p.add_argument("--dep-ids", type=str, default=None)

    p.add_argument("--test-metadata", type=str, default=None)
    p.add_argument("--train-matrix", type=str, default=None)
    p.add_argument("--train-ids", type=str, default=None)
    p.add_argument("--test-matrix", type=str, default=None)
    p.add_argument("--test-ids", type=str, default=None)

    p.add_argument("--ext-train-boots", type=int, default=10)
    p.add_argument("--ext-test-boots", type=int, default=0)
    p.add_argument("--ext-train-rates", type=str, default=None)
    p.add_argument("--ext-train-unbalanced", action="store_true")

    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--no-slow-models", action="store_true")
    p.add_argument("--no-memmap", action="store_true")
    p.add_argument("--fixed-n", type=int, default=None)

    p.add_argument("--export-kmers", action="store_true")
    p.add_argument("--kmer-top-n", type=int, default=50)
    p.add_argument("--models", type=str, default=None)

    p.add_argument("--norm", choices=["log1p_cpm", "log1p_raw"], default="log1p_cpm",
                   help="Used only when --input-scale counts.")
    p.add_argument("--input-scale", choices=["counts", "log1pcpm"], default="counts",
                   help="Whether input .npy is raw counts or already log1pCPM-transformed.")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    return p.parse_args()


def maybe_cast(X):
    return X.astype(np.float32, copy=False)


def read_id_list(fp: Path):
    with open(fp, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_matrix(fp: Path, use_memmap: bool):
    return np.load(fp, mmap_mode="r") if use_memmap else np.load(fp)


def detect_rate_col(df):
    for c in ["target_rate", "targetrate", "rate", "target_rate_fraction"]:
        if c in df.columns:
            return c
    raise ValueError("Metadata missing infection-rate column.")


def compute_metrics(y_true, y_pred):
    accuracy = float((y_pred == y_true).mean())
    is_healthy = (y_true == 0)
    is_infected = (y_true == 1)
    healthy_accuracy = float((y_pred[is_healthy] == 0).mean()) if np.any(is_healthy) else np.nan
    infected_accuracy = float((y_pred[is_infected] == 1).mean()) if np.any(is_infected) else np.nan
    return accuracy, healthy_accuracy, infected_accuracy


def index_to_kmer(idx: int, k: int) -> str:
    limit = 4 ** k
    if idx >= limit:
        return f"Hash{idx}"
    bases = ["A", "C", "G", "T"]
    out = []
    for _ in range(k):
        out.append(bases[idx % 4])
        idx //= 4
    return "".join(reversed(out))


def unwrap_estimator(model):
    if hasattr(model, "named_steps"):
        return model.steps[-1][1]
    return model


def parse_rate_list(s: str):
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out if out else None


def find_file(base: Path, pattern: str):
    p1 = base / pattern
    if p1.exists():
        return p1
    if not pattern.startswith("X_"):
        p2 = base / f"X_{pattern}"
        if p2.exists():
            return p2
    return None


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
    """
    If input_scale=log1pcpm, matrix is already transformed and returned as-is.
    If input_scale=counts, apply requested transform.
    """
    X = maybe_cast(X)
    if input_scale == "log1pcpm":
        return X
    if norm_method == "log1p_cpm":
        return transform_log1p_cpm(X)
    if norm_method == "log1p_raw":
        return transform_log1p_raw(X)
    raise ValueError(f"Unknown norm method: {norm_method}")


def needs_scaler(model_name):
    return model_name in {"LR", "SVM", "LSVM", "KNN", "MLP", "CNN"}


class BufferedSaver:
    def __init__(self, path, buffer_size=50):
        self.path = path
        self.buffer_size = buffer_size
        self.buffer = []
        self.header_written = path.exists() and path.stat().st_size > 0

    def add(self, data_dict):
        self.buffer.append(data_dict)
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


def load_and_align_data(k: int, taxid: int, use_memmap: bool,
                        metadata: str = None,
                        raw_matrix: str = None, raw_ids: str = None,
                        dep_matrix: str = None, dep_ids: str = None):

    if metadata:
        meta_fp = Path(metadata)
        if not meta_fp.exists():
            raise FileNotFoundError(f"No metadata file found at {meta_fp}")
        base = meta_fp.parent
    else:
        base = Path.cwd()
        meta_fp = base / "metadata.csv"
        if not meta_fp.exists():
            raise FileNotFoundError("No metadata provided and metadata.csv not found in cwd")

    km_dir = base / "results_kmer"
    if not km_dir.exists():
        km_dir = base

    print(f"[INFO] Scanning directory: {km_dir}")

    df = pd.read_csv(meta_fp, dtype=str)
    rate_col = detect_rate_col(df)
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)

    raw_path = Path(raw_matrix) if raw_matrix else (
        find_file(km_dir, f"A0_undepleted_k{k}.npy") or
        find_file(km_dir, f"matrix_k{k}_A0_undepleted.npy")
    )
    dep_path = Path(dep_matrix) if dep_matrix else (
        find_file(km_dir, f"B{taxid}_k{k}.npy") or
        find_file(km_dir, f"matrix_k{k}_B{taxid}.npy")
    )

    if not dep_path and taxid == 33090:
        dep_path = find_file(km_dir, f"matrix_k{k}_B33090.npy")
    if not dep_path and taxid == 13443:
        dep_path = find_file(km_dir, f"matrix_k{k}_B13443.npy")

    X_raw, X_dep = None, None
    raw_ids_list, dep_ids_list = None, None

    if raw_path:
        print(f"  -> Found Raw Matrix: {Path(raw_path).name}")
        X_raw = load_matrix(Path(raw_path), use_memmap)
        if raw_ids:
            raw_ids_list = read_id_list(Path(raw_ids))
            print(f"     (IDs: {Path(raw_ids).name})")

    if dep_path:
        print(f"  -> Found Kraken Matrix: {Path(dep_path).name}")
        X_dep = load_matrix(Path(dep_path), use_memmap)
        if dep_ids:
            dep_ids_list = read_id_list(Path(dep_ids))
            print(f"     (IDs: {Path(dep_ids).name})")

    final_ids = None
    if raw_ids_list and dep_ids_list:
        if raw_ids_list != dep_ids_list:
            raise RuntimeError("Sample ID mismatch between Raw and Kraken matrices.")
        final_ids = raw_ids_list
    elif raw_ids_list:
        final_ids = raw_ids_list
    elif dep_ids_list:
        final_ids = dep_ids_list

    if final_ids:
        df = df.set_index("sample_id").reindex(final_ids).reset_index()
        if df["label"].isna().any():
            missing = df["sample_id"][df["label"].isna()].head(3).tolist()
            raise RuntimeError(f"Metadata missing labels after reindexing. Example IDs: {missing}")

        if X_raw is not None and len(final_ids) != X_raw.shape[0]:
            raise RuntimeError("Raw matrix rows do not match ID count.")
        if X_dep is not None and len(final_ids) != X_dep.shape[0]:
            raise RuntimeError("Dep matrix rows do not match ID count.")
    else:
        print("[WARN] No ID list provided. Assuming metadata order matches matrix order.")
        if X_raw is not None and len(df) != X_raw.shape[0]:
            raise RuntimeError("Metadata rows do not match Raw matrix rows.")
        if X_dep is not None and len(df) != X_dep.shape[0]:
            raise RuntimeError("Metadata rows do not match Dep matrix rows.")

    return X_raw, X_dep, df, rate_col


class CNN_Model(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=5, batch_size=32, verbose=0, seed=42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        inp = tf.keras.layers.Input(shape=(self.input_dim, 1))
        x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self.model_ = tf.keras.Model(inputs=inp, outputs=out)
        self.model_.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model_.fit(
            X.reshape(-1, self.input_dim, 1), y,
            epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose
        )
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        p = self.model_.predict(X.reshape(-1, self.input_dim, 1), verbose=0)
        return np.hstack([1 - p, p])


def build_models(args):
    cw = "balanced" if args.class_weight == "balanced" else None

    models = {
        "LR": LogisticRegression(max_iter=1000, class_weight=cw),
        "RF": RandomForestClassifier(n_jobs=N_JOBS, class_weight=cw, random_state=SEED0),
        "ET": ExtraTreesClassifier(n_jobs=N_JOBS, class_weight=cw, random_state=SEED0),
        "GB": GradientBoostingClassifier(random_state=SEED0),
        "NB": GaussianNB(),
    }

    if not args.no_slow_models:
        models["KNN"] = KNeighborsClassifier(n_jobs=N_JOBS)
        models["SVM"] = SVC(probability=True, class_weight=cw, random_state=SEED0)
        models["MLP"] = MLPClassifier(max_iter=500, random_state=SEED0)
        models["AB"] = AdaBoostClassifier(random_state=SEED0)
        models["LSVM"] = LinearSVC(dual="auto", max_iter=5000, class_weight=cw, random_state=SEED0)

    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in models]
        if unknown:
            raise ValueError(f"Unknown models in --models: {unknown}. Available: {sorted(models.keys())}")
        models = {m: models[m] for m in wanted}

    return models


def extract_kmers(model, X, y, top_n=50, k=11):
    model = unwrap_estimator(model)
    importances = None

    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is None:
        return []

    indices = np.argsort(importances)[::-1][:top_n]
    out = []
    for idx in indices:
        mean_inf = X[y == 1, idx].mean()
        mean_hea = X[y == 0, idx].mean()
        direction = "Infected" if mean_inf > mean_hea else "Healthy"
        out.append({
            "kmer": index_to_kmer(idx, k),
            "importance": float(importances[idx]),
            "direction": direction,
        })
    return out


def get_balanced_indices(df, rate_col, target_rate, rng, fixed_n=None):
    rates = df[rate_col].astype(float).values
    mask_rate = np.isclose(rates, target_rate)

    idx_inf = np.where((df["label"] == "infected") & mask_rate)[0]
    idx_hea = np.where(df["label"] == "healthy")[0]

    if len(idx_inf) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n = min(len(idx_inf), len(idx_hea))
    if fixed_n is not None:
        n = min(n, int(fixed_n))

    return rng.choice(idx_inf, n, replace=False), rng.choice(idx_hea, n, replace=False)


def get_balanced_indices_with_replace(df, rate_col, target_rate, rng, fixed_n=None, replace=True):
    rates = df[rate_col].astype(float).values
    mask_rate = np.isclose(rates, target_rate)

    idx_inf = np.where((df["label"] == "infected") & mask_rate)[0]
    idx_hea = np.where(df["label"] == "healthy")[0]

    if len(idx_inf) == 0 or len(idx_hea) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n = min(len(idx_inf), len(idx_hea))
    if fixed_n is not None:
        n = min(n, int(fixed_n))

    return rng.choice(idx_inf, n, replace=replace), rng.choice(idx_hea, n, replace=replace)


def get_balanced_indices_mixed_rates(df, rate_col, train_rates, rng, fixed_n=None, replace=True):
    rates_all = df[rate_col].astype(float).values
    idx_hea = np.where(df["label"] == "healthy")[0]

    if len(idx_hea) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx_inf_pool = []
    for r in train_rates:
        idx = np.where((df["label"] == "infected") & np.isclose(rates_all, r))[0]
        if len(idx) > 0:
            idx_inf_pool.append(idx)

    if not idx_inf_pool:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx_inf_pool = np.concatenate(idx_inf_pool)

    n = min(len(idx_inf_pool), len(idx_hea))
    if fixed_n is not None:
        n = min(n, int(fixed_n))
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx_inf = rng.choice(idx_inf_pool, size=n, replace=replace)
    idx_hea_s = rng.choice(idx_hea, size=n, replace=replace)
    return idx_inf, idx_hea_s


def get_train_pools_for_rates(df, rate_col, train_rates):
    rates_all = df[rate_col].astype(float).values
    idx_hea_pool = np.where(df["label"] == "healthy")[0]

    inf_list = []
    for r in train_rates:
        idx = np.where((df["label"] == "infected") & np.isclose(rates_all, r))[0]
        if len(idx) > 0:
            inf_list.append(idx)

    idx_inf_pool = np.concatenate(inf_list) if inf_list else np.array([], dtype=int)
    return idx_inf_pool, idx_hea_pool


def stable_seed_from_text(*parts, modulo=1_000_000):
    """Create a deterministic integer seed offset from one or more text parts."""
    txt = "||".join([str(p) for p in parts if p is not None])
    digest = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def eval_internal(args, X_raw, X_dep, df, rate_col, models):
    print(f"\n>>> STARTING INTERNAL CV | CPUS={N_JOBS}")

    idx_inf_all = np.where(df["label"] == "infected")[0]
    avail_rates = sorted(df.iloc[idx_inf_all][rate_col].astype(float).unique())
    target_rates = [r for r in RATES if np.any(np.isclose(avail_rates, r))]
    if not target_rates:
        target_rates = avail_rates

    features = {}
    if X_raw is not None:
        features["Raw_full"] = X_raw
    if X_dep is not None:
        features["Kraken_full"] = X_dep

    rng = np.random.RandomState(SEED0)

    res_saver = BufferedSaver(Path(args.outdir) / f"benchmark_internal_long_{args.input_scale}_{args.norm}.csv", BUFFER_SIZE)
    kmer_saver = BufferedSaver(Path(args.outdir) / f"kmer_importance_{args.input_scale}_{args.norm}.csv", BUFFER_SIZE)

    for rate in target_rates:
        print(f"\n[INFO] Processing Rate: {rate * 100}%")

        for sim_i in range(N_SIM_REPEATS):
            idx_inf, idx_hea = get_balanced_indices(df, rate_col, rate, rng, args.fixed_n)
            if len(idx_inf) == 0:
                continue

            y = np.array([1] * len(idx_inf) + [0] * len(idx_hea))
            cv = RepeatedStratifiedKFold(
                n_splits=N_SPLITS, n_repeats=N_CV_REPEATS, random_state=sim_i
            )

            for f_name, X_full in features.items():
                print(f"  - Feature: {f_name} | Simulation: {sim_i + 1}/{N_SIM_REPEATS}")

                X_stack = np.vstack([X_full[idx_inf], X_full[idx_hea]])
                X_data = preprocess_matrix(X_stack, args.norm, args.input_scale)

                local_models = models.copy()
                if (not args.no_slow_models) and (not args.no_cnn) and ("CNN" not in local_models):
                    local_models["CNN"] = CNN_Model(
                        X_data.shape[1], epochs=CNN_EPOCHS, batch_size=CNN_BATCH, seed=sim_i
                    )

                for m_name, base_model in local_models.items():
                    accs, heas, infs = [], [], []

                    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_data, y)):
                        clf = clone(base_model)
                        if needs_scaler(m_name):
                            clf = make_pipeline(StandardScaler(), clf)

                        try:
                            clf.fit(X_data[train_idx], y[train_idx])
                            y_pred = clf.predict(X_data[test_idx])
                            acc, hea, inf = compute_metrics(y[test_idx], y_pred)
                            accs.append(acc)
                            heas.append(hea)
                            infs.append(inf)

                            if args.export_kmers and fold_i == 0:
                                res = extract_kmers(clf, X_data, y, args.kmer_top_n, args.k)
                                for r in res:
                                    r.update({"rate": rate, "model": m_name, "feature": f_name})
                                    kmer_saver.add(r)

                        except Exception as e:
                            if fold_i == 0:
                                print(f"    [ERR] {m_name} failed: {e}")

                    if accs:
                        res_saver.add({
                            "rate_percent": rate * 100.0,
                            "source_type": "Internal",
                            "feature_type": f_name,
                            "model": m_name,
                            "input_scale": args.input_scale,
                            "norm_method": args.norm if args.input_scale == "counts" else "precomputed_log1pcpm",
                            "accuracy": np.mean(accs),
                            "healthy_acc": np.mean(heas),
                            "infected_acc": np.mean(infs),
                            "sim_rep": sim_i
                        })

                del X_data
                gc.collect()

    res_saver.flush()
    kmer_saver.flush()


def eval_external(args, X_tr_raw, X_tr_dep, df_tr, rate_col_tr,
                  X_te_raw, X_te_dep, df_te, rate_col_te, models):

    print(f"\n>>> STARTING EXTERNAL VALIDATION | CPUS={N_JOBS}")

    tr_rates_all = df_tr[rate_col_tr].astype(float).values
    inf_rates = tr_rates_all[df_tr["label"] == "infected"]
    if len(inf_rates) == 0:
        print("[WARN] No infected samples in training metadata.")
        return

    avail_rates = sorted(np.unique(inf_rates))
    user_rates = parse_rate_list(args.ext_train_rates)

    if user_rates is None:
        train_rates = [float(max(avail_rates))]
        print(f"[INFO] Training on default max rate only: {train_rates[0]}")
    else:
        train_rates = []
        for r in user_rates:
            if np.any(np.isclose(avail_rates, r)):
                train_rates.append(float(r))
            else:
                raise ValueError(f"--ext-train-rates includes {r}, not present in training set. Available: {avail_rates}")
        print(f"[INFO] Training on user-specified rates: {train_rates}")

    features = {}
    if X_tr_raw is not None and X_te_raw is not None:
        features["Raw_full"] = (X_tr_raw, X_te_raw)
    if X_tr_dep is not None and X_te_dep is not None:
        features["Kraken_full"] = (X_tr_dep, X_te_dep)

    if not features:
        print("[WARN] No matching feature matrices found for external mode.")
        return

    te_inf_idx = np.where(df_te["label"] == "infected")[0]
    if len(te_inf_idx) == 0:
        print("[WARN] No infected samples in test metadata.")
        return
    test_rates = sorted(df_te.iloc[te_inf_idx][rate_col_te].astype(float).unique())

    out_fp = Path(args.outdir) / f"benchmark_external_long_{args.input_scale}_{args.norm}.csv"
    res_saver = BufferedSaver(out_fp, BUFFER_SIZE)

    y_test_full = (df_te["label"] == "infected").astype(int).values
    rates_test_full = df_te[rate_col_te].astype(float).values
    idx_hea_te_all = np.where(y_test_full == 0)[0]

    n_train_reps = max(1, int(args.ext_train_boots))
    n_test_reps = max(0, int(args.ext_test_boots))

    # Dataset/run-specific seed offsets so independent external jobs do not
    # reuse the exact same healthy resampling stream. This keeps results
    # reproducible within a run but different across Rust/Wilt or other datasets.
    run_seed_offset = stable_seed_from_text(
        args.metadata,
        args.test_metadata,
        args.train_matrix,
        args.test_matrix,
        args.train_ids,
        args.test_ids,
        ",".join([str(r) for r in train_rates]),
        args.input_scale,
        args.norm
    )

    for f_name, (X_tr_full, X_te_full) in features.items():
        feature_seed_offset = stable_seed_from_text(f_name)
        print(f"  - Feature: {f_name}")

        for train_rep in range(n_train_reps):
            rng_rep = np.random.RandomState(SEED0 + train_rep)

            if args.ext_train_unbalanced:
                idx_inf_pool, idx_hea_pool = get_train_pools_for_rates(df_tr, rate_col_tr, train_rates)
                if len(idx_inf_pool) == 0 or len(idx_hea_pool) == 0:
                    continue
                tr_inf = rng_rep.choice(idx_inf_pool, size=len(idx_inf_pool), replace=True)
                tr_hea = rng_rep.choice(idx_hea_pool, size=len(idx_hea_pool), replace=True)
            else:
                if len(train_rates) == 1:
                    tr_inf, tr_hea = get_balanced_indices_with_replace(
                        df_tr, rate_col_tr, train_rates[0], rng_rep, args.fixed_n, replace=True
                    )
                else:
                    tr_inf, tr_hea = get_balanced_indices_mixed_rates(
                        df_tr, rate_col_tr, train_rates, rng_rep, fixed_n=args.fixed_n, replace=True
                    )

            if len(tr_inf) == 0:
                continue

            y_train = np.array([1] * len(tr_inf) + [0] * len(tr_hea), dtype=int)
            X_stack = np.vstack([X_tr_full[tr_inf], X_tr_full[tr_hea]])
            X_train = preprocess_matrix(X_stack, args.norm, args.input_scale)

            local_models = models.copy()
            if (not args.no_slow_models) and (not args.no_cnn) and ("CNN" not in local_models):
                local_models["CNN"] = CNN_Model(
                    X_train.shape[1], epochs=CNN_EPOCHS, batch_size=CNN_BATCH, seed=SEED0 + train_rep
                )

            trained_models = {}
            for m_name, base_model in local_models.items():
                clf = clone(base_model)
                if needs_scaler(m_name):
                    clf = make_pipeline(StandardScaler(), clf)
                try:
                    clf.fit(X_train, y_train)
                    trained_models[m_name] = clf
                except Exception as e:
                    print(f"    [ERR] Training failed | rep={train_rep} | {m_name}: {e}")

            for t_rate in test_rates:
                idx_inf_te_rate = np.where((y_test_full == 1) & np.isclose(rates_test_full, t_rate))[0]
                if len(idx_inf_te_rate) == 0 or len(idx_hea_te_all) == 0:
                    continue

                n = min(len(idx_inf_te_rate), len(idx_hea_te_all))
                if args.fixed_n is not None:
                    n = min(n, int(args.fixed_n))
                if n == 0:
                    continue

                test_rep_list = [0] if n_test_reps == 0 else list(range(n_test_reps))

                for test_rep in test_rep_list:
                    rate_seed_offset = int(round(float(t_rate) * 1e9)) % 1_000_000
                    test_seed = (
                        SEED0
                        + run_seed_offset
                        + feature_seed_offset
                        + 10_000 * int(train_rep)
                        + 100 * int(test_rep)
                        + rate_seed_offset
                    ) % (2**32 - 1)
                    rng_test = np.random.RandomState(test_seed)

                    if n_test_reps == 0:
                        te_inf = rng_test.choice(idx_inf_te_rate, n, replace=False)
                        te_hea = rng_test.choice(idx_hea_te_all, n, replace=False)
                    else:
                        te_inf = rng_test.choice(idx_inf_te_rate, n, replace=True)
                        te_hea = rng_test.choice(idx_hea_te_all, n, replace=True)

                    target_indices = np.concatenate([te_inf, te_hea])
                    y_test_slice = y_test_full[target_indices]
                    X_test_slice = preprocess_matrix(X_te_full[target_indices], args.norm, args.input_scale)

                    for m_name, clf in trained_models.items():
                        try:
                            y_pred = clf.predict(X_test_slice)
                            acc, hea, inf = compute_metrics(y_test_slice, y_pred)
                            res_saver.add({
                                "rate_percent": float(t_rate) * 100.0,
                                "source_type": "External",
                                "feature_type": f_name,
                                "model": m_name,
                                "input_scale": args.input_scale,
                                "norm_method": args.norm if args.input_scale == "counts" else "precomputed_log1pcpm",
                                "accuracy": float(acc),
                                "healthy_acc": float(hea),
                                "infected_acc": float(inf),
                                "train_bootstrap_rep": int(train_rep),
                                "test_rep": int(test_rep),
                                "n_test_per_class": int(n),
                                "train_rates": ",".join([str(r) for r in train_rates]),
                                "train_rate_percent": float(max(train_rates)) * 100.0
                            })
                        except Exception:
                            pass

                    del X_test_slice
                    gc.collect()

            del X_train
            gc.collect()

    res_saver.flush()
    print(f"[DONE] External results saved to: {out_fp}")


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("\n==========================================")
    print(" K-mer ML Benchmark (Steps 1-4) | Started ")
    print("==========================================\n")

    try:
        if args.mode == "internal":
            X1, X2, df1, c1 = load_and_align_data(
                args.k, args.taxid, not args.no_memmap,
                metadata=args.metadata,
                raw_matrix=args.raw_matrix, raw_ids=args.raw_ids,
                dep_matrix=args.dep_matrix, dep_ids=args.dep_ids
            )
            models = build_models(args)
            eval_internal(args, X1, X2, df1, c1, models)

        elif args.mode == "external":
            if not args.metadata or not args.test_metadata:
                raise ValueError("External mode requires --metadata and --test-metadata.")
            if not args.train_matrix or not args.train_ids:
                raise ValueError("External mode requires --train-matrix and --train-ids.")
            if not args.test_matrix or not args.test_ids:
                raise ValueError("External mode requires --test-matrix and --test-ids.")

            X1, X2, df1, c1 = load_and_align_data(
                args.k, args.taxid, not args.no_memmap,
                metadata=args.metadata,
                raw_matrix=args.train_matrix, raw_ids=args.train_ids,
                dep_matrix=None, dep_ids=None
            )
            Y1, Y2, df2, c2 = load_and_align_data(
                args.k, args.taxid, not args.no_memmap,
                metadata=args.test_metadata,
                raw_matrix=args.test_matrix, raw_ids=args.test_ids,
                dep_matrix=None, dep_ids=None
            )

            models = build_models(args)
            eval_external(args, X1, X2, df1, c1, Y1, Y2, df2, c2, models)

        print(f"\n[DONE] All results saved to: {args.outdir}")
        print("==========================================")

    except Exception as e:
        traceback.print_exc()
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()