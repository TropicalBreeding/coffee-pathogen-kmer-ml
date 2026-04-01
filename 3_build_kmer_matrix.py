#!/usr/bin/env python3
"""
K-mer Count Matrix Builder (KMC)
================================

Description
-----------
Constructs a sample-by-k-mer count matrix from FASTQ files using the KMC
k-mer counter.  Each row of the output matrix corresponds to one sample and
each column to a distinct canonical k-mer, indexed by a base-4 encoding
(A=0, C=1, G=2, T=3).  Samples are processed in parallel via
``ProcessPoolExecutor``.

For small k (k < 10) the full 4^k feature space fits comfortably in memory
and a dense NumPy array (``.npy``) is produced.  For larger k (k >= 10) a
sparse CSR matrix (``.npz``, SciPy) is used by default to avoid excessive
memory consumption.  Both behaviours can be overridden with ``--dense`` or
``--sparse``.

Dependencies
------------
- Python >= 3.8
- NumPy, SciPy (for sparse output)
- KMC 3 (https://github.com/refresh-bio/KMC)

Inputs
------
1. ``metadata.csv`` — must contain at least ``sample_id`` and optionally
   ``fastq_path``.  If ``fastq_path`` is absent, ``--fastq-dir`` (and
   optionally ``--fastq-suffix``) must be provided to locate input files.
2. FASTQ files (raw or host-depleted).

Outputs
-------
- Dense NumPy array (``.npy``) for k < 10, or sparse CSR matrix (``.npz``)
  for k >= 10 (configurable).

Examples
--------
1. Build a dense 7-mer matrix using FASTQ paths from metadata::

     python 3_build_kmer_matrix.py \\
       --metadata metadata.csv \\
       --k 7 \\
       --out results_kmer/matrix_k7.npy \\
       --cpus 16

2. Build a sparse 11-mer matrix from host-depleted FASTQs in a directory::

     python 3_build_kmer_matrix.py \\
       --metadata metadata.csv \\
       --k 11 \\
       --fastq-dir ./results_depleted \\
       --fastq-suffix .no_coffee_13443.fastq \\
       --out results_kmer/matrix_k11.npz \\
       --cpus 32

3. Force dense output for k=11 (caution: large memory footprint)::

     python 3_build_kmer_matrix.py \\
       --metadata metadata.csv \\
       --k 11 \\
       --out results_kmer/matrix_k11_dense.npy \\
       --dense
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

DNA_MAP = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3}

def run_command(cmd):
    """Run a command; raise with stderr if it fails."""
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\nSTDERR:\n{r.stderr}")

def kmer_to_index(seq: str) -> int:
    """Convert A/C/G/T k-mer to base-4 integer index. Return -1 if non-ACGT."""
    idx = 0
    for ch in seq:
        v = DNA_MAP.get(ch)
        if v is None:
            return -1
        idx = (idx * 4) + v
    return idx

def dump_to_sparse(dump_path: Path, k: int):
    """
    Parse KMC dump text into (indices, values).
    dump format is usually: kmer<TAB>count or kmer<space>count
    """
    if (not dump_path.exists()) or dump_path.stat().st_size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    idxs, vals = [], []
    with open(dump_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # handles tab or space
            if len(parts) < 2:
                continue
            kmer, cnt = parts[0], parts[1]
            if len(kmer) != k:
                continue
            j = kmer_to_index(kmer)
            if j >= 0:
                idxs.append(j)
                vals.append(float(cnt))

    # remove dump to save space
    try:
        dump_path.unlink()
    except FileNotFoundError:
        pass

    return np.asarray(idxs, dtype=np.int32), np.asarray(vals, dtype=np.float32)

def resolve_fastq_path(row: dict, fastq_dir: Path | None, suffix: str | None):
    """
    Determine FASTQ path for a sample.
    Priority:
      1) row['fastq_path'] if present
      2) fastq_dir / (sample_id + suffix) if fastq_dir and suffix provided
      3) fastq_dir / sample_id if fastq_dir provided (row might store filename in sample_id)
    """
    if "fastq_path" in row and row["fastq_path"]:
        return Path(row["fastq_path"])

    if fastq_dir is None:
        raise ValueError("metadata.csv lacks fastq_path. Provide --fastq-dir (and usually --fastq-suffix).")

    sid = row.get("sample_id")
    if not sid:
        raise ValueError("metadata.csv must contain sample_id if fastq_path is missing.")

    if suffix is not None:
        return fastq_dir / f"{sid}{suffix}"

    # fallback: treat sample_id as filename
    return fastq_dir / sid

def process_sample(task):
    """
    Worker: FASTQ -> KMC db -> dump -> (idx, sparse_payload OR dense_row)
    """
    idx, fq_path, k, tmp_root, sparse, kmc_threads, kmc_mem_gb, kmc_ci, kmc_cs = task
    
    job_tmp = tmp_root / f"job_{idx:06d}"
    job_tmp.mkdir(parents=True, exist_ok=True)

    samp_db = job_tmp / "sample_db"
    dump_fp = job_tmp / "dump.txt"

    try:
        fq_path = Path(fq_path)
       
        run_command([
            "kmc",
            f"-k{k}",
            f"-cs{kmc_cs}",
            f"-ci{kmc_ci}",
            f"-t{kmc_threads}",
            f"-m{kmc_mem_gb}",
            str(fq_path),
            str(samp_db),
            str(job_tmp),
        ])

        # Dump k-mers
        run_command(["kmc_tools", "transform", str(samp_db), "dump", str(dump_fp)])

        if sparse:
            return idx, dump_to_sparse(dump_fp, k)
        else:
            row = np.zeros(4**k, dtype=np.float32)
            idxs, vals = dump_to_sparse(dump_fp, k)
            row[idxs] = vals
            return idx, row

    except Exception as e:
        print(f"[ERROR] idx={idx} fq={fq_path}: {e}")
        return idx, None
    finally:
        shutil.rmtree(job_tmp, ignore_errors=True)

def main():
    p = argparse.ArgumentParser(description="Build k-mer matrix using KMC")
    p.add_argument("--metadata", required=True, help="metadata.csv containing sample_id and/or fastq_path")
    p.add_argument("--k", type=int, required=True, choices=[7, 11, 15, 21, 31], help="k-mer length")
    p.add_argument("--out", required=True, help="Output file (.npz for sparse or .npy for dense)")
    p.add_argument("--cpus", type=int, default=16, help="Parallel samples to process (ProcessPool workers)")

    # FASTQ resolution options
    p.add_argument("--fastq-dir", default=None, help="Directory containing FASTQ files (if metadata lacks fastq_path)")
    p.add_argument("--fastq-suffix", default=None,
                   help="Suffix appended to sample_id to form FASTQ filename, e.g. '.no_coffee_13443.fastq'")

    # Output mode
    p.add_argument("--dense", action="store_true", help="Force dense output even for large k (not recommended for k>=11)")
    p.add_argument("--sparse", action="store_true", help="Force sparse output regardless of k (overrides default and --dense)")

    # KMC params
    p.add_argument("--kmc-threads", type=int, default=1,
                   help="Threads per KMC run (keep 1 if you parallelize many samples)")
    p.add_argument("--kmc-mem-gb", type=int, default=8, help="KMC memory limit in GB per worker")
    p.add_argument("--min-count", type=int, default=1, help="Minimum k-mer count (KMC -ci)")
    p.add_argument("--max-count", type=int, default=1000000000,
               help="Max k-mer counter value for KMC (-cs). Default prevents 255 saturation.")
    
    args = p.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata rows
    with open(args.metadata, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("ERROR: metadata.csv appears empty.")

    fastq_dir = Path(args.fastq_dir) if args.fastq_dir else None

    if args.sparse:
        sparse = True
    elif args.dense:
        sparse = False
    else:
        sparse = (args.k >= 10)

    # Safety: dense matrix gets huge very fast
    if not sparse:
        cols = 4**args.k
        est_gb = (len(rows) * cols * 4) / (1024**3)  # float32
        print(f"[INFO] Dense output requested. Estimated matrix size ~{est_gb:.2f} GB (float32).")

    print(f">>> k={args.k} | samples={len(rows)} | sparse={sparse} | out={out_file}")

    # Temp root for KMC job directories
    tmp_root = out_file.parent / f"tmp_kmc_k{args.k}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Build tasks
    tasks = []
    for i, r in enumerate(rows):
        fq = resolve_fastq_path(r, fastq_dir, args.fastq_suffix)
        
        if not fq.exists():
            raise SystemExit(f"[FATAL] No input file: {fq}")
        
        tasks.append((
            i,
            str(fq),
            args.k,
            tmp_root,
            sparse,
            args.kmc_threads,
            args.kmc_mem_gb,
            args.min_count,
            args.max_count,
        ))

    n = len(tasks)

    try:
        if sparse:
            import scipy.sparse as sp
            indptr, indices, data = [0], [], []

            with ProcessPoolExecutor(max_workers=args.cpus) as ex:
                for j, (idx, payload) in enumerate(ex.map(process_sample, tasks)):
                    if payload is not None:
                        idxs, vals = payload
                        indices.extend(idxs.tolist())
                        data.extend(vals.tolist())
                    indptr.append(len(indices))
                    if (j % 25 == 0) or (j + 1 == n):
                        print(f"Processing: {j+1}/{n}", end="\r")

            X = sp.csr_matrix((data, indices, indptr), shape=(n, 4**args.k), dtype=np.float32)
            sp.save_npz(out_file, X)
            print(f"\n[SUCCESS] Saved sparse matrix to {out_file}")

        else:
            X = np.zeros((n, 4**args.k), dtype=np.float32)

            with ProcessPoolExecutor(max_workers=args.cpus) as ex:
                for j, (idx, row) in enumerate(ex.map(process_sample, tasks)):
                    if row is not None:
                        X[idx, :] = row
                    if (j % 25 == 0) or (j + 1 == n):
                        print(f"Processing: {j+1}/{n}", end="\r")

            np.save(out_file, X)
            print(f"\n[SUCCESS] Saved dense matrix to {out_file}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
