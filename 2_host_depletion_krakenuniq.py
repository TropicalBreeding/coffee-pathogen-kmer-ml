#!/usr/bin/env python3
"""
Host DNA Depletion via KrakenUniq
=================================

Description
-----------
Removes host-derived reads (Coffea arabica, NCBI TaxID 13443) from FASTQ
sequencing data.  The pipeline operates in two steps:

  1. **Classification** — KrakenUniq assigns a taxonomic identifier to every
     read in the input FASTQ against a pre-built database.
  2. **Extraction** — KrakenTools ``extract_kraken_reads.py`` excludes all
     reads classified under Coffea arabica (including child taxa), producing
     a host-depleted FASTQ suitable for downstream pathogen analysis.

The script processes one sample per invocation, identified by a 0-based row
index into the metadata CSV.  This design supports Slurm array jobs where
each task handles a single sample in parallel.

Dependencies
------------
- Python >= 3.8
- KrakenUniq (https://github.com/fbreitwieser/krakenuniq)
- KrakenTools (https://github.com/jenniferlu717/KrakenTools)

Inputs
------
1. ``metadata.csv`` with columns: ``sample_id``, ``fastq_path``
2. KrakenUniq database directory (must contain ``tax/`` and ``library/``)
3. Path to ``KrakenTools/extract_kraken_reads.py``

Outputs
-------
For each sample:

- ``<sample_id>.report``            — KrakenUniq classification summary
- ``<sample_id>.kraken``            — per-read classification output
- ``<sample_id>.no_coffee_13443.fastq`` — FASTQ with Coffea reads removed

Examples
--------
1. Deplete host reads from a single sample (row 0 of the metadata)::

     python 2_host_depletion_krakenuniq.py \\
       --metadata metadata.csv \\
       --index 0 \\
       --db /path/to/krakenuniq_db \\
       --extract /path/to/KrakenTools/extract_kraken_reads.py \\
       --out_report results_reports \\
       --out_fastq results_depleted \\
       --threads 32

2. Submit as a Slurm array job (one task per sample)::

     sbatch --array=0-199 deplete.sh
     # Inside deplete.sh:
     python 2_host_depletion_krakenuniq.py \\
       --metadata metadata.csv \\
       --index $SLURM_ARRAY_TASK_ID \\
       --db /path/to/krakenuniq_db \\
       --extract /path/to/KrakenTools/extract_kraken_reads.py \\
       --out_report results_reports \\
       --out_fastq results_depleted \\
       --threads 16
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path

# NCBI TaxID for Coffea arabica
COFFEA_TAXID = "13443"

def main():
    # -------------------------------------------------------------------------
    # Argument Parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser( description="Remove Coffea (TaxID 13443) reads from FASTQ using KrakenUniq."    )
    parser.add_argument("--metadata", required=True, help="CSV file containing sample_id and fastq_path columns."    )
    parser.add_argument("--index", required=True, type=int, help="0-based row index in metadata.csv (use SLURM_ARRAY_TASK_ID)." )
    parser.add_argument("--db", required=True, help="Path to KrakenUniq database directory." )
    parser.add_argument("--extract", required=True, help="Path to KrakenTools extract_kraken_reads.py script."  )
    parser.add_argument("--out_report", required=True, help="Directory for kraken report and classification outputs."  )
    parser.add_argument("--out_fastq", required=True, help="Directory for final depleted FASTQ files."  )
    parser.add_argument("--threads", default="16", help="Number of CPU threads for krakenuniq."  )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    Path(args.out_report).mkdir(parents=True, exist_ok=True)
    Path(args.out_fastq).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 0: Read metadata and get the specific sample
    # -------------------------------------------------------------------------
    with open(args.metadata, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == args.index:
                sample_id = row["sample_id"]
                fastq = row["fastq_path"]
                break
        else:
            raise SystemExit(
                f"Index {args.index} out of range in {args.metadata}"
            )

    print("======================================================")
    print(f"Sample ID : {sample_id}")
    print(f"FASTQ     : {fastq}")
    print(f"Threads   : {args.threads}")
    print("======================================================")

    # Define output file paths
    report_file = os.path.join(args.out_report, f"{sample_id}.report")
    kraken_file = os.path.join(args.out_report, f"{sample_id}.kraken")
    out_fastq = os.path.join(
        args.out_fastq,
        f"{sample_id}.no_coffee_{COFFEA_TAXID}.fastq"
    )

    # -------------------------------------------------------------------------
    # Step 1: Run KrakenUniq classification
    # -------------------------------------------------------------------------
    # This assigns a taxonomic ID to each read in the FASTQ file.
    # If output already exists and is non-empty, skip.
    if not (os.path.exists(report_file) and os.path.getsize(report_file) > 0 and
            os.path.exists(kraken_file) and os.path.getsize(kraken_file) > 0):

        print("[Step 1] Running krakenuniq classification...")

        subprocess.run([
            "krakenuniq",
            "--db", args.db,
            "--threads", str(args.threads),
            "--report-file", report_file,
            "--output", kraken_file,
            fastq
        ], check=True)

    else:
        print("[Skip] Classification already completed.")

    # -------------------------------------------------------------------------
    # Step 2: Remove Coffea reads using KrakenTools
    # -------------------------------------------------------------------------
    # We exclude TaxID 13443 and all its children.
    # Result is a FASTQ file without coffee reads.
    if not (os.path.exists(out_fastq) and os.path.getsize(out_fastq) > 0):

        print("[Step 2] Removing Coffea (TaxID 13443) reads...")

        subprocess.run([
            "python", args.extract,
            "-k", kraken_file,
            "-s", fastq,
            "-o", out_fastq,
            "-t", COFFEA_TAXID,
            "--report", report_file,
            "--include-children",
            "--exclude",
            "--fastq-output"
        ], check=True)

    else:
        print("[Skip] Coffee depletion already completed.")

    print("======================================================")
    print(f"[DONE] Output written to: {out_fastq}")
    print("======================================================")


if __name__ == "__main__":
    main()
