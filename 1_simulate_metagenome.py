#!/usr/bin/env python3
"""
In-Silico Metagenome Simulator with Exact Composition Control
=============================================================

Description
-----------
Generates synthetic Illumina-like sequencing reads for healthy and pathogen-
infected plant samples with precisely controlled genomic composition.  Each
sample contains reads drawn from four compartments — nuclear genome,
chloroplast genome, mitochondrial genome, and environmental background — in
user-defined proportions that sum to 1.0.  For infected samples an additional
pathogen compartment is introduced at a specified infection rate, and the host
compartments are scaled proportionally to fill the remainder.

Read simulation is performed by InSilicoSeq (ISS) with the NovaSeq error
model (150 bp reads).  A nuclear-reserve strategy over-generates nuclear
reads so that any shortfall from rounding or ISS under-production can be
compensated by drawing from the unused tail of the nuclear FASTQ, thereby
guaranteeing every sample reaches exactly TOTAL_READS without read
duplication.  Final per-sample FASTQs are shuffled for randomness.

Dependencies
------------
- Python >= 3.8
- InSilicoSeq (``pip install insilicoseq``)

Inputs
------
Reference FASTA files for host nuclear, chloroplast, mitochondrial, and
pathogen genomes, plus a background metagenome FASTA.  Paths are configured
in the CONFIGURATION section below.

Outputs
-------
- One FASTQ file per sample (``<sample_id>.fastq``)
- ``metadata.csv`` recording sample composition (sample_id, label, host,
  pathogen, target_rate, read counts per compartment, fastq_path)

Examples
--------
1. Generate 100 healthy and 100 infected replicates across all default
   infection rates (0.005 %–20 %)::

     python 1_simulate_metagenome.py \\
       --outdir ./sim_output \\
       --cpus 32 \\
       --seed 42

2. Overwrite a previous run with a different seed::

     python 1_simulate_metagenome.py \\
       --outdir ./sim_output \\
       --cpus 16 \\
       --seed 123 \\
       --overwrite

3. Quick test with the default NovaSeq model on 4 cores::

     python 1_simulate_metagenome.py \\
       --outdir ./test_run \\
       --cpus 4

Runtime
-------
Approximately 4 minutes per replicate; ~6 h 40 min for 100 replicates on
32 cores.
"""

import argparse
import csv
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# 1. CONFIGURATION (UPDATE THESE PATHS)
# ==============================================================================

TOTAL_READS = 2_000_000

# Healthy fractions must sum to 1.0
HEALTHY_FRACTIONS = {
    "nuclear": 0.91,
    "chloroplast": 0.06,
    "mitochondria": 0.02,
    "background": 0.01,
}

# Background source FASTQ (real metagenome)
BG_SOURCE_FASTA = Path("reference_DNA/SRR36352696.fasta")

# Host reference genomes (NUCLEAR)
HOST_NUCLEAR_GENOMES = {
    "ET_39":  Path("reference_DNA/GCF_036785885.1_Coffea_Arabica_ET-39_HiFi_genomic.fna"),
    "Typica": Path("reference_DNA/GCA_049114775.1_ASM4911477v1_Typica_genomic.fna"),
}

# Host organelle genomes
HOST_CHLORO_GENOMES = {
    "ET_39":  Path("reference_DNA/arabica_chl.fasta"),
    "Typica": Path("reference_DNA/arabica_chl.fasta"),
}
HOST_MITO_GENOMES = {
    "ET_39":  Path("reference_DNA/arabica_mito.fasta"),
    "Typica": Path("reference_DNA/arabica_mito.fasta"),
}

# Pathogen genomes
PATHOGEN_GENOMES = {
    "Rust": Path("reference_DNA/GCA_030280995.1_Hv_R1_Cat_CENPAT_Rust.fna"),
    "Wilt": Path("reference_DNA/GCA_013183765.1_ASM1318376v1_Fusarium.fna"),
}

# Infection Rates (pathogen fraction of TOTAL_READS)
TARGET_RATES = [0.00005, 0.0001, 0.001, 0.01, 0.05, 0.10, 0.20]

HEALTHY_REPLICATES = 100
INFECTION_REPLICATES = 100

DEFAULT_MODEL = "novaseq"

# Nuclear reserve: generate extra nuclear reads up front, so top-up can come from tail without duplicates.
# This should exceed any plausible deficit coming from organelle under-production + rounding.
# 5% of total is usually plenty; increase if you still see "Cannot top-up enough nuclear reads".
NUCLEAR_RESERVE_READS = int(TOTAL_READS * 0.05)  # 100,000

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def assert_fractions_sum_to_one(fracs: Dict[str, float], eps: float = 1e-9) -> None:
    s = sum(fracs.values())
    if abs(s - 1.0) > eps:
        raise ValueError(f"Fractions must sum to 1.0, but sum={s}")

def allocate_exact_counts(total: int, fractions: Dict[str, float]) -> Dict[str, int]:
    """
    Convert fractions to exact integer counts that sum to total.
    Uses largest remainder method.
    """
    keys = list(fractions.keys())
    raw = {k: total * fractions[k] for k in keys}
    base = {k: int(raw[k]) for k in keys}
    remainder = total - sum(base.values())

    frac_parts = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
    for i in range(remainder):
        base[frac_parts[i % len(frac_parts)]] += 1

    if sum(base.values()) != total:
        raise RuntimeError("Exact allocation failed to sum to total.")
    return base

def scale_host_fractions_for_infection(path_rate: float, healthy_fracs: Dict[str, float]) -> Dict[str, float]:
    """
    Pathogen takes path_rate. Remaining (1 - path_rate) is distributed among
    nuclear/chloro/mito/background proportionally to healthy_fracs.
    """
    remain = 1.0 - path_rate
    if remain < 0:
        raise ValueError(f"path_rate must be <= 1.0, got {path_rate}")

    host_fracs = {k: healthy_fracs[k] * remain for k in healthy_fracs}
    host_fracs["pathogen"] = path_rate

    # Minor floating errors can happen; normalize defensively
    s = sum(host_fracs.values())
    if abs(s - 1.0) > 1e-9:
        # normalize
        for k in host_fracs:
            host_fracs[k] /= s
    return host_fracs

def find_iss_r1_fastq(output_prefix: Path) -> Path:
    """
    Non-gzip only.
    Prefer paired-end R1: <prefix>_R1.fastq
    Else single-end:      <prefix>.fastq
    Else fallback glob:   <prefix>*.fastq
    """
    p = Path(f"{output_prefix}_R1.fastq")
    if p.exists():
        return p

    p = Path(f"{output_prefix}.fastq")
    if p.exists():
        return p

    globbed = list(output_prefix.parent.glob(output_prefix.name + "*.fastq"))
    if globbed:
        globbed.sort()
        return globbed[0]

    raise FileNotFoundError(f"ISS output FASTQ not found for prefix={output_prefix}")

def run_iss(genome: Path, output_prefix: Path, n_reads_target: int, model: str, cpus: int, seed: int,
            extra_reserve: int = 0) -> Path:
    """
    Run ISS and return the R1 fastq path (non-gzip).
    Requests 20% extra plus extra_reserve to buffer under-production.
    """
    required = n_reads_target + extra_reserve
    if required <= 0:
        raise ValueError("n_reads_target must be > 0")

    n_request = max(1, int((n_reads_target + extra_reserve) * 1.2))
    
    # IMPORTANT: avoid ISS temp chunk concatenate bug at low read counts
    if n_request < 10000:
        cpus_used = 1
    elif n_request < 50000:
        cpus_used = min(cpus, 4)
    else:
        cpus_used = cpus
    
    cmd = [
        "iss", "generate",
        "--genomes", str(genome),
        "--model", model,
        "--n_reads", str(n_request * 2),  # ISS counts R1+R2; we use only R1
        "--output", str(output_prefix),
        "--cpus", str(cpus_used),
        "--seed", str(seed),
        "--quiet"
    ]
    subprocess.run(cmd, check=True)

    return find_iss_r1_fastq(output_prefix)

def sample_background_reads(source_file: Path, n_needed: int, seed: int) -> List[str]:
    """
    Samples n_needed reads from background file (FASTQ) using sampling with replacement.
    Returns list of FASTQ 4-line strings.
    """
    if n_needed <= 0:
        return []

    if not source_file.exists():
        # Dummy background if missing
        return [f"@BG_Sim_{i}\n{'N'*150}\n+\n{'#'*150}\n" for i in range(n_needed)]

    all_reads: List[str] = []
    with open(source_file, "r") as f:
        while True:
            h = f.readline()
            if not h:
                break
            seq = f.readline()
            plus = f.readline()
            qual = f.readline()
            if not qual:
                break
            all_reads.append(f"{h}{seq}{plus}{qual}")

    if not all_reads:
        raise RuntimeError(f"Background file has no reads: {source_file}")

    rng = random.Random(seed)
    return rng.choices(all_reads, k=n_needed)

def read_fastq_records_with_offset(fq: Path, start_record: int, n: int) -> List[str]:
    """
    Read n FASTQ records starting from record index start_record (0-based).
    Non-gzip FASTQ only.
    """
    out: List[str] = []
    if n <= 0:
        return out
    if start_record < 0:
        raise ValueError("start_record must be >= 0")

    skip_lines = start_record * 4

    with open(fq, "r") as f:
        # skip to offset
        for _ in range(skip_lines):
            if not f.readline():
                return out  # EOF before offset

        # read n records
        for _ in range(n):
            rec = "".join([f.readline() for _ in range(4)])
            if not rec or not rec.strip():
                break
            out.append(rec)

    return out

def merge_and_shuffle_multi(
    sources: List[Tuple[Path, int]],
    nuclear_fastq: Path,
    n_nuc_requested: int,
    output_file: Path,
    seed: int,
    require_exact_total: int
) -> None:
    """
    Merge reads from multiple FASTQ sources into a single output file.

    If the merged total falls short of require_exact_total, additional nuclear
    reads are drawn from the unused tail of the nuclear FASTQ to avoid
    duplicating reads already included.  If the total exceeds the target, it
    is trimmed.  The final read set is shuffled before writing.
    """
    final_reads: List[str] = []

    for fq, n in sources:
        if n <= 0:
            continue
        if not fq.exists():
            raise FileNotFoundError(f"Missing FASTQ source: {fq}")
        final_reads.extend(read_fastq_records_with_offset(fq, start_record=0, n=n))

    # Top-up from nuclear tail (no duplicates)
    if len(final_reads) < require_exact_total:
        deficit = require_exact_total - len(final_reads)

        # We already attempted to take n_nuc_requested nuclear reads from start.
        # Fill deficit from the "tail": start at offset = n_nuc_requested
        extra = read_fastq_records_with_offset(nuclear_fastq, start_record=n_nuc_requested, n=deficit)

        if len(extra) < deficit:
            raise RuntimeError(
                f"Cannot top-up enough nuclear reads without duplicates. "
                f"Needed {deficit}, got {len(extra)}. "
                f"Increase NUCLEAR_RESERVE_READS or increase ISS over-sampling."
            )

        final_reads.extend(extra)

    # Trim if over target
    if len(final_reads) > require_exact_total:
        final_reads = final_reads[:require_exact_total]

    # Shuffle and write
    rng = random.Random(seed)
    rng.shuffle(final_reads)

    with open(output_file, "w") as f:
        f.writelines(final_reads)

# ==============================================================================
# 3. MAIN
# ==============================================================================

def main():
    assert_fractions_sum_to_one(HEALTHY_FRACTIONS)
    if not BG_SOURCE_FASTA.exists():
        raise FileNotFoundError(f"Background source FASTA not found: {BG_SOURCE_FASTA}")

    parser = argparse.ArgumentParser(description="Generate Exact Composition Metagenomes (Organelles + NovaSeq)")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--cpus", type=int, default=16, help="CPUs for ISS")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="ISS error model (default: novaseq)")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.outdir / "iss_temp"
    temp_dir.mkdir(exist_ok=True)

    metadata_file = args.outdir / "metadata.csv"
    if not metadata_file.exists() or args.overwrite:
        with open(metadata_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "sample_id", "label", "host", "pathogen", "target_rate",
                "n_nuclear", "n_chloroplast", "n_mitochondria", "n_background", "n_pathogen",
                "fastq_path"
            ])

    print(">>> Starting Simulation...")
    print(f"    Total Reads: {TOTAL_READS}")
    print(f"    Healthy fractions: {HEALTHY_FRACTIONS}")
    print(f"    Nuclear reserve reads: {NUCLEAR_RESERVE_READS}")
    print(f"    Model: {args.model}")

    global_counter = 0

    # ---------------------------------------------------------
    # Loop 1: Healthy
    # ---------------------------------------------------------
    for host_name in HOST_NUCLEAR_GENOMES:
        nuc_genome = HOST_NUCLEAR_GENOMES[host_name]
        chl_genome = HOST_CHLORO_GENOMES[host_name]
        mito_genome = HOST_MITO_GENOMES[host_name]

        print(f"\n--- Processing Healthy: {host_name} ---")

        counts = allocate_exact_counts(TOTAL_READS, HEALTHY_FRACTIONS)
        n_nuc = counts["nuclear"]
        n_chl = counts["chloroplast"]
        n_mito = counts["mitochondria"]
        n_bg = counts["background"]

        for rep in range(1, HEALTHY_REPLICATES + 1):
            sample_id = f"healthy_{host_name}_rep{rep:03d}"
            out_fastq = args.outdir / f"{sample_id}.fastq"

            if out_fastq.exists() and not args.overwrite:
                continue

            seed = args.seed + global_counter
            global_counter += 1

            nuc_prefix = temp_dir / f"{sample_id}_nuclear"
            chl_prefix = temp_dir / f"{sample_id}_chloroplast"
            mito_prefix = temp_dir / f"{sample_id}_mitochondria"
            bg_prefix = temp_dir / f"{sample_id}_background"

            # Nuclear: generate with reserve to support top-up without duplicates
            nuc_fq = run_iss(nuc_genome, nuc_prefix, n_nuc, args.model, args.cpus, seed + 11,
                             extra_reserve=NUCLEAR_RESERVE_READS)

            chl_fq = run_iss(chl_genome, chl_prefix, n_chl, args.model, args.cpus, seed + 22)
            mito_fq = run_iss(mito_genome, mito_prefix, n_mito, args.model, args.cpus, seed + 33)
            bg_fq = run_iss(BG_SOURCE_FASTA, bg_prefix, n_bg, args.model, args.cpus, seed + 44)

            merge_and_shuffle_multi(
                sources=[(nuc_fq, n_nuc), (chl_fq, n_chl), (mito_fq, n_mito), (bg_fq, n_bg)],
                nuclear_fastq=nuc_fq,
                n_nuc_requested=n_nuc,
                output_file=out_fastq,
                seed=seed,
                require_exact_total=TOTAL_READS
            )

            with open(metadata_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    sample_id, "healthy", host_name, "None", 0.0,
                    n_nuc, n_chl, n_mito, n_bg, 0,
                    str(out_fastq)
                ])

            print(f"Generated {sample_id}", end="\r")

    # ---------------------------------------------------------
    # Loop 2: Infected
    # ---------------------------------------------------------
    for host_name in HOST_NUCLEAR_GENOMES:
        nuc_genome = HOST_NUCLEAR_GENOMES[host_name]
        chl_genome = HOST_CHLORO_GENOMES[host_name]
        mito_genome = HOST_MITO_GENOMES[host_name]

        for path_name, path_genome in PATHOGEN_GENOMES.items():
            print(f"\n--- Processing Infected: {host_name} + {path_name} ---")

            for rate in TARGET_RATES:
                fracs = scale_host_fractions_for_infection(rate, HEALTHY_FRACTIONS)
                counts = allocate_exact_counts(TOTAL_READS, fracs)

                n_nuc = counts.get("nuclear", 0)
                n_chl = counts.get("chloroplast", 0)
                n_mito = counts.get("mitochondria", 0)
                n_bg = counts.get("background", 0)
                n_path = counts.get("pathogen", 0)

                for rep in range(1, INFECTION_REPLICATES + 1):
                    sample_id = f"inf_{host_name}_{path_name}_r{rate}_rep{rep:03d}"
                    out_fastq = args.outdir / f"{sample_id}.fastq"

                    if out_fastq.exists() and not args.overwrite:
                        continue

                    seed = args.seed + global_counter
                    global_counter += 1

                    nuc_prefix = temp_dir / f"{sample_id}_nuclear"
                    chl_prefix = temp_dir / f"{sample_id}_chloroplast"
                    mito_prefix = temp_dir / f"{sample_id}_mitochondria"
                    path_prefix = temp_dir / f"{sample_id}_pathogen"
                    bg_prefix = temp_dir / f"{sample_id}_background"

                    # Nuclear with reserve (still helpful if organelles/pathogen under-produce)
                    nuc_fq = run_iss(nuc_genome, nuc_prefix, n_nuc, args.model, args.cpus, seed + 11,
                                     extra_reserve=NUCLEAR_RESERVE_READS)

                    chl_fq = run_iss(chl_genome, chl_prefix, n_chl, args.model, args.cpus, seed + 22) if n_chl > 0 else None
                    mito_fq = run_iss(mito_genome, mito_prefix, n_mito, args.model, args.cpus, seed + 33) if n_mito > 0 else None
                    path_fq = run_iss(path_genome, path_prefix, n_path, args.model, args.cpus, seed + 55) if n_path > 0 else None
                    bg_fq = run_iss(BG_SOURCE_FASTA, bg_prefix, n_bg, args.model, args.cpus, seed + 44) if n_bg > 0 else None

                    sources: List[Tuple[Path, int]] = [(nuc_fq, n_nuc)]
                    if chl_fq is not None:
                        sources.append((chl_fq, n_chl))
                    if mito_fq is not None:
                        sources.append((mito_fq, n_mito))
                    if path_fq is not None:
                        sources.append((path_fq, n_path))
                    if bg_fq is not None:
                        sources.append((bg_fq, n_bg))

                    merge_and_shuffle_multi(
                        sources=sources,
                        nuclear_fastq=nuc_fq,
                        n_nuc_requested=n_nuc,
                        output_file=out_fastq,
                        seed=seed,
                        require_exact_total=TOTAL_READS
                    )

                    with open(metadata_file, "a", newline="") as f:
                        csv.writer(f).writerow([
                            sample_id, "infected", host_name, path_name, rate,
                            n_nuc, n_chl, n_mito, n_bg, n_path,
                            str(out_fastq)
                        ])

                print(f"Rate {rate*100:.5f}% done...", end="\r")

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\n[Done] All simulations complete. Metadata saved to {metadata_file}")

if __name__ == "__main__":
    main()
