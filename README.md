# Coffee pathogen detection from simulated metagenomes using k-mer machine learning

This repository contains the analysis pipeline used to simulate coffee metagenome sequencing data, remove host reads, build k-mer count matrices, benchmark machine-learning classifiers, and run a two-stage framework for pathogen detection and infection-rate prediction.

The workflow was developed for simulated *Coffea arabica* datasets representing healthy tissue and tissue infected with *Hemileia vastatrix* (coffee leaf rust) or *Fusarium xylarioides* (coffee wilt), across a range of infection rates.

## Repository contents

- `1_simulate_metagenome.py`  
  Simulates Illumina-like metagenomic reads with controlled proportions of host nuclear, chloroplast, mitochondrial, environmental background, and pathogen reads. Outputs per-sample FASTQ files and a `metadata.csv` file. The current version contains dataset-specific reference paths and should be modified before reuse.
  
- `2_host_depletion_krakenuniq.py`  
  Removes host-derived reads from FASTQ files using KrakenUniq classification and KrakenTools extraction.

- `3_build_kmer_matrix.py`  
  Builds sample-by-k-mer count matrices from FASTQ files using KMC. Produces dense `.npy` matrices for smaller k and sparse `.npz` matrices for larger k by default.

- `4_ML_benchmark.py`  
  Benchmarks multiple machine-learning classifiers for same-rate, raw-vs-depleted, cross-rate, and external-transfer analyses.

- `5_two_stage_detection.py`  
  Implements a two-stage framework for pathogen detection and infection-rate prediction using elastic-net logistic regression for detection and ridge regression for quantitative rate prediction.

## Workflow overview

1. **Simulate metagenomes**  
   Generate healthy and infected sequencing datasets with known infection rates.

2. **Host depletion**  
   Remove host-derived reads using KrakenUniq.

3. **Build k-mer matrices**  
   Count canonical k-mers from raw or depleted FASTQ files.

4. **Benchmark classifiers**  
   Evaluate multiple models across infection rates and validation settings.

5. **Run the two-stage framework**  
   Detect infection and estimate infection rate from k-mer profiles.

### External software requirements
- InSilicoSeq
- KrakenUniq
- KrakenTools
- KMC 3

### Core software
- Python 3.10
- NumPy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- SciPy 1.15.2
- TensorFlow 2.19.1 

### Additional tools by step
- **Step 1 simulation:** InSilicoSeq
- **Step 2 host depletion:** KrakenUniq, KrakenTools
- **Step 3 k-mer counting:** KMC 3
- **Step 4 benchmarking:** TensorFlow (required only if CNN benchmarking is used)
- **Step 5 two-stage framework:** no TensorFlow required

## Input data expected by the scripts

### Metadata
Most downstream scripts expect a metadata table with at least:
- `sample_id`
- `label`
- an infection-rate column named one of:
  - `target_rate`
  - `targetrate`
  - `rate`
  - `target_rate_fraction`

For host depletion, metadata must also include:
- `fastq_path`

### Matrix/sample ID files
Benchmarking and two-stage scripts expect:
- a k-mer matrix (`.npy`; dense matrix expected by current scripts)
- a matching ordered sample ID file (`sample_ids.txt`)

## Typical usage

### 1) Simulate reads
```bash
python 1_simulate_metagenome.py \
  --outdir sim_output \
  --cpus 32 \
  --seed 42
```

### 2) Remove host reads
```bash
python 2_host_depletion_krakenuniq.py \
  --metadata sim_output/metadata.csv \
  --index 0 \
  --db /path/to/krakenuniq_db \
  --extract /path/to/KrakenTools/extract_kraken_reads.py \
  --out_report kraken_reports \
  --out_fastq depleted_fastq \
  --threads 16
```

### 3) Build a 7-mer matrix
```bash
python 3_build_kmer_matrix.py \
  --metadata sim_output/metadata.csv \
  --k 7 \
  --fastq-dir depleted_fastq \
  --fastq-suffix .no_coffee_13443.fastq \
  --out results_kmer/matrix_k7.npy \
  --cpus 16
```

### 4) Benchmark models
```bash
python 4_ML_benchmark.py \
  --outdir results_step1 \
  --k 7 \
  --mode internal \
  --metadata sim_output/metadata.csv \
  --raw-matrix results_kmer/matrix_k7.npy \
  --raw-ids results_kmer/sample_ids.txt \
  --input-scale counts \
  --norm log1p_cpm
```

### 5) Run the two-stage detection/rate framework
```bash
python 5_two_stage_detection.py \
  --mode internal \
  --metadata sim_output/metadata.csv \
  --matrix results_kmer/matrix_k7.npy \
  --ids results_kmer/sample_ids.txt \
  --outdir results_step5 \
  --input-scale counts \
  --norm log1p_cpm
```

## Main outputs

### Step 1
- Per-sample FASTQ files
- `metadata.csv`

### Step 2
- KrakenUniq reports
- Kraken classification files
- host-depleted FASTQ files

### Step 3
- k-mer count matrices (`.npy` or `.npz` depending on settings)

### Step 4
- internal benchmarking CSV files
- external benchmarking CSV files
- optional k-mer importance tables

### Step 5
- detection metrics CSV files
- per-sample infection-rate prediction CSV files
- infection-rate MAE summary CSV files

## Notes on reproducibility

- The scripts expose random seeds for simulation and repeated evaluation.
- The two-stage framework uses outer cross-validation and an inner validation split for threshold calibration.
- Infection-rate weighting is used in the detector to emphasize lower-rate infected samples.
- The current scripts contain dataset-specific configuration values and file paths in some places, especially in `1_simulate_metagenome.py`. These should be edited or parameterized before reuse on other systems.

## Adapting the repository for new datasets

To reuse this code on a different pathosystem, you will typically need to change:
- reference genome FASTA paths in `1_simulate_metagenome.py`
- host taxon/database settings for KrakenUniq depletion
- metadata and FASTQ locations
- k-mer length and matrix output settings
- training/test metadata for external validation

## Suggested citation

If you use this repository, please cite the associated manuscript and the software dependencies used in the workflow, including InSilicoSeq, KrakenUniq, KrakenTools, KMC, scikit-learn, and TensorFlow when applicable.

## Contact

For questions about the code or manuscript, please contact the repository author(s).
