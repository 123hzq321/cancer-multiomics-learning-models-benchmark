# Reproducing the Analyses

This file gives a high-level reproduction path for the manuscript analyses.

## 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, replace the activation command with the corresponding shell command.

## 2. Rebuild Data Tables

The scripts below download or reconstruct the public input tables used by the analysis.

```bash
python scripts/crawl_tcga_brca_multiomics_cbioportal.py
python scripts/crawl_metabric_external_validation.py
python scripts/crawl_tcga_pancancer_multiomics_cbioportal.py
```

SCAN-B/GSE96058 uses public processed expression resources. If the remote mirror changes, download the GSE96058 expression and sample-description files manually from GEO-linked public resources and place them in the expected data directory documented in the corresponding training script.

## 3. Run Main Model Analyses

```bash
python scripts/train_joined_metabric_small_liquid_external_validation.py
python scripts/train_large_scale_mrna_baselines.py
python scripts/train_multicancer_internal_benchmark.py
```

## 4. Run Audit and Sensitivity Analyses

```bash
python scripts/audit_sample_alignment_for_review.py
python scripts/reviewer_cv_noise_significance_analysis.py
```

These commands produce summary CSV files, model result tables, prediction outputs, figures, and markdown reports.

## 5. Compare with Deposited Results

The repository includes compact result outputs under:

- `processed_results/joined_metabric_external_validation/`
- `processed_results/large_scale_baselines/`
- `processed_results/multicancer_internal_benchmark_core_aligned/`
- `processed_results/tenfold_cv_noise_significance/`
- `processed_results/alignment_audit/`

The manuscript figures are stored in `figures/` and duplicated under `manuscript/figures/` for LaTeX compilation.

