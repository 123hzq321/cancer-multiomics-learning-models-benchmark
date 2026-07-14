# Cancer Multi-Omics Learning Models Benchmark

This repository accompanies the manuscript:

**Comparative Evaluation of Multi-Omics Learning Models for Cancer Prediction with Breast Cancer External Validation and a Multi-Cancer Internal Benchmark**

The repository contains analysis scripts, compact processed result tables, figures, reports, and the LaTeX manuscript source for a public cancer multi-omics prediction benchmark. The study evaluates Liquid/CfC-style neural fusion, a reduced small-Liquid/CfC model, classical machine-learning baselines, and neural baselines for breast cancer subtype prediction and five TCGA cancer-internal tasks.

## Main Analyses

1. Breast cancer external validation using TCGA-BRCA and METABRIC as a joined training pool, with CPTAC 2020, SMC 2018, and GSE96058/SCAN-B as external validation cohorts.
2. Large-scale mRNA-marker baseline comparison, including elastic-net logistic regression, calibrated linear SVM, ridge classifier, LDA, Gaussian naive Bayes, k-nearest neighbors, random forest, ExtraTrees, histogram gradient boosting, and MLP.
3. Five-task TCGA multi-cancer internal benchmark covering UCEC molecular subtype, COADREAD molecular subtype, HNSC HPV status, KIRC grade, and PRAD pathologic T stage.
4. Ten-fold cross-validation, fold-level effect-size diagnostics, and noise robustness analyses for the core-aligned multi-cancer benchmark.

## Repository Layout

- `scripts/`: data download, preprocessing, training, evaluation, and audit scripts.
- `processed_results/`: compact CSV outputs used to support manuscript tables and claims.
- `figures/`: manuscript figures generated from the analyses.
- `reports/`: markdown reports produced during analysis.
- `manuscript/`: LaTeX manuscript source and compiled PDF.

Large public molecular feature matrices are not duplicated in this repository. They can be regenerated from the public sources listed in `DATA_SOURCES.md` using the scripts in `scripts/`.

## Core Scripts

- `crawl_tcga_brca_multiomics_cbioportal.py`
- `crawl_metabric_external_validation.py`
- `crawl_tcga_pancancer_multiomics_cbioportal.py`
- `train_joined_metabric_small_liquid_external_validation.py`
- `train_large_scale_mrna_baselines.py`
- `train_multicancer_internal_benchmark.py`
- `reviewer_cv_noise_significance_analysis.py`
- `audit_sample_alignment_for_review.py`

## Environment

The analyses were run with Python 3.10 on Windows. Install the main dependencies with:

```bash
pip install -r requirements.txt
```

GPU acceleration is optional for the PyTorch neural models. Classical baselines run on CPU.

## Reproducibility Notes

The scripts use public cohorts from cBioPortal, GEO, and associated public processed resources. Because public APIs and mirrors can change, exact regenerated matrices may differ slightly if source annotations are updated. The compact results included here record the outputs used for the manuscript version deposited with this repository.

## License

Code is released under the MIT License. Public dataset terms remain governed by their original data providers.

