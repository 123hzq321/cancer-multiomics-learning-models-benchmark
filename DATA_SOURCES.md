# Data Sources

All datasets used in this study are publicly available.

## Breast Cancer Training and External Validation

- TCGA-BRCA: accessed through cBioPortal.
- METABRIC: accessed through cBioPortal.
- CPTAC 2020 breast cancer cohort: accessed through cBioPortal where molecular and clinical data were available.
- SMC 2018 breast cancer cohort: accessed through cBioPortal where available.
- GSE96058/SCAN-B: accessed through the Gene Expression Omnibus and associated public processed expression resources.

## Multi-Cancer Internal Benchmark

TCGA PanCancer Atlas cohorts were accessed through cBioPortal. The five cancer-internal tasks were:

- UCEC molecular subtype.
- COADREAD molecular subtype.
- HNSC HPV status.
- KIRC histologic grade.
- PRAD pathologic T stage.

## Data Handling

The repository does not redistribute complete raw public molecular matrices. Scripts in `scripts/` document the API calls and preprocessing steps used to regenerate analysis-ready tables. Compact result summaries and prediction outputs are included under `processed_results/`.

