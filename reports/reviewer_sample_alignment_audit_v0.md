# Sample Alignment Audit for Reviewer Response

This audit checks whether the processed analysis tables are organized at a single sample row per labelled tumor sample and whether modality matrices are attached to that same sample identifier. It does not re-train models.

## Sample-Level Alignment Summary

| analysis | dataset | sample_unit | alignment_key | labelled_samples | unique_sample_ids | duplicate_sample_ids | n_classes | modalities | compatible_experiment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BRCA joined training | TCGA+METABRIC | tumor sample | sampleId | 2737 | 2737 | 0 | 5 | mrna_z, cna, mutation | mRNA and multimodal |
| BRCA external validation | CPTAC 2020 | tumor sample | sampleId | 122 | 122 | 0 | 5 | mrna_z, cna, mutation | mRNA and multimodal |
| BRCA external validation | SMC 2018 | tumor sample | sampleId | 168 | 168 | 0 | 5 | mrna_z, mutation | mRNA-only reported for cross-cohort comparison |
| BRCA external validation | SCAN-B/GSE96058 | tumor sample | sampleId | 3137 | 3137 | 0 | 5 | mrna_z | mRNA-only |
| TCGA cancer-internal benchmark | UCEC molecular subtype | primary tumor sample | sampleId | 507 | 507 | 0 | 4 | mrna, gistic, log2cna, methylation, rppa, mutation | aligned internal multi-omics |
| TCGA cancer-internal benchmark | COADREAD molecular subtype | primary tumor sample | sampleId | 449 | 449 | 0 | 3 | mrna, gistic, log2cna, methylation, rppa, mutation | aligned internal multi-omics |
| TCGA cancer-internal benchmark | HNSC HPV status | primary tumor sample | sampleId | 487 | 487 | 0 | 2 | mrna, gistic, log2cna, methylation, rppa, mutation | aligned internal multi-omics |
| TCGA cancer-internal benchmark | KIRC grade | primary tumor sample | sampleId | 504 | 504 | 0 | 2 | mrna, gistic, log2cna, methylation, rppa, mutation | aligned internal multi-omics |
| TCGA cancer-internal benchmark | PRAD pathologic T stage | primary tumor sample | sampleId | 487 | 487 | 0 | 2 | mrna, gistic, log2cna, methylation, rppa, mutation | aligned internal multi-omics |

## Modality Coverage

| dataset | modality | n_features | rows_with_any_observed_feature | labelled_rows | feature_missing_percent |
| --- | --- | --- | --- | --- | --- |
| TCGA+METABRIC | mrna_z | 83 | 2737 | 2737 | 0.0% |
| TCGA+METABRIC | cna | 83 | 2737 | 2737 | 0.0% |
| TCGA+METABRIC | mutation | 83 | 2737 | 2737 | 0.0% |
| CPTAC 2020 | mrna_z | 83 | 122 | 122 | 1.8% |
| CPTAC 2020 | cna | 83 | 122 | 122 | 0.0% |
| CPTAC 2020 | mutation | 83 | 122 | 122 | 0.0% |
| SMC 2018 | mrna_z | 83 | 168 | 168 | 1.2% |
| SMC 2018 | mutation | 83 | 168 | 168 | 0.0% |
| SCAN-B/GSE96058 | mrna_z | 83 | 3137 | 3137 | 0.0% |
| UCEC molecular subtype | mrna | 469 | 507 | 507 | 1.3% |
| UCEC molecular subtype | gistic | 469 | 507 | 507 | 0.2% |
| UCEC molecular subtype | log2cna | 469 | 507 | 507 | 0.2% |
| UCEC molecular subtype | methylation | 387 | 507 | 507 | 0.6% |
| UCEC molecular subtype | rppa | 469 | 405 | 507 | 88.9% |
| UCEC molecular subtype | mutation | 469 | 507 | 507 | 0.0% |
| COADREAD molecular subtype | mrna | 469 | 449 | 449 | 0.5% |
| COADREAD molecular subtype | gistic | 469 | 449 | 449 | 0.2% |
| COADREAD molecular subtype | log2cna | 469 | 449 | 449 | 0.2% |
| COADREAD molecular subtype | methylation | 387 | 449 | 449 | 0.6% |
| COADREAD molecular subtype | rppa | 469 | 356 | 449 | 89.0% |
| COADREAD molecular subtype | mutation | 469 | 449 | 449 | 0.0% |
| HNSC HPV status | mrna | 469 | 487 | 487 | 0.0% |
| HNSC HPV status | gistic | 469 | 487 | 487 | 0.2% |
| HNSC HPV status | log2cna | 469 | 487 | 487 | 0.2% |
| HNSC HPV status | methylation | 387 | 487 | 487 | 0.6% |
| HNSC HPV status | rppa | 469 | 202 | 487 | 94.3% |
| HNSC HPV status | mutation | 469 | 487 | 487 | 0.0% |
| KIRC grade | mrna | 469 | 502 | 504 | 0.4% |
| KIRC grade | gistic | 469 | 501 | 504 | 0.8% |
| KIRC grade | log2cna | 469 | 501 | 504 | 0.8% |
| KIRC grade | methylation | 387 | 503 | 504 | 0.9% |
| KIRC grade | rppa | 469 | 451 | 504 | 87.6% |
| KIRC grade | mutation | 469 | 504 | 504 | 0.0% |
| PRAD pathologic T stage | mrna | 469 | 486 | 487 | 0.2% |
| PRAD pathologic T stage | gistic | 469 | 482 | 487 | 1.2% |
| PRAD pathologic T stage | log2cna | 469 | 482 | 487 | 1.2% |
| PRAD pathologic T stage | methylation | 387 | 487 | 487 | 0.5% |
| PRAD pathologic T stage | rppa | 469 | 345 | 487 | 90.2% |
| PRAD pathologic T stage | mutation | 469 | 487 | 487 | 0.0% |

## Interpretation

- The primary alignment key is `sampleId`; patient-level clinical labels are merged to sample rows through `patientId` only for tasks whose labels are patient-level annotations.
- No labelled analysis table contains duplicated `sampleId` rows after task filtering.
- CPTAC 2020 supports mRNA+CNA+mutation external validation. SMC 2018 and SCAN-B lack the complete multimodal marker set, so they are used only for compatible mRNA-focused external comparisons.
- Missing molecular values are handled by training-split median imputation and scaling; external cohorts are transformed with preprocessing objects fitted on the training split only.

CSV outputs: `work\data\reviewer_alignment_audit\processed\sample_alignment_summary.csv` and `work\data\reviewer_alignment_audit\processed\sample_alignment_modality_coverage.csv`.