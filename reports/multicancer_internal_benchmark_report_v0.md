# Multi-Cancer Internal Task Benchmark v0

## Scope

Extend the BRCA-focused benchmark with cancer-internal tasks from TCGA PanCancer Atlas cohorts.

Tasks:

- `UCEC_molecular_subtype`: UCEC molecular subtype: CN_HIGH, CN_LOW, MSI, POLE.
- `COADREAD_molecular_subtype`: COADREAD molecular subtype collapsed across colon/rectal labels: CIN, GS, MSI. Rare POLE classes are excluded.
- `HNSC_HPV_status`: HNSC HPV status from PanCancer subtype labels.
- `KIRC_grade_binary`: KIRC histologic grade, G1/G2 versus G3/G4.
- `PRAD_pathologic_T_stage`: PRAD pathologic T stage, T2 versus T3/T4.

Models: elastic-net logistic regression, ExtraTrees, HistGradientBoosting, MLP, Liquid/CfC, and small-Liquid/CfC.

## Task Label Counts

| task | label | n |
| --- | --- | --- |
| UCEC_molecular_subtype | CN_HIGH | 163 |
| UCEC_molecular_subtype | MSI | 148 |
| UCEC_molecular_subtype | CN_LOW | 147 |
| UCEC_molecular_subtype | POLE | 49 |
| COADREAD_molecular_subtype | CIN | 328 |
| COADREAD_molecular_subtype | MSI | 63 |
| COADREAD_molecular_subtype | GS | 58 |
| HNSC_HPV_status | HPV_NEG | 415 |
| HNSC_HPV_status | HPV_POS | 72 |
| KIRC_grade_binary | HIGH_GRADE | 277 |
| KIRC_grade_binary | LOW_GRADE | 227 |
| PRAD_pathologic_T_stage | T3_T4 | 300 |
| PRAD_pathologic_T_stage | T2 | 187 |

## Test Summary

| task | model | model_group | runs | n_mean | accuracy_mean | accuracy_sd | balanced_accuracy_mean | balanced_accuracy_sd | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | extra_trees | tree_ensemble | 3 | 68.0000 | 0.9510 | 0.0225 | 0.8722 | 0.0591 | 0.9060 | 0.0495 | 0.9546 | 0.0071 |
| COADREAD_molecular_subtype | mlp_early_fusion | static_nn | 3 | 68.0000 | 0.9265 | 0.0294 | 0.9228 | 0.0278 | 0.8907 | 0.0428 | 0.9731 | 0.0083 |
| COADREAD_molecular_subtype | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 68.0000 | 0.9265 | 0.0147 | 0.8878 | 0.0534 | 0.8756 | 0.0301 | 0.9558 | 0.0214 |
| COADREAD_molecular_subtype | hist_gradient_boosting | boosting | 3 | 68.0000 | 0.9363 | 0.0170 | 0.8483 | 0.0562 | 0.8749 | 0.0342 | 0.9594 | 0.0102 |
| COADREAD_molecular_subtype | logistic_elasticnet | linear | 3 | 68.0000 | 0.9167 | 0.0085 | 0.8717 | 0.0645 | 0.8630 | 0.0195 | 0.9684 | 0.0014 |
| COADREAD_molecular_subtype | liquid_cfc_modality_sequence | liquid_nn | 3 | 68.0000 | 0.9216 | 0.0085 | 0.8622 | 0.0342 | 0.8577 | 0.0255 | 0.9687 | 0.0037 |
| HNSC_HPV_status | extra_trees | tree_ensemble | 3 | 74.0000 | 0.9775 | 0.0206 | 0.9242 | 0.0694 | 0.9505 | 0.0459 | 0.9942 | 0.0100 |
| HNSC_HPV_status | logistic_elasticnet | linear | 3 | 74.0000 | 0.9640 | 0.0281 | 0.9663 | 0.0233 | 0.9359 | 0.0447 | 0.9774 | 0.0297 |
| HNSC_HPV_status | hist_gradient_boosting | boosting | 3 | 74.0000 | 0.9640 | 0.0206 | 0.9288 | 0.0549 | 0.9281 | 0.0417 | 0.9856 | 0.0161 |
| HNSC_HPV_status | liquid_cfc_modality_sequence | liquid_nn | 3 | 74.0000 | 0.9459 | 0.0234 | 0.9307 | 0.0400 | 0.9015 | 0.0364 | 0.9769 | 0.0316 |
| HNSC_HPV_status | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 74.0000 | 0.9234 | 0.0978 | 0.9425 | 0.0487 | 0.8885 | 0.1265 | 0.9639 | 0.0341 |
| HNSC_HPV_status | mlp_early_fusion | static_nn | 3 | 74.0000 | 0.8288 | 0.0475 | 0.8870 | 0.0367 | 0.7599 | 0.0511 | 0.9923 | 0.0133 |
| KIRC_grade_binary | hist_gradient_boosting | boosting | 3 | 76.0000 | 0.7061 | 0.0076 | 0.7005 | 0.0166 | 0.6996 | 0.0158 | 0.7556 | 0.0083 |
| KIRC_grade_binary | extra_trees | tree_ensemble | 3 | 76.0000 | 0.6974 | 0.0456 | 0.6870 | 0.0485 | 0.6878 | 0.0492 | 0.7526 | 0.0408 |
| KIRC_grade_binary | liquid_cfc_modality_sequence | liquid_nn | 3 | 76.0000 | 0.6886 | 0.0201 | 0.6902 | 0.0222 | 0.6858 | 0.0214 | 0.7696 | 0.0512 |
| KIRC_grade_binary | logistic_elasticnet | linear | 3 | 76.0000 | 0.6798 | 0.0793 | 0.6758 | 0.0775 | 0.6760 | 0.0785 | 0.7047 | 0.0462 |
| KIRC_grade_binary | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 76.0000 | 0.6798 | 0.0593 | 0.6758 | 0.0593 | 0.6750 | 0.0592 | 0.7183 | 0.0427 |
| KIRC_grade_binary | mlp_early_fusion | static_nn | 3 | 76.0000 | 0.6623 | 0.0593 | 0.6730 | 0.0496 | 0.6595 | 0.0621 | 0.7512 | 0.0405 |
| PRAD_pathologic_T_stage | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 74.0000 | 0.7117 | 0.0680 | 0.7239 | 0.0580 | 0.7071 | 0.0651 | 0.7717 | 0.0444 |
| PRAD_pathologic_T_stage | hist_gradient_boosting | boosting | 3 | 74.0000 | 0.7117 | 0.0281 | 0.6913 | 0.0252 | 0.6920 | 0.0270 | 0.7479 | 0.0649 |
| PRAD_pathologic_T_stage | extra_trees | tree_ensemble | 3 | 74.0000 | 0.7072 | 0.0206 | 0.6760 | 0.0146 | 0.6792 | 0.0164 | 0.7795 | 0.0483 |
| PRAD_pathologic_T_stage | mlp_early_fusion | static_nn | 3 | 74.0000 | 0.6757 | 0.0589 | 0.7112 | 0.0420 | 0.6743 | 0.0568 | 0.7681 | 0.0230 |
| PRAD_pathologic_T_stage | liquid_cfc_modality_sequence | liquid_nn | 3 | 74.0000 | 0.6802 | 0.0768 | 0.6752 | 0.0630 | 0.6689 | 0.0705 | 0.7319 | 0.0935 |
| PRAD_pathologic_T_stage | logistic_elasticnet | linear | 3 | 74.0000 | 0.6306 | 0.0563 | 0.6190 | 0.0440 | 0.6143 | 0.0475 | 0.7037 | 0.1007 |
| UCEC_molecular_subtype | logistic_elasticnet | linear | 3 | 77.0000 | 0.9221 | 0.0390 | 0.9268 | 0.0458 | 0.9134 | 0.0502 | 0.9831 | 0.0097 |
| UCEC_molecular_subtype | extra_trees | tree_ensemble | 3 | 77.0000 | 0.9177 | 0.0417 | 0.9059 | 0.0586 | 0.9057 | 0.0526 | 0.9842 | 0.0078 |
| UCEC_molecular_subtype | liquid_cfc_modality_sequence | liquid_nn | 3 | 77.0000 | 0.9134 | 0.0397 | 0.9023 | 0.0575 | 0.8895 | 0.0511 | 0.9862 | 0.0075 |
| UCEC_molecular_subtype | mlp_early_fusion | static_nn | 3 | 77.0000 | 0.8961 | 0.0260 | 0.9134 | 0.0211 | 0.8876 | 0.0268 | 0.9781 | 0.0117 |
| UCEC_molecular_subtype | hist_gradient_boosting | boosting | 3 | 77.0000 | 0.9004 | 0.0492 | 0.9078 | 0.0569 | 0.8871 | 0.0548 | 0.9831 | 0.0064 |
| UCEC_molecular_subtype | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 77.0000 | 0.9048 | 0.0525 | 0.8950 | 0.0879 | 0.8760 | 0.0770 | 0.9848 | 0.0135 |

## Best Model per Task

| task | model | model_group | runs | n_mean | accuracy_mean | accuracy_sd | balanced_accuracy_mean | balanced_accuracy_sd | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | extra_trees | tree_ensemble | 3 | 68.0000 | 0.9510 | 0.0225 | 0.8722 | 0.0591 | 0.9060 | 0.0495 | 0.9546 | 0.0071 |
| HNSC_HPV_status | extra_trees | tree_ensemble | 3 | 74.0000 | 0.9775 | 0.0206 | 0.9242 | 0.0694 | 0.9505 | 0.0459 | 0.9942 | 0.0100 |
| KIRC_grade_binary | hist_gradient_boosting | boosting | 3 | 76.0000 | 0.7061 | 0.0076 | 0.7005 | 0.0166 | 0.6996 | 0.0158 | 0.7556 | 0.0083 |
| PRAD_pathologic_T_stage | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 74.0000 | 0.7117 | 0.0680 | 0.7239 | 0.0580 | 0.7071 | 0.0651 | 0.7717 | 0.0444 |
| UCEC_molecular_subtype | logistic_elasticnet | linear | 3 | 77.0000 | 0.9221 | 0.0390 | 0.9268 | 0.0458 | 0.9134 | 0.0502 | 0.9831 | 0.0097 |

## Interpretation

This benchmark is internal-split only. It should be interpreted as a multi-cancer task suite that complements the deeper BRCA external-validation study. External validation per cancer remains a future extension.
