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
| COADREAD_molecular_subtype | liquid_cfc_modality_sequence | liquid_nn | 3 | 68.0000 | 0.9510 | 0.0085 | 0.9311 | 0.0186 | 0.9179 | 0.0115 | 0.9672 | 0.0125 |
| COADREAD_molecular_subtype | extra_trees | tree_ensemble | 3 | 68.0000 | 0.9510 | 0.0225 | 0.8722 | 0.0591 | 0.9060 | 0.0495 | 0.9618 | 0.0039 |
| COADREAD_molecular_subtype | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 68.0000 | 0.9314 | 0.0085 | 0.8989 | 0.0626 | 0.8832 | 0.0359 | 0.9573 | 0.0053 |
| COADREAD_molecular_subtype | logistic_elasticnet | linear | 3 | 68.0000 | 0.9216 | 0.0170 | 0.8856 | 0.0418 | 0.8751 | 0.0058 | 0.9672 | 0.0041 |
| COADREAD_molecular_subtype | hist_gradient_boosting | boosting | 3 | 68.0000 | 0.9314 | 0.0085 | 0.8578 | 0.0510 | 0.8726 | 0.0300 | 0.9540 | 0.0029 |
| COADREAD_molecular_subtype | mlp_early_fusion | static_nn | 3 | 68.0000 | 0.8824 | 0.0778 | 0.8911 | 0.0476 | 0.8429 | 0.0800 | 0.9550 | 0.0117 |
| HNSC_HPV_status | logistic_elasticnet | linear | 3 | 74.0000 | 0.9685 | 0.0206 | 0.9690 | 0.0227 | 0.9425 | 0.0340 | 0.9779 | 0.0300 |
| HNSC_HPV_status | extra_trees | tree_ensemble | 3 | 74.0000 | 0.9730 | 0.0270 | 0.9091 | 0.0909 | 0.9386 | 0.0633 | 0.9947 | 0.0092 |
| HNSC_HPV_status | liquid_cfc_modality_sequence | liquid_nn | 3 | 74.0000 | 0.9640 | 0.0281 | 0.9663 | 0.0233 | 0.9359 | 0.0447 | 0.9769 | 0.0387 |
| HNSC_HPV_status | hist_gradient_boosting | boosting | 3 | 74.0000 | 0.9640 | 0.0206 | 0.9288 | 0.0549 | 0.9281 | 0.0417 | 0.9841 | 0.0175 |
| HNSC_HPV_status | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 74.0000 | 0.9234 | 0.0078 | 0.9425 | 0.0243 | 0.8715 | 0.0137 | 0.9774 | 0.0392 |
| HNSC_HPV_status | mlp_early_fusion | static_nn | 3 | 74.0000 | 0.8243 | 0.0270 | 0.8843 | 0.0269 | 0.7539 | 0.0286 | 0.9875 | 0.0112 |
| KIRC_grade_binary | extra_trees | tree_ensemble | 3 | 76.0000 | 0.7193 | 0.0402 | 0.7096 | 0.0437 | 0.7108 | 0.0439 | 0.7409 | 0.0546 |
| KIRC_grade_binary | hist_gradient_boosting | boosting | 3 | 76.0000 | 0.6974 | 0.0263 | 0.6898 | 0.0310 | 0.6887 | 0.0298 | 0.7423 | 0.0092 |
| KIRC_grade_binary | mlp_early_fusion | static_nn | 3 | 76.0000 | 0.6842 | 0.0603 | 0.6872 | 0.0604 | 0.6836 | 0.0603 | 0.7577 | 0.0823 |
| KIRC_grade_binary | liquid_cfc_modality_sequence | liquid_nn | 3 | 76.0000 | 0.6798 | 0.0423 | 0.6767 | 0.0403 | 0.6761 | 0.0408 | 0.7236 | 0.0795 |
| KIRC_grade_binary | logistic_elasticnet | linear | 3 | 76.0000 | 0.6579 | 0.0863 | 0.6569 | 0.0853 | 0.6558 | 0.0859 | 0.7124 | 0.0544 |
| KIRC_grade_binary | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 76.0000 | 0.6535 | 0.0532 | 0.6529 | 0.0436 | 0.6490 | 0.0501 | 0.7040 | 0.0429 |
| PRAD_pathologic_T_stage | hist_gradient_boosting | boosting | 3 | 74.0000 | 0.7252 | 0.0639 | 0.6998 | 0.0564 | 0.7031 | 0.0619 | 0.7549 | 0.0785 |
| PRAD_pathologic_T_stage | mlp_early_fusion | static_nn | 3 | 74.0000 | 0.6937 | 0.0475 | 0.7117 | 0.0208 | 0.6885 | 0.0401 | 0.7487 | 0.0381 |
| PRAD_pathologic_T_stage | extra_trees | tree_ensemble | 3 | 74.0000 | 0.6982 | 0.0206 | 0.6711 | 0.0284 | 0.6719 | 0.0247 | 0.7635 | 0.0436 |
| PRAD_pathologic_T_stage | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 74.0000 | 0.6757 | 0.0844 | 0.6832 | 0.0656 | 0.6687 | 0.0770 | 0.7332 | 0.0779 |
| PRAD_pathologic_T_stage | liquid_cfc_modality_sequence | liquid_nn | 3 | 74.0000 | 0.6622 | 0.1020 | 0.6677 | 0.0837 | 0.6523 | 0.0917 | 0.7360 | 0.0852 |
| PRAD_pathologic_T_stage | logistic_elasticnet | linear | 3 | 74.0000 | 0.6216 | 0.0619 | 0.6095 | 0.0519 | 0.6059 | 0.0552 | 0.7050 | 0.0841 |
| UCEC_molecular_subtype | logistic_elasticnet | linear | 3 | 77.0000 | 0.9177 | 0.0327 | 0.9235 | 0.0416 | 0.9099 | 0.0448 | 0.9820 | 0.0107 |
| UCEC_molecular_subtype | extra_trees | tree_ensemble | 3 | 77.0000 | 0.9177 | 0.0417 | 0.9059 | 0.0586 | 0.9057 | 0.0526 | 0.9838 | 0.0078 |
| UCEC_molecular_subtype | liquid_cfc_modality_sequence | liquid_nn | 3 | 77.0000 | 0.9004 | 0.0327 | 0.9078 | 0.0339 | 0.8874 | 0.0412 | 0.9852 | 0.0105 |
| UCEC_molecular_subtype | hist_gradient_boosting | boosting | 3 | 77.0000 | 0.8961 | 0.0566 | 0.9040 | 0.0630 | 0.8831 | 0.0614 | 0.9823 | 0.0062 |
| UCEC_molecular_subtype | small_liquid_cfc_modality_sequence | small_liquid_nn | 3 | 77.0000 | 0.8918 | 0.0417 | 0.8926 | 0.0639 | 0.8727 | 0.0544 | 0.9844 | 0.0117 |
| UCEC_molecular_subtype | mlp_early_fusion | static_nn | 3 | 77.0000 | 0.8658 | 0.0397 | 0.8875 | 0.0331 | 0.8556 | 0.0433 | 0.9785 | 0.0098 |

## Best Model per Task

| task | model | model_group | runs | n_mean | accuracy_mean | accuracy_sd | balanced_accuracy_mean | balanced_accuracy_sd | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | liquid_cfc_modality_sequence | liquid_nn | 3 | 68.0000 | 0.9510 | 0.0085 | 0.9311 | 0.0186 | 0.9179 | 0.0115 | 0.9672 | 0.0125 |
| HNSC_HPV_status | logistic_elasticnet | linear | 3 | 74.0000 | 0.9685 | 0.0206 | 0.9690 | 0.0227 | 0.9425 | 0.0340 | 0.9779 | 0.0300 |
| KIRC_grade_binary | extra_trees | tree_ensemble | 3 | 76.0000 | 0.7193 | 0.0402 | 0.7096 | 0.0437 | 0.7108 | 0.0439 | 0.7409 | 0.0546 |
| PRAD_pathologic_T_stage | hist_gradient_boosting | boosting | 3 | 74.0000 | 0.7252 | 0.0639 | 0.6998 | 0.0564 | 0.7031 | 0.0619 | 0.7549 | 0.0785 |
| UCEC_molecular_subtype | logistic_elasticnet | linear | 3 | 77.0000 | 0.9177 | 0.0327 | 0.9235 | 0.0416 | 0.9099 | 0.0448 | 0.9820 | 0.0107 |

## Interpretation

This benchmark is internal-split only. It should be interpreted as a multi-cancer task suite that complements the deeper BRCA external-validation study. External validation per cancer remains a future extension.
