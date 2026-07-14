# TCGA+METABRIC Joined Training with New External Validation v0

## Scope

METABRIC is added to the training/validation pool. Independent external validation is performed on newly added cohorts.

- Joined training pool: TCGA BRCA + METABRIC marker panel samples.
- New external cohorts: SMC 2018, CPTAC 2020, and GSE96058/SCAN-B.
- Main feature sets: marker mRNA and marker multimodal expression-last.
- Models include original Liquid/CfC and small-Liquid/CfC.

## Trainable Parameters

| feature_set | model | trainable_parameters |
| --- | --- | --- |
| joined_marker_mrna | joined_marker_mrna__mlp | 19717 |
| joined_marker_mrna | joined_marker_mrna__liquid_cfc | 52613 |
| joined_marker_mrna | joined_marker_mrna__small_liquid_cfc | 14789 |
| joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__mlp | 40965 |
| joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__liquid_cfc | 63749 |
| joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__small_liquid_cfc | 20357 |

## Joined Training Pool Counts

| joint_split | joint_source | subtype_label | n |
| --- | --- | --- | --- |
| train | METABRIC | Basal | 176 |
| train | METABRIC | Her2 | 188 |
| train | METABRIC | LumA | 593 |
| train | METABRIC | LumB | 405 |
| train | METABRIC | Normal | 128 |
| train | TCGA | Basal | 147 |
| train | TCGA | Her2 | 69 |
| train | TCGA | LumA | 426 |
| train | TCGA | LumB | 166 |
| train | TCGA | Normal | 28 |
| valid | METABRIC | Basal | 33 |
| valid | METABRIC | Her2 | 36 |
| valid | METABRIC | LumA | 107 |
| valid | METABRIC | LumB | 70 |
| valid | METABRIC | Normal | 20 |
| valid | TCGA | Basal | 24 |
| valid | TCGA | Her2 | 9 |
| valid | TCGA | LumA | 73 |
| valid | TCGA | LumB | 31 |
| valid | TCGA | Normal | 8 |

## External Cohort Counts

| subtype_label | n | external |
| --- | --- | --- |
| LumB | 65 | smc_2018_external |
| LumA | 47 | smc_2018_external |
| Basal | 36 | smc_2018_external |
| Her2 | 18 | smc_2018_external |
| Normal | 2 | smc_2018_external |
| LumA | 57 | cptac_2020_external |
| Basal | 29 | cptac_2020_external |
| LumB | 17 | cptac_2020_external |
| Her2 | 14 | cptac_2020_external |
| Normal | 5 | cptac_2020_external |
| LumA | 1609 | scanb_external |
| LumB | 688 | scanb_external |
| Basal | 318 | scanb_external |
| Her2 | 307 | scanb_external |
| Normal | 215 | scanb_external |

## External Summary

| evaluation_split | feature_set | model | model_group | runs | n_mean | accuracy_mean | accuracy_sd | balanced_accuracy_mean | balanced_accuracy_sd | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cptac_2020_external | joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__liquid_cfc | liquid_nn | 3 | 122.0000 | 0.9153 | 0.0047 | 0.9049 | 0.0126 | 0.8806 | 0.0103 | 0.9933 | 0.0014 |
| cptac_2020_external | joined_marker_mrna | joined_marker_mrna__mlp | static_nn | 3 | 122.0000 | 0.9126 | 0.0125 | 0.9261 | 0.0016 | 0.8803 | 0.0211 | 0.9945 | 0.0020 |
| cptac_2020_external | joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__small_liquid_cfc | small_liquid_nn | 3 | 122.0000 | 0.8989 | 0.0171 | 0.9086 | 0.0180 | 0.8733 | 0.0278 | 0.9927 | 0.0014 |
| cptac_2020_external | joined_marker_mrna | joined_marker_mrna__small_liquid_cfc | small_liquid_nn | 3 | 122.0000 | 0.9153 | 0.0125 | 0.9157 | 0.0079 | 0.8676 | 0.0086 | 0.9942 | 0.0017 |
| cptac_2020_external | joined_marker_mrna | joined_marker_mrna__logistic | linear_baseline | 3 | 122.0000 | 0.9180 | 0.0000 | 0.8977 | 0.0000 | 0.8627 | 0.0000 | 0.9878 | 0.0000 |
| cptac_2020_external | joined_marker_mrna | joined_marker_mrna__liquid_cfc | liquid_nn | 3 | 122.0000 | 0.9180 | 0.0217 | 0.8449 | 0.0525 | 0.8584 | 0.0410 | 0.9901 | 0.0043 |
| cptac_2020_external | joined_marker_mrna | joined_marker_mrna__extra_trees | tree_baseline | 3 | 122.0000 | 0.8962 | 0.0095 | 0.8030 | 0.0487 | 0.8247 | 0.0507 | 0.9880 | 0.0008 |
| cptac_2020_external | joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__mlp | static_nn | 3 | 122.0000 | 0.8607 | 0.0217 | 0.8597 | 0.0689 | 0.8188 | 0.0547 | 0.9821 | 0.0041 |
| cptac_2020_external | joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__logistic | linear_baseline | 3 | 122.0000 | 0.8770 | 0.0000 | 0.7799 | 0.0000 | 0.7830 | 0.0000 | 0.9710 | 0.0000 |
| cptac_2020_external | joined_marker_multimodal_expression_last | joined_marker_multimodal_expression_last__extra_trees | tree_baseline | 3 | 122.0000 | 0.8907 | 0.0047 | 0.7437 | 0.0129 | 0.7476 | 0.0337 | 0.9827 | 0.0025 |
| scanb_external | joined_marker_mrna | joined_marker_mrna__extra_trees | tree_baseline | 3 | 3137.0000 | 0.8804 | 0.0034 | 0.8544 | 0.0038 | 0.8493 | 0.0046 | 0.9831 | 0.0002 |
| scanb_external | joined_marker_mrna | joined_marker_mrna__liquid_cfc | liquid_nn | 3 | 3137.0000 | 0.8493 | 0.0088 | 0.8629 | 0.0048 | 0.8172 | 0.0020 | 0.9830 | 0.0010 |
| scanb_external | joined_marker_mrna | joined_marker_mrna__mlp | static_nn | 3 | 3137.0000 | 0.8218 | 0.0242 | 0.8733 | 0.0111 | 0.8033 | 0.0214 | 0.9847 | 0.0013 |
| scanb_external | joined_marker_mrna | joined_marker_mrna__small_liquid_cfc | small_liquid_nn | 3 | 3137.0000 | 0.7999 | 0.0149 | 0.8676 | 0.0084 | 0.7857 | 0.0069 | 0.9818 | 0.0014 |
| scanb_external | joined_marker_mrna | joined_marker_mrna__logistic | linear_baseline | 3 | 3137.0000 | 0.8106 | 0.0000 | 0.8603 | 0.0000 | 0.7851 | 0.0000 | 0.9801 | 0.0000 |
| smc_2018_external | joined_marker_mrna | joined_marker_mrna__extra_trees | tree_baseline | 3 | 168.0000 | 0.8671 | 0.0034 | 0.9309 | 0.0025 | 0.9124 | 0.0036 | 0.9894 | 0.0004 |
| smc_2018_external | joined_marker_mrna | joined_marker_mrna__liquid_cfc | liquid_nn | 3 | 168.0000 | 0.8115 | 0.0225 | 0.8754 | 0.0159 | 0.8178 | 0.0194 | 0.9854 | 0.0020 |
| smc_2018_external | joined_marker_mrna | joined_marker_mrna__small_liquid_cfc | small_liquid_nn | 3 | 168.0000 | 0.8234 | 0.0225 | 0.8888 | 0.0162 | 0.7948 | 0.0347 | 0.9838 | 0.0014 |
| smc_2018_external | joined_marker_mrna | joined_marker_mrna__mlp | static_nn | 3 | 168.0000 | 0.8155 | 0.0119 | 0.8820 | 0.0077 | 0.7799 | 0.0057 | 0.9840 | 0.0010 |
| smc_2018_external | joined_marker_mrna | joined_marker_mrna__logistic | linear_baseline | 3 | 168.0000 | 0.8036 | 0.0000 | 0.8732 | 0.0000 | 0.7751 | 0.0000 | 0.9858 | 0.0000 |

## Best External Model: `joined_marker_mrna__extra_trees` on `smc_2018_external`

### Confusion Matrix

| true_label | pred__Basal | pred__Her2 | pred__LumA | pred__LumB | pred__Normal |
| --- | --- | --- | --- | --- | --- |
| true__Basal | 36 | 0 | 0 | 0 | 0 |
| true__Her2 | 0 | 18 | 0 | 0 | 0 |
| true__LumA | 0 | 0 | 47 | 0 | 0 |
| true__LumB | 1 | 1 | 20 | 43 | 0 |
| true__Normal | 0 | 0 | 0 | 0 | 2 |

### Classification Report

| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Basal | 0.9730 | 1.0000 | 0.9863 | 36.0000 |
| Her2 | 0.9474 | 1.0000 | 0.9730 | 18.0000 |
| LumA | 0.7015 | 1.0000 | 0.8246 | 47.0000 |
| LumB | 1.0000 | 0.6615 | 0.7963 | 65.0000 |
| Normal | 1.0000 | 1.0000 | 1.0000 | 2.0000 |
| accuracy | 0.8690 | 0.8690 | 0.8690 | 0.8690 |
| macro avg | 0.9244 | 0.9323 | 0.9160 | 168.0000 |
| weighted avg | 0.9051 | 0.8690 | 0.8663 | 168.0000 |

## Interpretation

This analysis answers whether increasing the training data by adding METABRIC and shrinking the Liquid/CfC model improves external transfer. Because SCAN-B and SMC are mRNA-only external validations, multimodal conclusions should rely mainly on CPTAC and internal validation.
