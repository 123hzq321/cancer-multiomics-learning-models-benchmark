# Large-Scale mRNA Baseline Expansion v0

## Scope

Expand training scale for mRNA marker subtype prediction by adding SCAN-B and SMC 2018 to the TCGA+METABRIC training pool, while keeping CPTAC 2020 as the independent external validation cohort.

## Training Scale Counts

| training_scale | split | subtype_label | n |
| --- | --- | --- | --- |
| joined_tcga_metabric | train | LumA | 1019 |
| joined_tcga_metabric | train | LumB | 571 |
| joined_tcga_metabric | train | Basal | 323 |
| joined_tcga_metabric | train | Her2 | 257 |
| joined_tcga_metabric | train | Normal | 156 |
| joined_tcga_metabric | valid | LumA | 180 |
| joined_tcga_metabric | valid | LumB | 101 |
| joined_tcga_metabric | valid | Basal | 57 |
| joined_tcga_metabric | valid | Her2 | 45 |
| joined_tcga_metabric | valid | Normal | 28 |
| expanded_tcga_metabric_scanb_smc | train | LumA | 2426 |
| expanded_tcga_metabric_scanb_smc | train | LumB | 1211 |
| expanded_tcga_metabric_scanb_smc | train | Basal | 624 |
| expanded_tcga_metabric_scanb_smc | train | Her2 | 533 |
| expanded_tcga_metabric_scanb_smc | train | Normal | 341 |
| expanded_tcga_metabric_scanb_smc | valid | LumA | 429 |
| expanded_tcga_metabric_scanb_smc | valid | LumB | 214 |
| expanded_tcga_metabric_scanb_smc | valid | Basal | 110 |
| expanded_tcga_metabric_scanb_smc | valid | Her2 | 94 |
| expanded_tcga_metabric_scanb_smc | valid | Normal | 60 |

## CPTAC External Summary

| training_scale | split | model | model_group | runs | n_mean | accuracy_mean | accuracy_sd | balanced_accuracy_mean | balanced_accuracy_sd | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanded_tcga_metabric_scanb_smc | cptac_external | sklearn_mlp | static_nn | 3 | 122.0000 | 0.9317 | 0.0206 | 0.8337 | 0.0616 | 0.8559 | 0.0597 | 0.9940 | 0.0027 |
| expanded_tcga_metabric_scanb_smc | cptac_external | logistic_elasticnet | linear | 3 | 122.0000 | 0.9317 | 0.0125 | 0.8436 | 0.0178 | 0.8509 | 0.0218 | 0.9911 | 0.0024 |
| expanded_tcga_metabric_scanb_smc | cptac_external | hist_gradient_boosting | boosting | 3 | 122.0000 | 0.9180 | 0.0164 | 0.8240 | 0.0521 | 0.8443 | 0.0573 | 0.9869 | 0.0007 |
| expanded_tcga_metabric_scanb_smc | cptac_external | logistic_l2 | linear | 3 | 122.0000 | 0.9235 | 0.0095 | 0.8241 | 0.0173 | 0.8312 | 0.0135 | 0.9896 | 0.0028 |
| expanded_tcga_metabric_scanb_smc | cptac_external | gaussian_nb | probabilistic | 3 | 122.0000 | 0.8689 | 0.0000 | 0.8659 | 0.0000 | 0.8184 | 0.0000 | 0.9814 | 0.0002 |
| expanded_tcga_metabric_scanb_smc | cptac_external | lda_shrinkage | linear_discriminant | 3 | 122.0000 | 0.8989 | 0.0047 | 0.7989 | 0.0040 | 0.8087 | 0.0085 | 0.9860 | 0.0014 |
| expanded_tcga_metabric_scanb_smc | cptac_external | extra_trees | tree_ensemble | 3 | 122.0000 | 0.9098 | 0.0082 | 0.7710 | 0.0080 | 0.7923 | 0.0078 | 0.9905 | 0.0009 |
| expanded_tcga_metabric_scanb_smc | cptac_external | ridge_classifier | linear | 3 | 122.0000 | 0.8361 | 0.0142 | 0.8416 | 0.0123 | 0.7731 | 0.0143 | 0.9725 | 0.0002 |
| expanded_tcga_metabric_scanb_smc | cptac_external | random_forest | tree_ensemble | 3 | 122.0000 | 0.9044 | 0.0095 | 0.7510 | 0.0299 | 0.7658 | 0.0447 | 0.9902 | 0.0009 |
| expanded_tcga_metabric_scanb_smc | cptac_external | linear_svm_calibrated | linear_svm | 3 | 122.0000 | 0.8989 | 0.0047 | 0.7386 | 0.0206 | 0.7347 | 0.0389 | 0.9754 | 0.0022 |
| expanded_tcga_metabric_scanb_smc | cptac_external | knn_distance | instance_based | 3 | 122.0000 | 0.8443 | 0.0082 | 0.6440 | 0.0130 | 0.6674 | 0.0121 | 0.9768 | 0.0008 |
| joined_tcga_metabric | cptac_external | logistic_elasticnet | linear | 3 | 122.0000 | 0.9344 | 0.0000 | 0.9130 | 0.0000 | 0.8830 | 0.0000 | 0.9899 | 0.0000 |
| joined_tcga_metabric | cptac_external | sklearn_mlp | static_nn | 3 | 122.0000 | 0.9344 | 0.0142 | 0.8583 | 0.0447 | 0.8823 | 0.0440 | 0.9940 | 0.0029 |
| joined_tcga_metabric | cptac_external | logistic_l2 | linear | 3 | 122.0000 | 0.9180 | 0.0000 | 0.8977 | 0.0000 | 0.8627 | 0.0000 | 0.9878 | 0.0000 |
| joined_tcga_metabric | cptac_external | extra_trees | tree_ensemble | 3 | 122.0000 | 0.8962 | 0.0095 | 0.8030 | 0.0487 | 0.8247 | 0.0507 | 0.9880 | 0.0008 |
| joined_tcga_metabric | cptac_external | random_forest | tree_ensemble | 3 | 122.0000 | 0.9016 | 0.0000 | 0.7905 | 0.0000 | 0.8216 | 0.0000 | 0.9881 | 0.0007 |
| joined_tcga_metabric | cptac_external | gaussian_nb | probabilistic | 3 | 122.0000 | 0.8689 | 0.0000 | 0.8659 | 0.0000 | 0.8075 | 0.0000 | 0.9816 | 0.0000 |
| joined_tcga_metabric | cptac_external | hist_gradient_boosting | boosting | 3 | 122.0000 | 0.8934 | 0.0000 | 0.7869 | 0.0000 | 0.8037 | 0.0000 | 0.9848 | 0.0000 |
| joined_tcga_metabric | cptac_external | lda_shrinkage | linear_discriminant | 3 | 122.0000 | 0.8689 | 0.0000 | 0.7590 | 0.0000 | 0.7458 | 0.0000 | 0.9856 | 0.0000 |
| joined_tcga_metabric | cptac_external | linear_svm_calibrated | linear_svm | 3 | 122.0000 | 0.9098 | 0.0000 | 0.7400 | 0.0000 | 0.7278 | 0.0000 | 0.9660 | 0.0000 |
| joined_tcga_metabric | cptac_external | ridge_classifier | linear | 3 | 122.0000 | 0.8033 | 0.0000 | 0.8029 | 0.0000 | 0.7256 | 0.0000 | 0.9651 | 0.0000 |
| joined_tcga_metabric | cptac_external | knn_distance | instance_based | 3 | 122.0000 | 0.8361 | 0.0000 | 0.6323 | 0.0000 | 0.6512 | 0.0000 | 0.9737 | 0.0000 |

## Best CPTAC External Confusion Matrix

| true_label | pred__Basal | pred__Her2 | pred__LumA | pred__LumB | pred__Normal |
| --- | --- | --- | --- | --- | --- |
| true__Basal | 28 | 1 | 0 | 0 | 0 |
| true__Her2 | 0 | 13 | 0 | 1 | 0 |
| true__LumA | 0 | 0 | 53 | 1 | 3 |
| true__LumB | 0 | 0 | 1 | 16 | 0 |
| true__Normal | 0 | 1 | 0 | 0 | 4 |

## Best CPTAC External Classification Report

| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Basal | 1.0000 | 0.9655 | 0.9825 | 29.0000 |
| Her2 | 0.8667 | 0.9286 | 0.8966 | 14.0000 |
| LumA | 0.9815 | 0.9298 | 0.9550 | 57.0000 |
| LumB | 0.8889 | 0.9412 | 0.9143 | 17.0000 |
| Normal | 0.5714 | 0.8000 | 0.6667 | 5.0000 |
| accuracy | 0.9344 | 0.9344 | 0.9344 | 0.9344 |
| macro avg | 0.8617 | 0.9130 | 0.8830 | 122.0000 |
| weighted avg | 0.9430 | 0.9344 | 0.9373 | 122.0000 |

## Interpretation

This experiment tests whether increased sample size changes the ranking of classical baselines. It is intentionally mRNA-only because SCAN-B and SMC do not provide the same complete CNA/mutation multimodal feature set as CPTAC.
