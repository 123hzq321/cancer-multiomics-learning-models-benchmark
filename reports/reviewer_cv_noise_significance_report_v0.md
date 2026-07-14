# Reviewer CV, Significance, Noise, and Elbow Analysis

Core-aligned modalities: mRNA, GISTIC CNA, log2 CNA, methylation, and mutation. RPPA is excluded because the sample-alignment audit showed sparse RPPA coverage.

## Five-Fold CV: Best Model per Task

| task | model | folds | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean |
| --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | Liquid/CfC | 5 | 0.8802 | 0.0165 | 0.9715 |
| HNSC_HPV_status | Logistic ElasticNet | 5 | 0.9582 | 0.0144 | 0.9538 |
| KIRC_grade_binary | ExtraTrees | 5 | 0.6756 | 0.0381 | 0.7279 |
| PRAD_pathologic_T_stage | small-Liquid/CfC | 5 | 0.6932 | 0.0535 | 0.7622 |
| UCEC_molecular_subtype | ExtraTrees | 5 | 0.8944 | 0.0439 | 0.9829 |

## Top-vs-Runner-Up Fold-Level Significance

| task | best_model | second_model | mean_f1_diff | ci95_low | ci95_high | wilcoxon_or_sign_p | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | Liquid/CfC | ExtraTrees | 0.0144 | -0.0066 | 0.0354 | 0.3125 | 1.0000 |
| HNSC_HPV_status | Logistic ElasticNet | HistGradientBoosting | 0.0099 | -0.0076 | 0.0234 | 0.2500 | 1.0000 |
| KIRC_grade_binary | ExtraTrees | HistGradientBoosting | 0.0074 | -0.0202 | 0.0351 | 0.6250 | 1.0000 |
| PRAD_pathologic_T_stage | small-Liquid/CfC | Liquid/CfC | 0.0145 | -0.0340 | 0.0544 | 0.4375 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | Logistic ElasticNet | 0.0014 | -0.0340 | 0.0341 | 1.0000 | 1.0000 |

Interpretation: with only five folds per task, most top-vs-runner-up differences are not statistically significant after Holm correction. The manuscript should therefore avoid claiming universal superiority from small F1 differences.

## Noise Elbow for Best CV Model per Task

| task | model | clean_f1 | f1_at_sigma_0_20 | f1_at_sigma_0_50 | first_5pct_drop_sigma | elbow_sigma |
| --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | Liquid/CfC | 0.8802 | 0.8720 | 0.8632 | nan | 0.1000 |
| HNSC_HPV_status | Logistic ElasticNet | 0.9582 | 0.9496 | 0.9549 | nan | 0.2000 |
| KIRC_grade_binary | ExtraTrees | 0.6756 | 0.6572 | 0.5395 | 0.3000 | 0.2000 |
| PRAD_pathologic_T_stage | small-Liquid/CfC | 0.6932 | 0.6933 | 0.6695 | nan | 0.3000 |
| UCEC_molecular_subtype | ExtraTrees | 0.8944 | 0.8992 | 0.8660 | nan | 0.3000 |

Noise is added after training to standardized test features. `first_5pct_drop_sigma` is the first Gaussian noise SD at which macro F1 drops by at least 5% relative to the clean test set. `elbow_sigma` is the maximum-distance knee point of the clean-to-high-noise curve.

## Files

- Raw metrics: `work\data\reviewer_cv_noise_significance\processed\cv_noise_raw_metrics.csv`
- CV summary: `work\data\reviewer_cv_noise_significance\processed\fivefold_cv_summary.csv`
- Significance: `work\data\reviewer_cv_noise_significance\processed\fivefold_top_model_significance.csv`
- Noise summary: `work\data\reviewer_cv_noise_significance\processed\noise_robustness_summary.csv`
- Noise elbow: `work\data\reviewer_cv_noise_significance\processed\noise_elbow_summary.csv`