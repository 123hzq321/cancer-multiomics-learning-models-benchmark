# Reviewer CV, Significance, Noise, and Elbow Analysis

Core-aligned modalities: mRNA, GISTIC CNA, log2 CNA, methylation, and mutation. RPPA is excluded because the sample-alignment audit showed sparse RPPA coverage.

## Ten-Fold CV: Best Model per Task

| task | model | folds | f1_macro_mean | f1_macro_sd | roc_auc_ovr_macro_mean |
| --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | ExtraTrees | 10 | 0.8676 | 0.0693 | 0.9554 |
| HNSC_HPV_status | Logistic ElasticNet | 10 | 0.9469 | 0.0376 | 0.9611 |
| KIRC_grade_binary | Random Forest | 10 | 0.6784 | 0.0425 | 0.7335 |
| PRAD_pathologic_T_stage | Random Forest | 10 | 0.6839 | 0.0873 | 0.7734 |
| UCEC_molecular_subtype | ExtraTrees | 10 | 0.8944 | 0.0677 | 0.9831 |

## Top-vs-Runner-Up Fold-Level Significance

| task | top_model | runner_up | top_wins | runner_up_wins | ties | mean_f1_diff | ci95_low | ci95_high | raw_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | ExtraTrees | small-Liquid/CfC | 5 | 5 | 0 | 0.0048 | -0.0431 | 0.0514 | 1.0000 |
| HNSC_HPV_status | Logistic ElasticNet | HistGradientBoosting | 5 | 2 | 3 | 0.0159 | -0.0141 | 0.0470 | 0.3750 |
| KIRC_grade_binary | Random Forest | ExtraTrees | 4 | 4 | 2 | 0.0122 | -0.0057 | 0.0310 | 0.3125 |
| PRAD_pathologic_T_stage | Random Forest | HistGradientBoosting | 7 | 3 | 0 | 0.0021 | -0.0451 | 0.0389 | 0.4316 |
| UCEC_molecular_subtype | ExtraTrees | Logistic ElasticNet | 7 | 3 | 0 | 0.0053 | -0.0391 | 0.0471 | 0.6953 |

## Top-vs-All Fold-Level Significance

| task | top_model | comparator | top_wins | comparator_wins | ties | mean_f1_diff | ci95_low | ci95_high | wilcoxon_or_sign_p | holm_within_task_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | ExtraTrees | small-Liquid/CfC | 5 | 5 | 0 | 0.0048 | -0.0419 | 0.0510 | 1.0000 | 1.0000 |
| COADREAD_molecular_subtype | ExtraTrees | MLP | 5 | 4 | 1 | 0.0080 | -0.0192 | 0.0339 | 0.6523 | 1.0000 |
| COADREAD_molecular_subtype | ExtraTrees | Logistic ElasticNet | 7 | 3 | 0 | 0.0224 | -0.0109 | 0.0536 | 0.1934 | 0.7734 |
| COADREAD_molecular_subtype | ExtraTrees | Liquid/CfC | 6 | 3 | 1 | 0.0254 | -0.0160 | 0.0659 | 0.2500 | 0.7734 |
| COADREAD_molecular_subtype | ExtraTrees | Random Forest | 6 | 3 | 1 | 0.0376 | 0.0037 | 0.0719 | 0.0977 | 0.5859 |
| COADREAD_molecular_subtype | ExtraTrees | HistGradientBoosting | 8 | 1 | 1 | 0.0441 | -0.0088 | 0.0845 | 0.0977 | 0.5859 |
| HNSC_HPV_status | Logistic ElasticNet | HistGradientBoosting | 5 | 2 | 3 | 0.0159 | -0.0129 | 0.0455 | 0.3750 | 0.4453 |
| HNSC_HPV_status | Logistic ElasticNet | ExtraTrees | 7 | 2 | 1 | 0.0221 | -0.0157 | 0.0533 | 0.2031 | 0.4453 |
| HNSC_HPV_status | Logistic ElasticNet | Random Forest | 8 | 1 | 1 | 0.0423 | 0.0083 | 0.0752 | 0.0547 | 0.2188 |
| HNSC_HPV_status | Logistic ElasticNet | Liquid/CfC | 6 | 2 | 2 | 0.0675 | 0.0064 | 0.1392 | 0.1484 | 0.4453 |
| HNSC_HPV_status | Logistic ElasticNet | small-Liquid/CfC | 6 | 2 | 2 | 0.0722 | 0.0174 | 0.1354 | 0.0391 | 0.1953 |
| HNSC_HPV_status | Logistic ElasticNet | MLP | 8 | 1 | 1 | 0.0939 | 0.0444 | 0.1382 | 0.0078 | 0.0469 |
| KIRC_grade_binary | Random Forest | ExtraTrees | 4 | 4 | 2 | 0.0122 | -0.0058 | 0.0319 | 0.3125 | 0.6250 |
| KIRC_grade_binary | Random Forest | HistGradientBoosting | 5 | 4 | 1 | 0.0255 | -0.0171 | 0.0682 | 0.3594 | 0.6250 |
| KIRC_grade_binary | Random Forest | small-Liquid/CfC | 7 | 3 | 0 | 0.0283 | 0.0012 | 0.0543 | 0.1055 | 0.5273 |
| KIRC_grade_binary | Random Forest | Liquid/CfC | 7 | 3 | 0 | 0.0331 | 0.0007 | 0.0689 | 0.1602 | 0.5273 |
| KIRC_grade_binary | Random Forest | Logistic ElasticNet | 8 | 2 | 0 | 0.0392 | 0.0160 | 0.0606 | 0.0195 | 0.1172 |
| KIRC_grade_binary | Random Forest | MLP | 7 | 3 | 0 | 0.0402 | 0.0040 | 0.0780 | 0.1055 | 0.5273 |
| PRAD_pathologic_T_stage | Random Forest | HistGradientBoosting | 7 | 3 | 0 | 0.0021 | -0.0440 | 0.0373 | 0.4316 | 1.0000 |
| PRAD_pathologic_T_stage | Random Forest | ExtraTrees | 7 | 3 | 0 | 0.0057 | -0.0319 | 0.0381 | 0.6250 | 1.0000 |
| PRAD_pathologic_T_stage | Random Forest | Liquid/CfC | 6 | 4 | 0 | 0.0097 | -0.0449 | 0.0524 | 0.4316 | 1.0000 |
| PRAD_pathologic_T_stage | Random Forest | small-Liquid/CfC | 7 | 3 | 0 | 0.0117 | -0.0491 | 0.0627 | 0.3750 | 1.0000 |
| PRAD_pathologic_T_stage | Random Forest | MLP | 7 | 3 | 0 | 0.0177 | -0.0425 | 0.0702 | 0.6250 | 1.0000 |
| PRAD_pathologic_T_stage | Random Forest | Logistic ElasticNet | 7 | 3 | 0 | 0.0531 | -0.0155 | 0.1108 | 0.2324 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | Logistic ElasticNet | 7 | 3 | 0 | 0.0053 | -0.0387 | 0.0459 | 0.6953 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | Liquid/CfC | 5 | 4 | 1 | 0.0059 | -0.0304 | 0.0419 | 0.8203 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | Random Forest | 7 | 3 | 0 | 0.0069 | -0.0221 | 0.0290 | 0.1934 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | MLP | 7 | 3 | 0 | 0.0146 | -0.0252 | 0.0578 | 0.6250 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | HistGradientBoosting | 6 | 4 | 0 | 0.0209 | -0.0205 | 0.0573 | 0.3223 | 1.0000 |
| UCEC_molecular_subtype | ExtraTrees | small-Liquid/CfC | 5 | 4 | 1 | 0.0222 | -0.0198 | 0.0656 | 0.3594 | 1.0000 |

Interpretation: with 10 folds per task, top-vs-runner-up differences are small, fold-win patterns are mixed, and all top-vs-runner-up bootstrap confidence intervals cross zero. Top-vs-all comparisons may identify isolated inferior comparators, but they do not support a universal architecture-level superiority claim.

## Noise Elbow for Best CV Model per Task

| task | model | clean_f1 | f1_at_sigma_0_20 | f1_at_sigma_0_50 | first_5pct_drop_sigma | elbow_sigma |
| --- | --- | --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | ExtraTrees | 0.8676 | 0.8405 | 0.6821 | 0.3000 | 0.3000 |
| HNSC_HPV_status | Logistic ElasticNet | 0.9469 | 0.9433 | 0.9345 | nan | 0.1000 |
| KIRC_grade_binary | Random Forest | 0.6784 | 0.5823 | 0.4214 | 0.1000 | 0.0500 |
| PRAD_pathologic_T_stage | Random Forest | 0.6839 | 0.3812 | 0.3812 | 0.0500 | 0.2000 |
| UCEC_molecular_subtype | ExtraTrees | 0.8944 | 0.8978 | 0.8697 | nan | 0.3000 |

Noise is added after training to standardized test features. `first_5pct_drop_sigma` is the first Gaussian noise SD at which macro F1 drops by at least 5% relative to the clean test set. `elbow_sigma` is the maximum-distance knee point of the clean-to-high-noise curve.

## Files

- Raw metrics: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_cv_noise_raw_metrics.csv`
- CV summary: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_cv_summary.csv`
- Significance: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_top_model_significance.csv`
- Significance with fold wins: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_top_model_significance_with_wins.csv`
- Top-vs-all significance: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_top_vs_all_pairwise_significance.csv`
- Noise summary: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_noise_robustness_summary.csv`
- Noise elbow: `work\data\reviewer_tenfold_cv_noise_significance\processed\tenfold_noise_elbow_summary.csv`
