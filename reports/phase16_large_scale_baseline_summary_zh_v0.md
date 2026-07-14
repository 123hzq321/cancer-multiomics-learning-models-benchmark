# Phase 16：补充传统 baseline + 扩大 mRNA 训练规模

## 做了什么

本阶段补充了更完整的传统/经典 baseline，并测试“扩大训练规模”是否提升外部泛化。

训练规模分两组：

1. `joined_tcga_metabric`
   - 训练池：TCGA BRCA + METABRIC
   - train：2326
   - valid：411
   - 外部验证：CPTAC 2020

2. `expanded_tcga_metabric_scanb_smc`
   - 训练池：TCGA BRCA + METABRIC + SCAN-B + SMC 2018
   - 只用于 mRNA marker，因为 SCAN-B/SMC 缺少完整 CNA 模态
   - 外部验证仍为 CPTAC 2020

新增 baseline：

- Logistic L2
- Logistic ElasticNet
- Ridge Classifier
- Calibrated Linear SVM
- LDA shrinkage
- GaussianNB
- KNN distance
- RandomForest
- ExtraTrees
- HistGradientBoosting
- sklearn MLP

## CPTAC 外部验证：最强结果

| 排名 | 训练规模 | 模型 | 特征 | Macro F1 | Macro ROC-AUC |
| --- | --- | --- | --- | ---: | ---: |
| 1 | TCGA+METABRIC | Logistic ElasticNet | mRNA marker | 0.8830 | 0.9899 |
| 2 | TCGA+METABRIC | sklearn MLP | mRNA marker | 0.8823 | 0.9940 |
| 3 | TCGA+METABRIC | Liquid/CfC | marker multimodal | 0.8806 | 0.9933 |
| 4 | TCGA+METABRIC | MLP | mRNA marker | 0.8803 | 0.9945 |
| 5 | TCGA+METABRIC | small-Liquid/CfC | marker multimodal | 0.8733 | 0.9927 |
| 6 | expanded | sklearn MLP | mRNA marker | 0.8559 | 0.9940 |
| 7 | expanded | Logistic ElasticNet | mRNA marker | 0.8509 | 0.9911 |
| 8 | expanded | HistGradientBoosting | mRNA marker | 0.8443 | 0.9869 |

## 重要变化

之前 Phase 15 中，多组学 Liquid/CfC 在 CPTAC 上是最强之一：

- Macro F1：0.8806
- Macro ROC-AUC：0.9933

这次补充更强传统 baseline 后，`Logistic ElasticNet` 和 `sklearn MLP` 的 mRNA-only 结果略高：

- Logistic ElasticNet：Macro F1 0.8830
- sklearn MLP：Macro F1 0.8823

差距非常小，但写论文时必须如实表述：

> Liquid/CfC was highly competitive on CPTAC multimodal external validation, but it did not clearly outperform all strengthened mRNA-marker baselines after the extended baseline study.

中文就是：

> Liquid/CfC 在 CPTAC 多组学外部验证中表现非常接近最优，但在补充 ElasticNet Logistic 和 sklearn MLP 后，不能再声称它绝对优于所有 baseline。

## 扩大训练规模是否提升了？

没有整体提升。

把 SCAN-B 和 SMC 加入 mRNA marker 训练后：

- sklearn MLP：0.8823 -> 0.8559
- Logistic ElasticNet：0.8830 -> 0.8509
- Logistic L2：0.8627 -> 0.8312

只有部分模型如 HistGradientBoosting 从 0.8037 提升到 0.8443，但整体最强模型没有变强。

这说明：

> 更多样本不一定带来更好外部泛化。跨队列平台差异、标签分布差异和表达归一化差异可能抵消样本量收益。

## 对论文结论的影响

论文结论应该进一步变得诚实和稳健：

1. mRNA marker 是 BRCA subtype 预测的核心信号。
2. 加强 baseline 后，ElasticNet Logistic 和 MLP 仍然极强。
3. Liquid/CfC 的优势不是“绝对分类性能第一”，而是提供了一个跨组学状态更新结构。
4. 扩大训练规模不必然提升外部验证，跨队列域偏移是主要问题。
5. 更强的论文主线应是系统评估，而不是宣称 Liquid 全面胜出。

## 推荐写法

可以写进论文：

> We further expanded the baseline study to include elastic-net logistic regression, calibrated linear SVM, ridge classifier, shrinkage LDA, Gaussian Naive Bayes, k-nearest neighbors, random forest, ExtraTrees, histogram gradient boosting, and an sklearn MLP. Elastic-net logistic regression and MLP slightly exceeded multimodal Liquid/CfC on CPTAC external macro F1, although the difference was small. Expanding the mRNA training pool with SCAN-B and SMC did not improve the best CPTAC external performance, indicating that cross-cohort domain shift can offset gains from larger sample size.

## 生成文件

- 报告：`outputs/large_scale_mrna_baseline_report_v0.md`
- 中文总结：`outputs/phase16_large_scale_baseline_summary_zh_v0.md`
- 汇总表：`work/data/large_scale_baselines/processed/large_scale_mrna_baseline_summary.csv`
- 全结果：`work/data/large_scale_baselines/processed/large_scale_mrna_baseline_results.csv`
- 训练规模计数：`work/data/large_scale_baselines/processed/large_scale_mrna_training_counts.csv`
- 综合 CPTAC 对比：`work/data/large_scale_baselines/processed/integrated_cptac_external_comparison.csv`
- 图：
  - `outputs/large_scale_mrna_baseline_cptac_f1.png`
  - `outputs/large_scale_mrna_training_delta_cptac_f1.png`
- 脚本：`work/scripts/train_large_scale_mrna_baselines.py`
