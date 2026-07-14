# Phase 15：METABRIC 加入训练 + 新外部验证集 + small-Liquid

## 做了什么

本阶段完成两件事：

1. 数据层面：把 METABRIC 从外部验证集改为加入训练/验证池。
2. 模型层面：加入 small-Liquid/CfC，并与原始 Liquid/CfC 对比。

新的训练/验证设计：

- 训练池：TCGA BRCA + METABRIC，共 2737 个样本。
- 重新按 85/15 分为 train/valid：
  - train：2326
  - valid：411
- 新外部验证集：
  - SMC 2018：168 个样本，mRNA marker 外部验证。
  - CPTAC 2020：122 个样本，mRNA + CNA + mutation marker 多组学外部验证。
  - GSE96058 / SCAN-B：3137 个样本，mRNA marker 外部验证。

## small-Liquid 参数量

small-Liquid 使用：

- `embed_dim = 32`
- `hidden_dim = 48`

原始 Liquid 使用：

- `embed_dim = 64`
- `hidden_dim = 96`

| 特征集 | 模型 | 可训练参数 |
| --- | --- | ---: |
| marker mRNA | MLP | 19,717 |
| marker mRNA | Liquid/CfC | 52,613 |
| marker mRNA | small-Liquid/CfC | 14,789 |
| marker multimodal | MLP | 40,965 |
| marker multimodal | Liquid/CfC | 63,749 |
| marker multimodal | small-Liquid/CfC | 20,357 |

small-Liquid 把参数量压低了约 68-72%。

## CPTAC 2020 外部验证

CPTAC 是本轮最适合看多组学 Liquid 的外部集，因为它有 mRNA、CNA、mutation。

| 排名 | 模型 | 特征集 | Macro F1 | Macro ROC-AUC |
| --- | --- | --- | ---: | ---: |
| 1 | Liquid/CfC | marker multimodal | 0.8806 | 0.9933 |
| 2 | MLP | marker mRNA | 0.8803 | 0.9945 |
| 3 | small-Liquid/CfC | marker multimodal | 0.8733 | 0.9927 |
| 4 | small-Liquid/CfC | marker mRNA | 0.8676 | 0.9942 |
| 5 | Logistic | marker mRNA | 0.8627 | 0.9878 |

解释：

- 加入 METABRIC 训练后，CPTAC 上多组学 Liquid/CfC 成为第一。
- small-Liquid 比原始 Liquid 参数少很多，但 CPTAC 上略低于原始 Liquid。
- mRNA-only MLP 几乎追平多组学 Liquid，说明表达信号仍非常强。

## SCAN-B / GSE96058 外部验证

SCAN-B 是最大的新外部集，但只有 mRNA marker 可以对齐。

| 排名 | 模型 | Macro F1 | Macro ROC-AUC |
| --- | --- | ---: | ---: |
| 1 | ExtraTrees | 0.8493 | 0.9831 |
| 2 | Liquid/CfC | 0.8172 | 0.9830 |
| 3 | MLP | 0.8033 | 0.9847 |
| 4 | small-Liquid/CfC | 0.7857 | 0.9818 |
| 5 | Logistic | 0.7851 | 0.9801 |

解释：

- 大样本 RNA-seq 外部验证中，ExtraTrees 最强。
- 原始 Liquid/CfC 高于 MLP、small-Liquid 和 Logistic。
- small-Liquid 没有超过原始 Liquid。

## SMC 2018 外部验证

SMC 也只能做 mRNA marker 外部验证。注意 Normal 类只有 2 个样本，因此宏平均指标需要谨慎看。

| 排名 | 模型 | Macro F1 | Macro ROC-AUC |
| --- | --- | ---: | ---: |
| 1 | ExtraTrees | 0.9124 | 0.9894 |
| 2 | Liquid/CfC | 0.8178 | 0.9854 |
| 3 | small-Liquid/CfC | 0.7948 | 0.9838 |
| 4 | MLP | 0.7799 | 0.9840 |
| 5 | Logistic | 0.7751 | 0.9858 |

解释：

- SMC 上 ExtraTrees 明显最好。
- Liquid/CfC 在神经网络里最好。
- small-Liquid 仍低于原始 Liquid。

## 结论有没有改变

有一部分改变，但不是彻底反转。

之前的结论是：

> BRCA subtype 外部泛化主要由 mRNA marker 驱动，Liquid/CfC 不是最强模型。

现在更新为：

> 在加入 METABRIC 训练后，Liquid/CfC 的外部表现明显增强；在具有 mRNA+CNA+mutation 的 CPTAC 小样本外部集上，多组学 Liquid/CfC 达到最佳 Macro F1。但在更大的 mRNA-only 外部队列 SCAN-B 和 SMC 上，树模型或表达模型仍更强。因此，Liquid/CfC 的价值更适合定位为跨组学融合结构，而不是所有外部队列上的统一最优模型。

## small-Liquid 的判断

small-Liquid 的优点：

- 参数量大幅下降。
- CPTAC 上接近原始 Liquid。
- 更适合小样本、低过拟合风险设定。

small-Liquid 的不足：

- 没有稳定超过原始 Liquid。
- 在 SCAN-B 和 SMC 上低于原始 Liquid。

所以论文里可以写：

> We further evaluated a reduced-capacity small-Liquid/CfC model. Although it reduced trainable parameters by approximately 68-72%, it did not consistently outperform the original Liquid/CfC model, suggesting that capacity reduction alone was insufficient to improve cross-cohort generalization.

## 现在更强的论文主线

这一步之后，论文主线可以更积极一点：

1. 原始 TCGA-only 训练时，Liquid/CfC 不占绝对优势。
2. 加入 METABRIC 后，训练样本增多，Liquid/CfC 外部表现改善。
3. 在 CPTAC 多组学外部集上，marker multimodal Liquid/CfC 排第一。
4. 但在大样本 RNA 外部集上，表达 marker 和树模型仍非常强。
5. 这说明 Liquid/CfC 的优势更可能出现在“真正有多组学可对齐”的外部场景，而不是单纯 mRNA 表达任务。

## 生成文件

- 主报告：`outputs/tcga_metabric_joined_small_liquid_external_validation_report_v0.md`
- 中文总结：`outputs/phase15_joined_metabric_small_liquid_external_summary_zh_v0.md`
- 汇总表：`work/data/joined_metabric_external_validation/processed/joined_small_liquid_external_model_summary.csv`
- 全结果：`work/data/joined_metabric_external_validation/processed/joined_small_liquid_external_model_results.csv`
- 参数表：`work/data/joined_metabric_external_validation/processed/joined_small_liquid_parameter_counts.csv`
- 训练脚本：`work/scripts/train_joined_metabric_small_liquid_external_validation.py`
- 图：
  - `outputs/joined_small_liquid_cptac_2020_external_comparison.png`
  - `outputs/joined_small_liquid_scanb_external_comparison.png`
  - `outputs/joined_small_liquid_smc_2018_external_comparison.png`
  - `outputs/joined_small_liquid_top_external_macro_f1.png`
