# Phase 17：多癌种内部任务 benchmark v2

## 为什么补这一轮

之前 BRCA 做得比较深，但确实只是一个癌种。为了避免 benchmark 过窄，本阶段新增 5 个癌种内部任务，不再做“癌种分类”这种过于简单的任务，而是在每个癌种内部预测有医学意义的分子/临床终点。

## 新增任务

| 任务 | 癌种 | 预测目标 | 类别数/分布 |
| --- | --- | --- | --- |
| UCEC molecular subtype | 子宫内膜癌 | CN-high / CN-low / MSI / POLE | 163 / 147 / 148 / 49 |
| COADREAD molecular subtype | 结直肠癌 | CIN / MSI / GS | 328 / 63 / 58 |
| HNSC HPV status | 头颈癌 | HPV+ / HPV- | 72 / 415 |
| KIRC grade | 肾透明细胞癌 | G1/G2 vs G3/G4 | 227 / 277 |
| PRAD pathologic T stage | 前列腺癌 | T2 vs T3/T4 | 187 / 300 |

输入仍然是 TCGA PanCancer Atlas 多组学：

- mRNA
- GISTIC CNA
- log2 CNA
- methylation
- RPPA
- mutation

模型：

- Logistic ElasticNet
- ExtraTrees
- HistGradientBoosting
- MLP
- Liquid/CfC
- small-Liquid/CfC

每个任务做 70/15/15 stratified train/valid/test，3 个随机种子。

## 最佳模型结果

| 任务 | 最佳模型 | Macro F1 | Macro ROC-AUC |
| --- | --- | ---: | ---: |
| HNSC HPV status | ExtraTrees | 0.9505 | 0.9942 |
| UCEC molecular subtype | Logistic ElasticNet | 0.9134 | 0.9831 |
| COADREAD molecular subtype | ExtraTrees | 0.9060 | 0.9546 |
| PRAD pathologic T stage | small-Liquid/CfC | 0.7071 | 0.7717 |
| KIRC grade | HistGradientBoosting | 0.6996 | 0.7556 |

## Liquid/CfC 表现

Liquid/CfC 不是所有任务最强，但表现有任务依赖性：

- UCEC subtype：Liquid/CfC Macro F1 0.8895，接近 ElasticNet / ExtraTrees。
- COADREAD subtype：small-Liquid/CfC Macro F1 0.8756，高于原始 Liquid 0.8577，但低于 ExtraTrees。
- HNSC HPV：Liquid/CfC Macro F1 0.9015，低于 ExtraTrees / ElasticNet，但仍较强。
- KIRC grade：Liquid/CfC Macro F1 0.6858，接近最佳 HistGradientBoosting 0.6996。
- PRAD T-stage：small-Liquid/CfC Macro F1 0.7071，是该任务最佳模型。

## 关键结论

这轮结果把论文从“单癌种 BRCA 研究”升级成了：

> BRCA 深度外部验证 + 多癌种内部任务 benchmark。

更重要的是，结论更稳：

1. Liquid/CfC 不是 universal winner。
2. 不同癌种、不同终点的最佳模型不同。
3. 传统强 baseline 仍然非常重要。
4. small-Liquid 在 PRAD T-stage 中表现最好，说明低容量 Liquid 在某些较难临床任务中可能有价值。
5. KIRC grade 和 PRAD T-stage 明显比 UCEC/HNSC/COADREAD 分子任务更难，说明 benchmark 不是全都接近饱和。

## 论文里最合适的写法

可以写：

> To address the limitation of a BRCA-only benchmark, we further constructed a five-task multi-cancer internal benchmark from TCGA PanCancer Atlas cohorts, including UCEC molecular subtype, COADREAD molecular subtype, HNSC HPV status, KIRC grade, and PRAD pathologic T stage. Across these tasks, no model dominated universally. Tree-based and linear baselines were strongest for several molecular subtype tasks, whereas small-Liquid/CfC achieved the best macro F1 for PRAD pathologic T-stage prediction. These results support the central conclusion that Liquid/CfC is a context-dependent cross-omics fusion candidate rather than a universally superior model.

中文：

> 为避免 BRCA 单癌种 benchmark 的局限，我们进一步构建了 5 个 TCGA PanCancer 癌种内部任务。结果显示，不同癌种和不同终点的最佳模型不同，Liquid/CfC 并非普遍最优，但在部分较难临床任务中具有竞争力。

## 生成文件

- 报告：`outputs/multicancer_internal_benchmark_report_v0.md`
- 中文总结：`outputs/phase17_multicancer_internal_benchmark_summary_zh_v0.md`
- 汇总表：`work/data/multicancer_internal_benchmark/processed/multicancer_internal_benchmark_summary.csv`
- 最佳模型表：`work/data/multicancer_internal_benchmark/processed/multicancer_internal_best_by_task.csv`
- 标签计数：`work/data/multicancer_internal_benchmark/processed/multicancer_internal_task_label_counts.csv`
- 训练脚本：`work/scripts/train_multicancer_internal_benchmark.py`
- 图：
  - `outputs/multicancer_internal_best_by_task.png`
  - `outputs/multicancer_internal_model_task_heatmap.png`
