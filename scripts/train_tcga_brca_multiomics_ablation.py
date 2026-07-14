from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    MODALITIES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    EarlyFusionMLP,
    ModalityLiquidCfC,
    ModalityTCN,
    MultiOmicsPreprocessor,
    compute_metrics,
    load_data,
    make_loaders,
    make_processed_splits,
    markdown_table,
    set_seed,
    train_torch_model,
)


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def subset_modality_cols(
    modality_cols: dict[str, list[str]],
    selected_modalities: list[str],
) -> dict[str, list[str]]:
    return {modality: modality_cols[modality] for modality in selected_modalities}


def prepare_experiment(
    data: pd.DataFrame,
    modality_cols: dict[str, list[str]],
    selected_modalities: list[str],
) -> tuple[dict[str, dict[str, object]], list[int], int]:
    selected_cols = subset_modality_cols(modality_cols, selected_modalities)
    train_df = data[data["split"].eq("train")].copy()
    preprocessor = MultiOmicsPreprocessor(selected_cols).fit(train_df)
    processed = make_processed_splits(data, preprocessor)
    input_dims = [processed["train"]["modalities"][idx].shape[1] for idx in range(len(selected_modalities))]
    fusion_dim = processed["train"]["fusion"].shape[1]
    return processed, input_dims, fusion_dim


def fit_logistic(processed: dict[str, dict[str, object]], seed: int) -> dict[str, object]:
    model = LogisticRegression(
        max_iter=5000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(processed["train"]["fusion"], processed["train"]["y"])
    metrics = {}
    for split_name, payload in processed.items():
        proba = model.predict_proba(payload["fusion"])
        metrics[split_name] = compute_metrics(payload["y"], proba)
    return {
        "metrics": metrics,
        "best_epoch": np.nan,
        "best_validation_score": metrics["valid"]["roc_auc_ovr_macro"],
    }


def rows_from_result(
    *,
    experiment: str,
    ablation_type: str,
    selected_modalities: list[str],
    seed: int,
    model_name: str,
    model_group: str,
    result: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for split_name, metrics in result["metrics"].items():
        row = {
            "experiment": experiment,
            "ablation_type": ablation_type,
            "selected_modalities": "+".join(selected_modalities),
            "n_modalities": len(selected_modalities),
            "seed": seed,
            "model": model_name,
            "model_group": model_group,
            "split": split_name,
            "best_epoch": result.get("best_epoch", np.nan),
            "best_validation_score": result.get("best_validation_score", np.nan),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy()
    summary = (
        test.groupby(["experiment", "ablation_type", "selected_modalities", "n_modalities", "model", "model_group"])
        .agg(
            runs=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            roc_auc_ovr_macro_mean=("roc_auc_ovr_macro", "mean"),
            roc_auc_ovr_macro_sd=("roc_auc_ovr_macro", "std"),
        )
        .reset_index()
    )
    return summary.sort_values(["ablation_type", "f1_macro_mean"], ascending=[True, False]).reset_index(drop=True)


def make_experiments() -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []
    full = list(MODALITIES)

    experiments.append(
        {
            "experiment": "full_logistic_reference",
            "ablation_type": "logistic_feature_contribution",
            "modalities": full,
            "models": ["logistic"],
        }
    )
    for modality in full:
        experiments.append(
            {
                "experiment": f"single_{modality}",
                "ablation_type": "single_modality",
                "modalities": [modality],
                "models": ["logistic"],
            }
        )
    for modality in full:
        experiments.append(
            {
                "experiment": f"drop_{modality}",
                "ablation_type": "leave_one_modality_out",
                "modalities": [item for item in full if item != modality],
                "models": ["logistic"],
            }
        )

    no_mrna = [item for item in full if item != "mrna"]
    experiments.append(
        {
            "experiment": "no_mrna_neural_comparison",
            "ablation_type": "weak_modality_fusion",
            "modalities": no_mrna,
            "models": ["mlp", "tcn", "liquid"],
        }
    )

    order_configs = {
        "liquid_order_original_full": full,
        "liquid_order_reverse_full": list(reversed(full)),
        "liquid_order_genome_to_expression": ["mutation", "gistic", "log2cna", "methylation", "mrna", "rppa"],
        "liquid_order_expression_last": ["gistic", "log2cna", "methylation", "rppa", "mutation", "mrna"],
        "liquid_order_random_fixed": ["methylation", "mutation", "mrna", "rppa", "gistic", "log2cna"],
        "liquid_order_no_mrna_original": no_mrna,
        "liquid_order_no_mrna_reverse": list(reversed(no_mrna)),
    }
    for experiment, modalities in order_configs.items():
        experiments.append(
            {
                "experiment": experiment,
                "ablation_type": "liquid_order_sensitivity",
                "modalities": modalities,
                "models": ["liquid"],
            }
        )
    return experiments


def run_model(
    model_key: str,
    processed: dict[str, dict[str, object]],
    input_dims: list[int],
    fusion_dim: int,
    num_classes: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[str, str, dict[str, object]]:
    if model_key == "logistic":
        return "logistic_early_fusion", "linear_baseline", fit_logistic(processed, seed)

    loaders = make_loaders(processed, args.batch_size)
    y_train = processed["train"]["y"]
    if model_key == "mlp":
        model_name = "mlp_early_fusion"
        model_group = "static_nn"
        model = EarlyFusionMLP(fusion_dim, num_classes)
    elif model_key == "tcn":
        model_name = "tcn_modality_sequence"
        model_group = "sequence_nn"
        model = ModalityTCN(input_dims, num_classes)
    elif model_key == "liquid":
        model_name = "liquid_cfc_modality_sequence"
        model_group = "liquid_nn"
        model = ModalityLiquidCfC(input_dims, num_classes)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    result = train_torch_model(
        model,
        loaders,
        y_train,
        num_classes,
        device=torch.device(args.device),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
    )
    return model_name, model_group, result


def build_delta_tables(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    logistic = summary[summary["model"].eq("logistic_early_fusion")].copy()
    full = logistic[logistic["experiment"].eq("full_logistic_reference")]
    if not full.empty:
        full_f1 = float(full.iloc[0]["f1_macro_mean"])
        full_auc = float(full.iloc[0]["roc_auc_ovr_macro_mean"])
        leave_one = logistic[logistic["ablation_type"].eq("leave_one_modality_out")].copy()
        leave_one["dropped_modality"] = leave_one["experiment"].str.replace("drop_", "", regex=False)
        leave_one["delta_f1_vs_full"] = leave_one["f1_macro_mean"] - full_f1
        leave_one["delta_auc_vs_full"] = leave_one["roc_auc_ovr_macro_mean"] - full_auc
        tables["leave_one_logistic_delta"] = leave_one[
            [
                "dropped_modality",
                "f1_macro_mean",
                "delta_f1_vs_full",
                "roc_auc_ovr_macro_mean",
                "delta_auc_vs_full",
                "balanced_accuracy_mean",
            ]
        ].sort_values("delta_f1_vs_full")

        single = logistic[logistic["ablation_type"].eq("single_modality")].copy()
        single["modality"] = single["experiment"].str.replace("single_", "", regex=False)
        tables["single_modality_logistic"] = single[
            [
                "modality",
                "f1_macro_mean",
                "roc_auc_ovr_macro_mean",
                "balanced_accuracy_mean",
                "accuracy_mean",
            ]
        ].sort_values("f1_macro_mean", ascending=False)

    liquid_order = summary[
        (summary["ablation_type"].eq("liquid_order_sensitivity"))
        & (summary["model"].eq("liquid_cfc_modality_sequence"))
    ].copy()
    if not liquid_order.empty:
        tables["liquid_order_sensitivity"] = liquid_order[
            [
                "experiment",
                "selected_modalities",
                "f1_macro_mean",
                "f1_macro_sd",
                "balanced_accuracy_mean",
                "roc_auc_ovr_macro_mean",
            ]
        ].sort_values("f1_macro_mean", ascending=False)

    weak = summary[summary["ablation_type"].eq("weak_modality_fusion")].copy()
    if not weak.empty:
        tables["weak_modality_no_mrna"] = weak[
            [
                "model",
                "model_group",
                "f1_macro_mean",
                "f1_macro_sd",
                "balanced_accuracy_mean",
                "roc_auc_ovr_macro_mean",
            ]
        ].sort_values("f1_macro_mean", ascending=False)

    return tables


def write_report(
    out_path: Path,
    summary: pd.DataFrame,
    delta_tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# TCGA BRCA Multi-omics Ablation v0",
        "",
        "## Scope",
        "",
        "Ablation experiments for TCGA BRCA subtype prediction after the initial baseline-vs-Liquid run.",
        "",
        "- Single-modality Logistic: estimates how predictive each omics layer is alone.",
        "- Leave-one-modality-out Logistic: estimates the marginal loss after removing each layer.",
        "- No-mRNA neural comparison: tests whether Liquid/CfC helps when the dominant expression layer is removed.",
        "- Liquid/CfC order sensitivity: tests whether cross-modality sequence order changes performance.",
        "",
        "## Single-Modality Logistic",
        "",
        markdown_table(delta_tables.get("single_modality_logistic", pd.DataFrame()), floatfmt=".4f"),
        "",
        "## Leave-One-Modality-Out Logistic Delta",
        "",
        "Negative delta means performance dropped after removing that modality.",
        "",
        markdown_table(delta_tables.get("leave_one_logistic_delta", pd.DataFrame()), floatfmt=".4f"),
        "",
        "## No-mRNA Neural Comparison",
        "",
        markdown_table(delta_tables.get("weak_modality_no_mrna", pd.DataFrame()), floatfmt=".4f"),
        "",
        "## Liquid/CfC Order Sensitivity",
        "",
        markdown_table(delta_tables.get("liquid_order_sensitivity", pd.DataFrame()), floatfmt=".4f"),
        "",
        "## Full Summary",
        "",
        markdown_table(summary, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "These ablations are intended to separate feature contribution from architecture contribution. "
        "If mRNA alone is already strong and removing mRNA causes the largest drop, then the model is primarily "
        "using expression biology rather than discovering a unique liquid-network advantage. "
        "If Liquid/CfC changes substantially across modality orders, the ordering should be treated as a modeling "
        "choice requiring validation rather than a biologically fixed sequence.",
        "",
        "## Reproduction",
        "",
        f"- Seeds: `{args.seeds}`",
        f"- Epochs: `{args.epochs}`",
        f"- Patience: `{args.patience}`",
        f"- Device: `{args.device}`",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_chinese_summary(out_path: Path, delta_tables: dict[str, pd.DataFrame]) -> None:
    single = delta_tables.get("single_modality_logistic", pd.DataFrame())
    leave_one = delta_tables.get("leave_one_logistic_delta", pd.DataFrame())
    weak = delta_tables.get("weak_modality_no_mrna", pd.DataFrame())
    order = delta_tables.get("liquid_order_sensitivity", pd.DataFrame())

    top_single = single.iloc[0].to_dict() if not single.empty else {}
    biggest_drop = leave_one.iloc[0].to_dict() if not leave_one.empty else {}
    best_weak = weak.iloc[0].to_dict() if not weak.empty else {}
    best_order = order.iloc[0].to_dict() if not order.empty else {}

    lines = [
        "# Phase 10：TCGA BRCA 多组学 ablation 总结",
        "",
        "## 本阶段做了什么",
        "",
        "在上一阶段 BRCA subtype 预测中，Logistic early fusion 强于 Liquid/CfC。"
        "本阶段继续做 ablation，判断性能主要来自哪个组学模态，以及 Liquid/CfC 是否真的受益于跨模态状态更新。",
        "",
        "## 主要发现",
        "",
    ]

    if top_single:
        lines.extend(
            [
                "### 1. 单模态中最强的是：`{}`".format(top_single["modality"]),
                "",
                "- 单模态 macro F1 = {:.4f}".format(float(top_single["f1_macro_mean"])),
                "- 单模态 macro ROC-AUC = {:.4f}".format(float(top_single["roc_auc_ovr_macro_mean"])),
                "",
            ]
        )

    if biggest_drop:
        lines.extend(
            [
                "### 2. 去掉后损失最大的模态是：`{}`".format(biggest_drop["dropped_modality"]),
                "",
                "- 去掉该模态后的 macro F1 = {:.4f}".format(float(biggest_drop["f1_macro_mean"])),
                "- 相对全模态 Logistic 的 macro F1 变化 = {:.4f}".format(float(biggest_drop["delta_f1_vs_full"])),
                "",
            ]
        )

    if best_weak:
        lines.extend(
            [
                "### 3. 去掉 mRNA 后，弱模态融合中最强模型是：`{}`".format(best_weak["model"]),
                "",
                "- macro F1 = {:.4f}".format(float(best_weak["f1_macro_mean"])),
                "- balanced accuracy = {:.4f}".format(float(best_weak["balanced_accuracy_mean"])),
                "- macro ROC-AUC = {:.4f}".format(float(best_weak["roc_auc_ovr_macro_mean"])),
                "",
            ]
        )

    if best_order:
        lines.extend(
            [
                "### 4. Liquid/CfC 最优模态顺序实验是：`{}`".format(best_order["experiment"]),
                "",
                "- 顺序：`{}`".format(best_order["selected_modalities"]),
                "- macro F1 = {:.4f}".format(float(best_order["f1_macro_mean"])),
                "- macro ROC-AUC = {:.4f}".format(float(best_order["roc_auc_ovr_macro_mean"])),
                "",
            ]
        )

    lines.extend(
        [
            "## 单模态 Logistic 排名",
            "",
            markdown_table(single, floatfmt=".4f"),
            "",
            "## Leave-one-out Logistic 结果",
            "",
            markdown_table(leave_one, floatfmt=".4f"),
            "",
            "## 去掉 mRNA 后的神经网络对比",
            "",
            markdown_table(weak, floatfmt=".4f"),
            "",
            "## Liquid/CfC 顺序敏感性",
            "",
            markdown_table(order, floatfmt=".4f"),
            "",
            "## 论文解释",
            "",
            "如果单模态 mRNA 或去掉 mRNA 的结果显示性能大幅变化，那么 BRCA subtype 预测主要由表达层驱动。"
            "这会削弱“液态神经网络带来普遍优势”的说法，但会让论文更严谨：我们可以把 Liquid/CfC 定位为一种跨组学融合候选结构，"
            "并展示它在某些弱模态或特定顺序下的表现，而不是夸大它的优势。",
            "",
            "## 输出文件",
            "",
            "- `outputs/tcga_brca_multiomics_ablation_report_v0.md`",
            "- `outputs/phase10_tcga_brca_ablation_summary_zh_v0.md`",
            "- `work/data/tcga_brca_cbioportal/processed/brca_subtype_ablation_results.csv`",
            "- `work/data/tcga_brca_cbioportal/processed/brca_subtype_ablation_summary.csv`",
            "- `work/data/tcga_brca_cbioportal/processed/brca_subtype_ablation_delta_tables.json`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(PROCESSED_DIR / "brca_multimodal_impact468_table.csv"))
    parser.add_argument("--splits", default=str(PROCESSED_DIR / "brca_subtype_splits_70_15_15.csv"))
    parser.add_argument("--results-csv", default=str(PROCESSED_DIR / "brca_subtype_ablation_results.csv"))
    parser.add_argument("--summary-csv", default=str(PROCESSED_DIR / "brca_subtype_ablation_summary.csv"))
    parser.add_argument("--delta-json", default=str(PROCESSED_DIR / "brca_subtype_ablation_delta_tables.json"))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "tcga_brca_multiomics_ablation_report_v0.md"))
    parser.add_argument("--zh-report", default=str(OUTPUTS_DIR / "phase10_tcga_brca_ablation_summary_zh_v0.md"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data, modality_cols, label_encoder = load_data(Path(args.table), Path(args.splits))
    num_classes = len(label_encoder.classes_)
    experiments = make_experiments()
    all_rows: list[dict[str, object]] = []

    for experiment in experiments:
        selected_modalities = list(experiment["modalities"])
        print(f"Preparing {experiment['experiment']}: {selected_modalities}")
        processed, input_dims, fusion_dim = prepare_experiment(data, modality_cols, selected_modalities)
        for seed in seeds:
            set_seed(seed)
            for model_key in experiment["models"]:
                print(f"  seed={seed} model={model_key}")
                model_name, model_group, result = run_model(
                    model_key,
                    processed,
                    input_dims,
                    fusion_dim,
                    num_classes,
                    seed,
                    args,
                )
                all_rows.extend(
                    rows_from_result(
                        experiment=str(experiment["experiment"]),
                        ablation_type=str(experiment["ablation_type"]),
                        selected_modalities=selected_modalities,
                        seed=seed,
                        model_name=model_name,
                        model_group=model_group,
                        result=result,
                    )
                )

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    delta_tables = build_delta_tables(summary)
    json_payload = {name: table.to_dict(orient="records") for name, table in delta_tables.items()}
    Path(args.delta_json).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_report(Path(args.report), summary, delta_tables, args)
    write_chinese_summary(Path(args.zh_report), delta_tables)
    print(f"Done. Report: {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
