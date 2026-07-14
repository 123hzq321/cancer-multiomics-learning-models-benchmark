from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    MODALITIES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    ModalityLiquidCfC,
    MultiOmicsPreprocessor,
    compute_metrics,
    make_loaders,
    make_processed_splits,
    markdown_table,
    set_seed,
    train_torch_model,
)


ORDER_CONFIGS = {
    "liquid_original": ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"],
    "liquid_expression_last": ["gistic", "log2cna", "methylation", "rppa", "mutation", "mrna"],
    "liquid_reverse": ["mutation", "rppa", "methylation", "log2cna", "gistic", "mrna"],
}


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_labeled_table(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], LabelEncoder]:
    data = pd.read_csv(path)
    data = data[data["SUBTYPE"].notna()].copy()
    data = data[~data["SUBTYPE"].astype(str).str.contains("NA|Unknown|Not", case=False, regex=True)].copy()
    counts = data["SUBTYPE"].value_counts()
    data = data[data["SUBTYPE"].isin(counts[counts >= 10].index)].copy()
    modality_cols = {
        modality: [col for col in data.columns if col.startswith(f"{modality}__")]
        for modality in MODALITIES
    }
    label_encoder = LabelEncoder()
    data["target"] = label_encoder.fit_transform(data["SUBTYPE"].astype(str))
    return data.reset_index(drop=True), modality_cols, label_encoder


def assign_split(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    train_idx, temp_idx = train_test_split(
        data.index,
        train_size=0.70,
        random_state=seed,
        stratify=data["target"],
    )
    temp = data.loc[temp_idx]
    valid_idx, test_idx = train_test_split(
        temp.index,
        train_size=0.50,
        random_state=seed,
        stratify=temp["target"],
    )
    split_data = data.copy()
    split_data["split"] = "unused"
    split_data.loc[train_idx, "split"] = "train"
    split_data.loc[valid_idx, "split"] = "valid"
    split_data.loc[test_idx, "split"] = "test"
    return split_data


def select_modalities(modality_cols: dict[str, list[str]], order: list[str]) -> dict[str, list[str]]:
    return {modality: modality_cols[modality] for modality in order}


def prepare(split_data: pd.DataFrame, selected_cols: dict[str, list[str]]) -> dict[str, dict[str, object]]:
    preprocessor = MultiOmicsPreprocessor(selected_cols).fit(split_data[split_data["split"].eq("train")])
    return make_processed_splits(split_data, preprocessor)


def fit_logistic(processed: dict[str, dict[str, object]], seed: int) -> dict[str, dict[str, float]]:
    model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed)
    model.fit(processed["train"]["fusion"], processed["train"]["y"])
    metrics = {}
    for split_name, payload in processed.items():
        metrics[split_name] = compute_metrics(payload["y"], model.predict_proba(payload["fusion"]))
    return metrics


def rows_from_metrics(
    *,
    split_seed: int,
    model_seed: int,
    model: str,
    order_name: str,
    selected_modalities: list[str],
    metrics: dict[str, dict[str, float]],
    best_epoch: float | int | None = None,
) -> list[dict[str, object]]:
    rows = []
    for split_name, values in metrics.items():
        row = {
            "split_seed": split_seed,
            "model_seed": model_seed,
            "model": model,
            "order_name": order_name,
            "selected_modalities": "+".join(selected_modalities),
            "split": split_name,
            "best_epoch": best_epoch,
        }
        row.update(values)
        rows.append(row)
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy()
    summary = (
        test.groupby(["model", "order_name", "selected_modalities"])
        .agg(
            runs=("f1_macro", "size"),
            split_seeds=("split_seed", "nunique"),
            model_seeds=("model_seed", "nunique"),
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
    return summary.sort_values(["f1_macro_mean", "balanced_accuracy_mean"], ascending=False).reset_index(drop=True)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    data: pd.DataFrame,
    split_seeds: list[int],
    model_seeds: list[int],
) -> None:
    class_counts = data["SUBTYPE"].value_counts().rename_axis("subtype").reset_index(name="n")
    lines = [
        "# BRCA Liquid/CfC Order Validation v0",
        "",
        "## Scope",
        "",
        "Repeated random 70/15/15 stratified splits to validate whether the previously observed `mRNA-last` Liquid/CfC ordering is stable.",
        "",
        f"- Split seeds: `{','.join(map(str, split_seeds))}`",
        f"- Model seeds for Liquid/CfC: `{','.join(map(str, model_seeds))}`",
        "- Baseline: Logistic early fusion on all modalities.",
        "- Liquid/CfC orders: original, expression-last, reverse.",
        "",
        "## Class Counts",
        "",
        markdown_table(class_counts),
        "",
        "## Test Summary",
        "",
        markdown_table(summary, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "If `liquid_expression_last` remains above `liquid_original` across repeated splits, the order effect is more likely to be a stable modeling phenomenon. "
        "If it also exceeds Logistic early fusion on average, it becomes a publishable hypothesis, but still needs external or pan-cancer validation before being written as a general conclusion.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(PROCESSED_DIR / "brca_multimodal_impact468_table.csv"))
    parser.add_argument("--results-csv", default=str(PROCESSED_DIR / "brca_liquid_order_repeated_split_results.csv"))
    parser.add_argument("--summary-csv", default=str(PROCESSED_DIR / "brca_liquid_order_repeated_split_summary.csv"))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "brca_liquid_order_validation_report_v0.md"))
    parser.add_argument("--split-seeds", default="101,202,303,404,505")
    parser.add_argument("--model-seeds", default="42,43")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_seeds = parse_ints(args.split_seeds)
    model_seeds = parse_ints(args.model_seeds)
    data, modality_cols, label_encoder = load_labeled_table(Path(args.table))
    num_classes = len(label_encoder.classes_)

    all_rows = []
    for split_seed in split_seeds:
        print(f"Split seed {split_seed}")
        split_data = assign_split(data, split_seed)
        full_cols = select_modalities(modality_cols, list(MODALITIES))
        processed_full = prepare(split_data, full_cols)
        log_metrics = fit_logistic(processed_full, split_seed)
        all_rows.extend(
            rows_from_metrics(
                split_seed=split_seed,
                model_seed=split_seed,
                model="logistic_early_fusion",
                order_name="early_fusion",
                selected_modalities=list(MODALITIES),
                metrics=log_metrics,
            )
        )

        for order_name, order in ORDER_CONFIGS.items():
            selected_cols = select_modalities(modality_cols, order)
            processed = prepare(split_data, selected_cols)
            input_dims = [processed["train"]["modalities"][idx].shape[1] for idx in range(len(order))]
            for model_seed in model_seeds:
                set_seed(model_seed)
                print(f"  {order_name} model_seed={model_seed}")
                loaders = make_loaders(processed, args.batch_size)
                model = ModalityLiquidCfC(input_dims, num_classes)
                result = train_torch_model(
                    model,
                    loaders,
                    processed["train"]["y"],
                    num_classes,
                    device=torch.device(args.device),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    max_epochs=args.epochs,
                    patience=args.patience,
                )
                all_rows.extend(
                    rows_from_metrics(
                        split_seed=split_seed,
                        model_seed=model_seed,
                        model="liquid_cfc_modality_sequence",
                        order_name=order_name,
                        selected_modalities=order,
                        metrics=result["metrics"],
                        best_epoch=result["best_epoch"],
                    )
                )

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    write_report(Path(args.report), summary, data, split_seeds, model_seeds)
    print(f"Done. Report: {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
