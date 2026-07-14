from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from train_sequence_liquid_baselines import (
    LiquidClassifier,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    evaluate_model,
    markdown_table,
    set_seed,
)
from train_sequence_liquid_cv import (
    class_weights_from_train,
    make_cv_splits,
    make_loaders,
    make_sequence_bundle_for_splits,
    split_summary,
)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def validation_score(metrics: dict[str, float]) -> float:
    score = metrics.get("roc_auc", float("nan"))
    if math.isnan(score):
        score = metrics["balanced_accuracy"]
    return float(score)


def train_with_early_stopping(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    start_with_current_as_best: bool = False,
) -> tuple[nn.Module, dict[str, object]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_score = -math.inf
    best_epoch = -1
    stale = 0

    if start_with_current_as_best:
        current_val = evaluate_model(model, loaders["validation"], device)
        best_score = validation_score(current_val)
        best_state = clone_state(model)
        best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, dt, lengths, y in loaders["train"]:
            x = x.to(device)
            dt = dt.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, dt, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_metrics = evaluate_model(model, loaders["validation"], device)
        score = validation_score(val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "train": evaluate_model(model, loaders["train"], device),
        "validation": evaluate_model(model, loaders["validation"], device),
        "test": evaluate_model(model, loaders["test"], device),
    }


def linear_weight_parameters(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(module, "weight") for module in model.modules() if isinstance(module, nn.Linear)]


def apply_global_linear_pruning(model: nn.Module, amount: float) -> float:
    parameters = linear_weight_parameters(model)
    if not parameters:
        return 0.0
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    zero = 0
    total = 0
    for module, _ in parameters:
        mask = module.weight_mask.detach()
        zero += int(torch.sum(mask == 0).item())
        total += int(mask.numel())
    return float(zero / total) if total else 0.0


def remove_pruning_reparameterization(model: nn.Module) -> None:
    for module, _ in linear_weight_parameters(model):
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")


def result_rows(
    fold: int,
    seed: int,
    variant: str,
    sparsity: float,
    stage_result: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for split_name in ["train", "validation", "test"]:
        metrics = stage_result[split_name]
        rows.append(
            {
                "fold": int(fold),
                "seed": int(seed),
                "variant": variant,
                "split": split_name,
                "linear_sparsity": float(sparsity),
                "n": metrics["n"],
                "active_rate": metrics["active_rate"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "best_epoch": stage_result["best_epoch"],
                "best_validation_score": stage_result["best_validation_score"],
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"] == "test"].copy()
    return (
        test.groupby("variant")
        .agg(
            runs=("roc_auc", "size"),
            sparsity_mean=("linear_sparsity", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_sd=("average_precision", "std"),
        )
        .reset_index()
    )


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"] == "test"].copy()
    wide = test.pivot_table(index=["fold", "seed"], columns="variant", values=["balanced_accuracy", "f1_macro", "roc_auc", "average_precision"])
    base_variant = "liquid_base"
    rows = []
    for variant in sorted(test["variant"].unique()):
        if variant == base_variant:
            continue
        for metric in ["balanced_accuracy", "f1_macro", "roc_auc", "average_precision"]:
            diff = wide[(metric, variant)] - wide[(metric, base_variant)]
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean_delta_vs_base": float(diff.mean()),
                    "sd": float(diff.std()),
                    "n": int(diff.count()),
                    "variant_better_runs": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "base_better_runs": int((diff < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    splits: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = PROCESSED_DIR / "cv_sequence_liquid_prune_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "liquid_prune_finetune_results.csv", index=False)
    summary.to_csv(out_dir / "liquid_prune_finetune_summary.csv", index=False)
    deltas.to_csv(out_dir / "liquid_prune_finetune_paired_deltas.csv", index=False)
    splits.to_csv(out_dir / "liquid_prune_finetune_split_summary.csv", index=False)

    report = [
        "# Liquid/CfC Pruning and Fine-tuning CV v0",
        "",
        "## Setup",
        "",
        f"- Folds: {', '.join(str(f) for f in args.folds)}",
        f"- Seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- Selected omics features per fold: {args.features}",
        f"- Hidden dimension: {args.hidden}",
        f"- Base training epochs / patience: {args.epochs} / {args.patience}",
        f"- Fine-tuning epochs / patience: {args.finetune_epochs} / {args.finetune_patience}",
        f"- Base LR / fine-tune LR: {args.lr} / {args.finetune_lr}",
        f"- Pruning amounts: {', '.join(str(a) for a in args.prune_amounts)}",
        "- Pruning method: global L1 unstructured pruning on Linear layer weights inside Liquid/CfC.",
        "- Feature selection and scaling are fitted inside each training fold only.",
        "",
        "## Test Summary",
        "",
        markdown_table(summary),
        "",
        "## Paired Deltas vs Liquid Base",
        "",
        markdown_table(deltas),
        "",
        "## Split Summary",
        "",
        markdown_table(splits),
        "",
        "## Per-run Test Results",
        "",
        markdown_table(results[results["split"] == "test"].sort_values(["fold", "seed", "variant"])),
        "",
        "## Generated Files",
        "",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid_prune_finetune/liquid_prune_finetune_results.csv`",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid_prune_finetune/liquid_prune_finetune_summary.csv`",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid_prune_finetune/liquid_prune_finetune_paired_deltas.csv`",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid_prune_finetune/liquid_prune_finetune_split_summary.csv`",
    ]
    out = OUTPUTS_DIR / "liquid_prune_finetune_cv_report_v0.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Liquid/CfC fine-tuning and pruning on sequence CV folds.")
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--prune-amounts", type=parse_float_list, default=parse_float_list("0.2,0.4"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list("0,1,2,3,4"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    all_rows = []
    split_rows = []

    for fold in args.folds:
        print(f"Building fold {fold}")
        bundle = make_sequence_bundle_for_splits(make_cv_splits(fold), args.features, fold)
        split_rows.append(split_summary(bundle, fold))
        loaders = None
        input_dim = int(bundle.x.shape[-1])
        class_weights = class_weights_from_train(bundle)

        for seed in args.seeds:
            print(f"fold={fold} seed={seed} train liquid_base")
            set_seed(seed)
            loaders = make_loaders(bundle, args.batch_size, seed)
            base_model = LiquidClassifier(input_dim=input_dim, hidden_dim=args.hidden)
            base_model, base_result = train_with_early_stopping(
                base_model,
                loaders,
                class_weights,
                device,
                args.epochs,
                args.patience,
                args.lr,
            )
            base_state = clone_state(base_model)
            all_rows.extend(result_rows(fold, seed, "liquid_base", 0.0, base_result))

            print(f"fold={fold} seed={seed} fine_tune_only")
            ft_model = LiquidClassifier(input_dim=input_dim, hidden_dim=args.hidden)
            ft_model.load_state_dict(deepcopy(base_state))
            ft_model, ft_result = train_with_early_stopping(
                ft_model,
                loaders,
                class_weights,
                device,
                args.finetune_epochs,
                args.finetune_patience,
                args.finetune_lr,
                start_with_current_as_best=True,
            )
            all_rows.extend(result_rows(fold, seed, "liquid_finetune_only", 0.0, ft_result))

            for amount in args.prune_amounts:
                variant = f"liquid_prune{int(round(amount * 100)):02d}_finetune"
                print(f"fold={fold} seed={seed} {variant}")
                pruned_model = LiquidClassifier(input_dim=input_dim, hidden_dim=args.hidden)
                pruned_model.load_state_dict(deepcopy(base_state))
                pruned_model = pruned_model.to(device)
                sparsity = apply_global_linear_pruning(pruned_model, amount)
                pruned_model, pruned_result = train_with_early_stopping(
                    pruned_model,
                    loaders,
                    class_weights,
                    device,
                    args.finetune_epochs,
                    args.finetune_patience,
                    args.finetune_lr,
                    start_with_current_as_best=True,
                )
                remove_pruning_reparameterization(pruned_model)
                all_rows.extend(result_rows(fold, seed, variant, sparsity, pruned_result))

    results = pd.DataFrame(all_rows)
    split_df = pd.concat(split_rows, ignore_index=True)
    summary = summarize(results)
    deltas = paired_deltas(results)
    write_report(results, summary, deltas, split_df, args)


if __name__ == "__main__":
    main()
