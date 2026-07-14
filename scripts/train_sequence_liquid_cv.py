from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from train_sequence_liquid_baselines import (
    GRUClassifier,
    INT_TO_LABEL,
    LABEL_TO_INT,
    LiquidClassifier,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    ROOT,
    SequenceBundle,
    SequenceDataset,
    build_examples,
    choose_features,
    collate_batch,
    evaluate_model,
    load_visit_table,
    markdown_table,
    set_seed,
)


def make_cv_splits(fold: int) -> pd.DataFrame:
    fold_df = pd.read_csv(PROCESSED_DIR / "splits_5fold_by_participant.csv")
    validation_fold = (fold + 1) % 5
    split = np.where(
        fold_df["fold"] == fold,
        "test",
        np.where(fold_df["fold"] == validation_fold, "validation", "train"),
    )
    return pd.DataFrame({"participant_id": fold_df["participant_id"], "split": split})


def make_sequence_bundle_for_splits(splits: pd.DataFrame, k_features: int, fold: int) -> SequenceBundle:
    df = load_visit_table()
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("mgx_path__") or c.startswith("mtx_path__") or c in ["has_mgx_path", "has_mtx_path"]
    ]
    examples = build_examples(df, splits)
    train_examples = examples[examples["split"] == "train"]
    selected_features = choose_features(df, train_examples, feature_cols, k_features)

    train_participants = set(splits.loc[splits["split"] == "train", "participant_id"])
    train_visits = df[df["participant_id"].isin(train_participants)]
    train_matrix = np.log1p(train_visits[selected_features].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))
    scaler = StandardScaler()
    scaler.fit(train_matrix)

    max_len = int(df.groupby("participant_id").size().max())
    n_examples = len(examples)
    x = np.zeros((n_examples, max_len, len(selected_features) + 2), dtype=np.float32)
    dt = np.zeros((n_examples, max_len), dtype=np.float32)
    lengths = np.zeros(n_examples, dtype=np.int64)
    y = np.zeros(n_examples, dtype=np.int64)
    split = examples["split"].fillna("unknown").to_numpy(dtype=object)
    participant_id = examples["participant_id"].to_numpy(dtype=object)
    current_week = examples["week_num_num"].to_numpy(dtype=np.float32)
    next_week = examples["next_week_num"].to_numpy(dtype=np.float32)

    grouped = {pid: g.sort_values(["week_num_num", "visit_num_num"]).reset_index(drop=True) for pid, g in df.groupby("participant_id")}
    for row_idx, (_, ex) in enumerate(examples.iterrows()):
        pid = ex["participant_id"]
        current_week_value = ex["week_num_num"]
        patient = grouped[pid]
        current_positions = np.where(patient["week_num_num"].to_numpy() == current_week_value)[0]
        if len(current_positions) == 0:
            raise ValueError(f"Could not locate current visit for {pid} week={current_week_value}")
        end_pos = int(current_positions[-1])
        seq = patient.iloc[: end_pos + 1].copy()
        seq_features = np.log1p(seq[selected_features].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))
        seq_features = scaler.transform(seq_features).astype(np.float32)
        weeks = seq["week_num_num"].ffill().fillna(0.0).to_numpy(dtype=np.float32)
        delta = np.zeros(len(seq), dtype=np.float32)
        if len(seq) > 1:
            delta[1:] = np.maximum(0.0, np.diff(weeks))
        time_since_start = weeks - weeks[0]
        time_features = np.stack([np.log1p(delta), np.log1p(np.maximum(0.0, time_since_start))], axis=1).astype(np.float32)
        full_seq = np.concatenate([seq_features, time_features], axis=1)
        seq_len = len(seq)
        x[row_idx, :seq_len, :] = full_seq
        dt[row_idx, :seq_len] = np.log1p(delta)
        lengths[row_idx] = seq_len
        y[row_idx] = LABEL_TO_INT[str(ex["future_activity"])]

    feature_names = selected_features + ["time__log_delta_week", "time__log_week_since_start"]
    meta = {
        "fold": int(fold),
        "n_examples": int(n_examples),
        "max_len": int(max_len),
        "input_dim": int(x.shape[-1]),
        "selected_omics_features": int(len(selected_features)),
        "splits": pd.Series(split).value_counts().to_dict(),
        "classes": pd.Series(y).map(INT_TO_LABEL).value_counts().to_dict(),
    }
    cv_dir = PROCESSED_DIR / "cv_sequence_liquid"
    cv_dir.mkdir(parents=True, exist_ok=True)
    (cv_dir / f"fold_{fold}_sequence_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return SequenceBundle(
        x=x,
        dt=dt,
        lengths=lengths,
        y=y,
        split=split,
        participant_id=participant_id,
        current_week=current_week,
        next_week=next_week,
        feature_names=feature_names,
    )


def make_loaders(bundle: SequenceBundle, batch_size: int, seed: int) -> dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {}
    for split_name in ["train", "validation", "test"]:
        dataset = SequenceDataset(bundle, split_name)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=collate_batch,
            generator=generator if split_name == "train" else None,
        )
    return loaders


def class_weights_from_train(bundle: SequenceBundle) -> torch.Tensor:
    train_y = bundle.y[bundle.split == "train"]
    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_cv_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
) -> dict[str, object]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_score = -math.inf
    best_epoch = -1
    stale = 0

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
        val_score = val_metrics.get("roc_auc", float("nan"))
        if math.isnan(val_score):
            val_score = val_metrics["balanced_accuracy"]
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "train": evaluate_model(model, loaders["train"], device),
        "validation": evaluate_model(model, loaders["validation"], device),
        "test": evaluate_model(model, loaders["test"], device),
    }


def split_summary(bundle: SequenceBundle, fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "split": split_name,
                "examples": int(np.sum(bundle.split == split_name)),
                "participants": int(len(set(bundle.participant_id[bundle.split == split_name]))),
                "active": int(np.sum((bundle.split == split_name) & (bundle.y == 1))),
                "remission": int(np.sum((bundle.split == split_name) & (bundle.y == 0))),
            }
            for split_name in ["train", "validation", "test"]
        ]
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"] == "test"].copy()
    grouped = (
        test.groupby("model")
        .agg(
            runs=("roc_auc", "size"),
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
    return grouped


def write_report(results: pd.DataFrame, summary: pd.DataFrame, splits: pd.DataFrame, args: argparse.Namespace) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_dir = PROCESSED_DIR / "cv_sequence_liquid"
    cv_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cv_dir / "sequence_liquid_cv_results.csv", index=False)
    summary.to_csv(cv_dir / "sequence_liquid_cv_summary.csv", index=False)
    splits.to_csv(cv_dir / "sequence_liquid_cv_split_summary.csv", index=False)
    report = [
        "# 5-fold Sequence Liquid CV v0",
        "",
        "## Setup",
        "",
        f"- Folds: {args.folds}",
        f"- Seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- Selected omics features per fold: {args.features}",
        f"- Hidden dimension: {args.hidden}",
        f"- Epochs / patience: {args.epochs} / {args.patience}",
        "- Target: next visit active/remission",
        "- Split policy: each run uses one participant fold for test, the next fold for validation, and the remaining folds for train.",
        "- Feature selection and scaling are fitted inside each training fold only.",
        "",
        "## Test Summary Across Fold x Seed Runs",
        "",
        markdown_table(summary),
        "",
        "## Fold Split Summary",
        "",
        markdown_table(splits),
        "",
        "## Per-run Test Results",
        "",
        markdown_table(results[results["split"] == "test"].sort_values(["fold", "seed", "model"])),
        "",
        "## Interpretation",
        "",
        "- This CV experiment is more reliable than a single 70/15/15 split because every participant fold is used as test once.",
        "- If Liquid/CfC remains above GRU on mean ROC-AUC or balanced accuracy, it supports the longitudinal liquid-model hypothesis.",
        "- Variance should be reported together with the mean because active labels are sparse and per-fold test sets are small.",
        "",
        "## Generated Files",
        "",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid/sequence_liquid_cv_results.csv`",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid/sequence_liquid_cv_summary.csv`",
        "- `work/data/ibdmdb/processed/cv_sequence_liquid/sequence_liquid_cv_split_summary.csv`",
    ]
    out = OUTPUTS_DIR / "sequence_liquid_cv_report_v0.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out}")
    print(summary.to_string(index=False))


def parse_seeds(seed_text: str) -> list[int]:
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-fold CV for GRU and liquid/CfC sequence models.")
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44"))
    parser.add_argument("--folds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    all_rows = []
    split_rows = []
    for fold in args.folds:
        print(f"Building fold {fold}")
        bundle = make_sequence_bundle_for_splits(make_cv_splits(fold), args.features, fold)
        split_rows.append(split_summary(bundle, fold))
        input_dim = int(bundle.x.shape[-1])
        class_weights = class_weights_from_train(bundle)
        for seed in args.seeds:
            set_seed(seed)
            loaders = make_loaders(bundle, args.batch_size, seed)
            model_specs = [
                ("gru", GRUClassifier(input_dim=input_dim, hidden_dim=args.hidden)),
                ("liquid_cfc", LiquidClassifier(input_dim=input_dim, hidden_dim=args.hidden)),
            ]
            for model_name, model in model_specs:
                print(f"fold={fold} seed={seed} model={model_name}")
                result = train_one_cv_model(model, loaders, class_weights, device, args.epochs, args.patience, args.lr)
                for split_name in ["train", "validation", "test"]:
                    metrics = result[split_name]
                    all_rows.append(
                        {
                            "fold": int(fold),
                            "seed": int(seed),
                            "model": model_name,
                            "split": split_name,
                            "n": metrics["n"],
                            "active_rate": metrics["active_rate"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "f1_macro": metrics["f1_macro"],
                            "roc_auc": metrics["roc_auc"],
                            "average_precision": metrics["average_precision"],
                            "best_epoch": result["best_epoch"],
                            "best_validation_score": result["best_validation_score"],
                        }
                    )

    results = pd.DataFrame(all_rows)
    split_df = pd.concat(split_rows, ignore_index=True)
    summary = summarize_results(results)
    write_report(results, summary, split_df, args)


if __name__ == "__main__":
    main()
