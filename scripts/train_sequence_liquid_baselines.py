from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "work" / "data" / "ibdmdb" / "processed"
OUTPUTS_DIR = ROOT / "outputs"


LABEL_TO_INT = {"remission": 0, "active": 1}
INT_TO_LABEL = {0: "remission", 1: "active"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SequenceBundle:
    x: np.ndarray
    dt: np.ndarray
    lengths: np.ndarray
    y: np.ndarray
    split: np.ndarray
    participant_id: np.ndarray
    current_week: np.ndarray
    next_week: np.ndarray
    feature_names: list[str]


def load_visit_table() -> pd.DataFrame:
    path = PROCESSED_DIR / "visit_level_mgx_mtx_path_table.csv"
    df = pd.read_csv(path)
    df = df.sort_values(["participant_id", "week_num_num", "visit_num_num"]).reset_index(drop=True)
    return df


def choose_features(df: pd.DataFrame, train_example_rows: pd.DataFrame, feature_cols: list[str], k: int) -> list[str]:
    y = train_example_rows["future_activity"].map(LABEL_TO_INT).to_numpy()
    x = np.log1p(train_example_rows[feature_cols].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))
    # Drop globally constant columns before univariate scoring.
    variance = x.var(axis=0)
    variable_idx = np.where(variance > 0)[0]
    if len(variable_idx) == 0:
        raise ValueError("No variable features available for sequence training.")
    x_var = x[:, variable_idx]
    k_eff = min(k, x_var.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_eff)
    selector.fit(x_var, y)
    selected_variable_idx = variable_idx[selector.get_support(indices=True)]
    return [feature_cols[i] for i in selected_variable_idx]


def build_examples(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_activity"] = df.groupby("participant_id")["activity_binary"].shift(-1)
    df["next_week_num"] = df.groupby("participant_id")["week_num_num"].shift(-1)
    df["delta_week_to_next"] = df["next_week_num"] - df["week_num_num"]
    examples = df[
        df["future_activity"].isin(LABEL_TO_INT)
        & df["delta_week_to_next"].notna()
        & (df["delta_week_to_next"] >= 0)
    ].copy()
    examples = examples.merge(splits[["participant_id", "split"]], on="participant_id", how="left")
    return examples


def make_sequence_bundle(k_features: int = 512) -> SequenceBundle:
    df = load_visit_table()
    splits = pd.read_csv(PROCESSED_DIR / "splits_70_15_15_by_participant.csv")
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("mgx_path__") or c.startswith("mtx_path__") or c in ["has_mgx_path", "has_mtx_path"]
    ]
    examples = build_examples(df, splits)
    train_examples = examples[examples["split"] == "train"]
    selected_features = choose_features(df, train_examples, feature_cols, k_features)

    # Fit scaling only on training visits from patients in the training split.
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
        # Log-compress time intervals so a few long gaps do not dominate.
        time_features = np.stack([np.log1p(delta), np.log1p(np.maximum(0.0, time_since_start))], axis=1).astype(np.float32)
        full_seq = np.concatenate([seq_features, time_features], axis=1)
        seq_len = len(seq)
        x[row_idx, :seq_len, :] = full_seq
        dt[row_idx, :seq_len] = np.log1p(delta)
        lengths[row_idx] = seq_len
        y[row_idx] = LABEL_TO_INT[str(ex["future_activity"])]

    feature_names = selected_features + ["time__log_delta_week", "time__log_week_since_start"]
    bundle = SequenceBundle(
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

    npz_path = PROCESSED_DIR / "sequence_future_activity_dataset.npz"
    np.savez_compressed(
        npz_path,
        x=x,
        dt=dt,
        lengths=lengths,
        y=y,
        split=split,
        participant_id=participant_id,
        current_week=current_week,
        next_week=next_week,
        feature_names=np.array(feature_names, dtype=object),
    )
    meta = {
        "n_examples": int(n_examples),
        "max_len": int(max_len),
        "input_dim": int(x.shape[-1]),
        "selected_omics_features": int(len(selected_features)),
        "feature_selection": "SelectKBest(f_classif) fitted on train split current visits only",
        "scaling": "StandardScaler fitted on train split visits only after log1p",
        "target": "next visit active/remission",
        "splits": pd.Series(split).value_counts().to_dict(),
        "classes": pd.Series(y).map(INT_TO_LABEL).value_counts().to_dict(),
    }
    (PROCESSED_DIR / "sequence_future_activity_dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle


class SequenceDataset(Dataset):
    def __init__(self, bundle: SequenceBundle, split_name: str):
        self.indices = np.where(bundle.split == split_name)[0]
        self.x = bundle.x
        self.dt = bundle.dt
        self.lengths = bundle.lengths
        self.y = bundle.y

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        length = int(self.lengths[idx])
        return (
            torch.from_numpy(self.x[idx, :length, :]),
            torch.from_numpy(self.dt[idx, :length]),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )


def collate_batch(batch):
    xs, dts, lengths, ys = zip(*batch)
    max_len = max(int(length) for length in lengths)
    input_dim = xs[0].shape[-1]
    x_pad = torch.zeros(len(batch), max_len, input_dim, dtype=torch.float32)
    dt_pad = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, (x_i, dt_i, length_i) in enumerate(zip(xs, dts, lengths)):
        length = int(length_i)
        x_pad[i, :length, :] = x_i
        dt_pad[i, :length] = dt_i
    return x_pad, dt_pad, torch.stack(lengths), torch.stack(ys)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor, dt: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        return self.head(h_last)


class ClosedFormLiquidCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.rate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.skip_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        candidate = self.candidate(combined)
        rate = self.rate(combined) + 1e-4
        decay = torch.exp(-rate * dt_t.unsqueeze(-1))
        liquid_update = decay * h + (1.0 - decay) * candidate
        gate = self.skip_gate(combined)
        return gate * liquid_update + (1.0 - gate) * candidate


class LiquidClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.cell = ClosedFormLiquidCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor, dt: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, max_len, _ = x.shape
        h = torch.zeros(batch, self.head[1].in_features, dtype=x.dtype, device=x.device)
        projected = self.input_proj(x)
        for t in range(max_len):
            h_new = self.cell(projected[:, t, :], h, dt[:, t])
            valid = (lengths > t).unsqueeze(-1)
            h = torch.where(valid, h_new, h)
        return self.head(h)


def make_loaders(bundle: SequenceBundle, batch_size: int):
    loaders = {}
    for split_name in ["train", "validation", "test"]:
        dataset = SequenceDataset(bundle, split_name)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=collate_batch,
        )
    return loaders


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true = []
    y_score = []
    y_pred = []
    with torch.no_grad():
        for x, dt, lengths, y in loader:
            x = x.to(device)
            dt = dt.to(device)
            lengths = lengths.to(device)
            logits = model(x, dt, lengths)
            prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_score.extend(prob.tolist())
            y_pred.extend(pred.tolist())
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_score_arr = np.array(y_score)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
    }
    if len(np.unique(y_true_arr)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        metrics["average_precision"] = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    metrics["n"] = int(len(y_true_arr))
    metrics["active_rate"] = float(np.mean(y_true_arr == 1))
    return metrics


def train_one_model(
    model_name: str,
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
    history = []
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, dt, lengths, y in loaders["train"]:
            x = x.to(device)
            dt = dt.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, dt, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
            train_count += len(y)

        val_metrics = evaluate_model(model, loaders["validation"], device)
        val_score = val_metrics.get("roc_auc", float("nan"))
        if math.isnan(val_score):
            val_score = val_metrics["balanced_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(1, train_count),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
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
    result = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_validation_score": float(best_score),
        "train": evaluate_model(model, loaders["train"], device),
        "validation": evaluate_model(model, loaders["validation"], device),
        "test": evaluate_model(model, loaders["test"], device),
        "history": history,
    }
    torch.save(best_state, PROCESSED_DIR / f"{model_name}_future_activity_state.pt")
    return result


def class_weights_from_train(bundle: SequenceBundle) -> torch.Tensor:
    train_y = bundle.y[bundle.split == "train"]
    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}" if np.isfinite(val) else "")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(bundle: SequenceBundle, results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        for split_name in ["train", "validation", "test"]:
            metrics = result[split_name]
            rows.append(
                {
                    "model": result["model"],
                    "split": split_name,
                    "n": metrics["n"],
                    "active_rate": metrics["active_rate"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "best_epoch": result["best_epoch"],
                }
            )
    result_df = pd.DataFrame(rows)
    result_df.to_csv(PROCESSED_DIR / "sequence_model_results.csv", index=False)
    serializable = []
    for result in results:
        result_copy = {k: v for k, v in result.items() if k != "history"}
        result_copy["history"] = result["history"]
        serializable.append(result_copy)
    (PROCESSED_DIR / "sequence_model_results.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": split_name,
                "examples": int(np.sum(bundle.split == split_name)),
                "participants": int(len(set(bundle.participant_id[bundle.split == split_name]))),
                "active": int(np.sum((bundle.split == split_name) & (bundle.y == 1))),
                "remission": int(np.sum((bundle.split == split_name) & (bundle.y == 0))),
            }
            for split_name in ["train", "validation", "test"]
        ]
    )
    split_summary.to_csv(PROCESSED_DIR / "sequence_dataset_split_summary.csv", index=False)

    report = [
        "# Sequence Liquid Baseline v0",
        "",
        "## Dataset",
        "",
        f"- Examples: {len(bundle.y)}",
        f"- Max sequence length: {bundle.x.shape[1]}",
        f"- Input dimension: {bundle.x.shape[2]}",
        f"- Selected omics features: {len(bundle.feature_names) - 2}",
        "- Target: next visit active/remission",
        "- Split: existing 70/15/15 participant-level split",
        "",
        "### Split Summary",
        "",
        markdown_table(split_summary),
        "",
        "## Model Results",
        "",
        markdown_table(result_df),
        "",
        "## Notes",
        "",
        "- GRU uses standard discrete recurrent updates.",
        "- LiquidCfC uses a closed-form time-decay cell: `h_new = exp(-rate * dt) * h + (1-exp(-rate*dt)) * candidate`, with an additional learned gate.",
        "- Feature selection and scaling are fitted on the train split only.",
        "- Metrics should be interpreted cautiously because active labels are sparse.",
        "",
        "## Generated Files",
        "",
        "- `work/data/ibdmdb/processed/sequence_future_activity_dataset.npz`",
        "- `work/data/ibdmdb/processed/sequence_future_activity_dataset_meta.json`",
        "- `work/data/ibdmdb/processed/sequence_model_results.csv`",
        "- `work/data/ibdmdb/processed/sequence_model_results.json`",
        "- `work/data/ibdmdb/processed/gru_future_activity_state.pt`",
        "- `work/data/ibdmdb/processed/liquid_cfc_future_activity_state.pt`",
    ]
    out = OUTPUTS_DIR / "sequence_liquid_baseline_report_v0.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out}")
    print(result_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU and liquid/CfC-style sequence baselines.")
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device. CPU is the default because some visible CUDA devices are unsupported by installed PyTorch wheels.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bundle = make_sequence_bundle(k_features=args.features)
    loaders = make_loaders(bundle, args.batch_size)
    class_weights = class_weights_from_train(bundle)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    input_dim = bundle.x.shape[-1]
    print(f"device={device} input_dim={input_dim} class_weights={class_weights.tolist()}")

    results = []
    results.append(
        train_one_model(
            "gru",
            GRUClassifier(input_dim=input_dim, hidden_dim=args.hidden),
            loaders,
            class_weights,
            device,
            args.epochs,
            args.patience,
            args.lr,
        )
    )
    results.append(
        train_one_model(
            "liquid_cfc",
            LiquidClassifier(input_dim=input_dim, hidden_dim=args.hidden),
            loaders,
            class_weights,
            device,
            args.epochs,
            args.patience,
            args.lr,
        )
    )
    write_report(bundle, results)


if __name__ == "__main__":
    main()
