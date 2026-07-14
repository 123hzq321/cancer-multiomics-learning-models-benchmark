from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROCESSED_DIR = Path("work/data/tcga_brca_cbioportal/processed")
OUTPUTS_DIR = Path("outputs")
MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No rows._"
    rendered = frame.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            rendered[col] = rendered[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(rendered.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rendered.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


def safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        classes = np.unique(y_true)
        if len(classes) < 2:
            return float("nan")
        if len(classes) == 2:
            if proba.ndim == 2 and proba.shape[1] > 1:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_pred = proba.argmax(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": safe_roc_auc(y_true, proba),
    }


def load_data(table_path: Path, split_path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], LabelEncoder]:
    table = pd.read_csv(table_path)
    splits = pd.read_csv(split_path)
    data = table.merge(splits, on="sampleId", how="inner")
    data = data[data["split"].isin(["train", "valid", "test"])].copy()
    data = data[data["SUBTYPE"].notna()].copy()

    modality_cols = {
        modality: [col for col in data.columns if col.startswith(f"{modality}__")]
        for modality in MODALITIES
    }
    missing = [modality for modality, cols in modality_cols.items() if not cols]
    if missing:
        raise ValueError(f"Missing modality columns: {missing}")

    label_encoder = LabelEncoder()
    data["target"] = label_encoder.fit_transform(data["SUBTYPE"].astype(str))
    return data, modality_cols, label_encoder


def make_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


class MultiOmicsPreprocessor:
    def __init__(self, modality_cols: dict[str, list[str]]):
        self.modality_cols = modality_cols
        self.imputers: dict[str, SimpleImputer] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_dims: dict[str, int] = {}

    def fit(self, train_df: pd.DataFrame) -> "MultiOmicsPreprocessor":
        for modality, cols in self.modality_cols.items():
            x = train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            imputer = make_imputer()
            scaler = StandardScaler()
            x_imp = imputer.fit_transform(x)
            x_scaled = scaler.fit_transform(x_imp)
            self.imputers[modality] = imputer
            self.scalers[modality] = scaler
            self.feature_dims[modality] = x_scaled.shape[1]
        return self

    def transform_modalities(self, frame: pd.DataFrame) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for modality, cols in self.modality_cols.items():
            x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            x = self.imputers[modality].transform(x)
            x = self.scalers[modality].transform(x)
            arrays.append(x.astype(np.float32))
        return arrays

    def transform_fusion(self, frame: pd.DataFrame) -> np.ndarray:
        return np.concatenate(self.transform_modalities(frame), axis=1).astype(np.float32)


class MultiOmicsDataset(Dataset):
    def __init__(self, fusion: np.ndarray, modalities: list[np.ndarray], y: np.ndarray):
        self.fusion = torch.tensor(fusion, dtype=torch.float32)
        self.modalities = [torch.tensor(x, dtype=torch.float32) for x in modalities]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "fusion": self.fusion[idx],
            "modalities": [x[idx] for x in self.modalities],
            "y": self.y[idx],
        }


class EarlyFusionMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        return self.net(fusion)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class EarlyFusionResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 192,
        blocks: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        return self.head(self.blocks(self.input(fusion)))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dims: list[int], embed_dim: int, dropout: float = 0.15):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(embed_dim),
                )
                for input_dim in input_dims
            ]
        )
        self.modality_embeddings = nn.Parameter(torch.zeros(len(input_dims), embed_dim))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)

    def forward(self, modalities: list[torch.Tensor]) -> torch.Tensor:
        encoded = []
        for idx, (encoder, x) in enumerate(zip(self.encoders, modalities, strict=True)):
            encoded.append(encoder(x) + self.modality_embeddings[idx])
        return torch.stack(encoded, dim=1)


class ModalityGRU(nn.Module):
    def __init__(self, input_dims: list[int], num_classes: int, embed_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        _, h = self.gru(z)
        return self.head(h[-1])


class ModalityLSTM(nn.Module):
    def __init__(self, input_dims: list[int], num_classes: int, embed_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.chomp = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.chomp:
            y = y[:, :, :-self.chomp]
        return x + self.dropout(self.activation(self.norm(y)))


class ModalityTCN(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        num_classes: int,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            CausalConvBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            CausalConvBlock(hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            CausalConvBlock(hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities).transpose(1, 2)
        h = self.blocks(self.input_proj(z)).transpose(1, 2)
        return self.head(h[:, -1, :])


class ModalityDeepSets(nn.Module):
    def __init__(self, input_dims: list[int], num_classes: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        pooled = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
        return self.head(pooled)


class ModalityAttentionFusion(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        num_classes: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        query = self.query.expand(z.shape[0], -1, -1)
        pooled, _ = self.attention(query, z, z, need_weights=False)
        return self.head(pooled[:, 0, :])


class ModalityTransformer(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        num_classes: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        cls = self.cls_token.expand(z.shape[0], -1, -1)
        h = self.transformer(torch.cat([cls, z], dim=1))
        return self.head(h[:, 0, :])


class ModalityGatedLateFusion(nn.Module):
    def __init__(self, input_dims: list[int], num_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in input_dims])
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        gate = torch.softmax(self.gate(z), dim=1)
        logits = torch.stack([head(z[:, idx, :]) for idx, head in enumerate(self.classifiers)], dim=1)
        return (gate * logits).sum(dim=1)


class ModalityLiquidCfC(nn.Module):
    def __init__(self, input_dims: list[int], num_classes: int, embed_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.encoder = ModalityEncoder(input_dims, embed_dim)
        self.input_to_candidate = nn.Linear(embed_dim, hidden_dim)
        self.hidden_to_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rate_net = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.gate_net = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, fusion: torch.Tensor, modalities: list[torch.Tensor]) -> torch.Tensor:
        z = self.encoder(modalities)
        h = torch.zeros(z.shape[0], self.hidden_to_candidate.out_features, device=z.device, dtype=z.dtype)
        dt = torch.tensor(1.0, device=z.device, dtype=z.dtype)
        for step in range(z.shape[1]):
            x_t = z[:, step, :]
            combined = torch.cat([x_t, h], dim=-1)
            rate = torch.nn.functional.softplus(self.rate_net(combined)) + 1e-3
            alpha = 1.0 - torch.exp(-rate * dt)
            candidate = torch.tanh(self.input_to_candidate(x_t) + self.hidden_to_candidate(h))
            gate = torch.sigmoid(self.gate_net(combined))
            target = gate * candidate + (1.0 - gate) * h
            h = (1.0 - alpha) * h + alpha * target
        return self.head(h)


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(
    processed: dict[str, dict[str, object]],
    batch_size: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split_name, payload in processed.items():
        dataset = MultiOmicsDataset(
            payload["fusion"],
            payload["modalities"],
            payload["y"],
        )
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name == "train"))
    return loaders


@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for batch in loader:
        fusion = batch["fusion"].to(device)
        modalities = [x.to(device) for x in batch["modalities"]]
        y = batch["y"].numpy()
        logits = model(fusion, modalities)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        ys.append(y)
        probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)


def train_torch_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    y_train: np.ndarray,
    num_classes: int,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> dict[str, object]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, num_classes).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loaders["train"]:
            fusion = batch["fusion"].to(device)
            modalities = [x.to(device) for x in batch["modalities"]]
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(fusion, modalities), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        y_valid, valid_proba = predict_torch(model, loaders["valid"], device)
        valid_metrics = compute_metrics(y_valid, valid_proba)
        score = valid_metrics["roc_auc_ovr_macro"]
        if math.isnan(score):
            score = valid_metrics["balanced_accuracy"]
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    split_metrics = {}
    split_predictions = {}
    for split_name, loader in loaders.items():
        y_true, proba = predict_torch(model, loader, device)
        split_metrics[split_name] = compute_metrics(y_true, proba)
        split_predictions[split_name] = (y_true, proba)
    return {
        "metrics": split_metrics,
        "predictions": split_predictions,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
    }


def fit_sklearn_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    processed: dict[str, dict[str, object]],
    seed: int,
) -> list[dict[str, object]]:
    models = [
        (
            "logistic_early_fusion",
            "linear_baseline",
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        (
            "extra_trees_early_fusion",
            "tree_baseline",
            ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ]
    results = []
    for model_name, group, model in models:
        model.fit(x_train, y_train)
        split_metrics = {}
        split_predictions = {}
        for split_name, payload in processed.items():
            x = payload["fusion"]
            y = payload["y"]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
            else:
                scores = model.decision_function(x)
                proba = torch.softmax(torch.tensor(scores), dim=-1).numpy()
            split_metrics[split_name] = compute_metrics(y, proba)
            split_predictions[split_name] = (y, proba)
        results.append(
            {
                "model": model_name,
                "model_group": group,
                "metrics": split_metrics,
                "predictions": split_predictions,
                "best_epoch": np.nan,
                "best_validation_score": split_metrics["valid"]["roc_auc_ovr_macro"],
            }
        )
    return results


def rows_from_result(seed: int, model_name: str, model_group: str, result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for split_name, metrics in result["metrics"].items():
        row = {
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


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"] == "test"].copy()
    summary = (
        test.groupby(["model", "model_group"])
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
    return summary.sort_values(
        ["f1_macro_mean", "balanced_accuracy_mean", "roc_auc_ovr_macro_mean"],
        ascending=False,
    ).reset_index(drop=True)


def make_processed_splits(
    data: pd.DataFrame,
    preprocessor: MultiOmicsPreprocessor,
) -> dict[str, dict[str, object]]:
    processed: dict[str, dict[str, object]] = {}
    for split_name in ["train", "valid", "test"]:
        frame = data[data["split"].eq(split_name)].copy()
        modalities = preprocessor.transform_modalities(frame)
        processed[split_name] = {
            "frame": frame,
            "modalities": modalities,
            "fusion": np.concatenate(modalities, axis=1).astype(np.float32),
            "y": frame["target"].to_numpy(dtype=np.int64),
            "sampleId": frame["sampleId"].to_numpy(),
        }
    return processed


def model_factories(input_dims: list[int], fusion_dim: int, num_classes: int) -> list[tuple[str, str, nn.Module]]:
    return [
        ("mlp_early_fusion", "static_nn", EarlyFusionMLP(fusion_dim, num_classes)),
        ("resmlp_early_fusion", "static_nn", EarlyFusionResMLP(fusion_dim, num_classes)),
        ("deepsets_modality_fusion", "set_nn", ModalityDeepSets(input_dims, num_classes)),
        ("attention_modality_fusion", "attention_nn", ModalityAttentionFusion(input_dims, num_classes)),
        ("transformer_modality_sequence", "attention_nn", ModalityTransformer(input_dims, num_classes)),
        ("gated_late_fusion", "late_fusion_nn", ModalityGatedLateFusion(input_dims, num_classes)),
        ("gru_modality_sequence", "sequence_nn", ModalityGRU(input_dims, num_classes)),
        ("lstm_modality_sequence", "sequence_nn", ModalityLSTM(input_dims, num_classes)),
        ("tcn_modality_sequence", "sequence_nn", ModalityTCN(input_dims, num_classes)),
        ("liquid_cfc_modality_sequence", "liquid_nn", ModalityLiquidCfC(input_dims, num_classes)),
    ]


def save_predictions(
    prediction_rows: list[dict[str, object]],
    out_path: Path,
    label_encoder: LabelEncoder,
) -> None:
    rows = []
    labels = list(label_encoder.classes_)
    for item in prediction_rows:
        y_true, proba = item["y_true"], item["proba"]
        sample_ids = item["sample_ids"]
        for idx, sample_id in enumerate(sample_ids):
            row = {
                "seed": item["seed"],
                "model": item["model"],
                "split": item["split"],
                "sampleId": sample_id,
                "true_label": labels[int(y_true[idx])],
                "pred_label": labels[int(np.argmax(proba[idx]))],
            }
            for class_idx, label in enumerate(labels):
                row[f"prob__{label}"] = float(proba[idx, class_idx])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_report(
    out_path: Path,
    summary: pd.DataFrame,
    results: pd.DataFrame,
    data: pd.DataFrame,
    label_encoder: LabelEncoder,
    modality_cols: dict[str, list[str]],
    best_confusion: pd.DataFrame,
    best_model: str,
) -> None:
    split_counts = data.groupby(["split", "SUBTYPE"]).size().unstack(fill_value=0).reset_index()
    feature_counts = pd.DataFrame(
        [{"modality": modality, "features": len(cols)} for modality, cols in modality_cols.items()]
    )
    best_rows = results[(results["split"] == "test") & (results["model"] == best_model)].copy()
    best_metric_line = best_rows[
        ["seed", "accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr_macro"]
    ].sort_values("f1_macro", ascending=False)

    lines = [
        "# TCGA BRCA Subtype Baselines vs Liquid/CfC v0",
        "",
        "## Task",
        "",
        "Predict TCGA BRCA molecular subtype from matched multi-omics features.",
        "",
        "- Target: `SUBTYPE`",
        f"- Classes: {', '.join(label_encoder.classes_)}",
        "- Splits: provided 70/15/15 stratified train/validation/test split.",
        "- Modalities: mRNA, GISTIC CNA, log2 CNA, HM450 methylation, RPPA, mutation.",
        "- Preprocessing: median imputation and standard scaling fitted on train split only.",
        "",
        "## Feature Counts",
        "",
        markdown_table(feature_counts),
        "",
        "## Split by Class",
        "",
        markdown_table(split_counts),
        "",
        "## Test Summary",
        "",
        markdown_table(summary, floatfmt=".4f"),
        "",
        f"## Best Model by Macro F1: `{best_model}`",
        "",
        markdown_table(best_metric_line, floatfmt=".4f"),
        "",
        "## Best Model Confusion Matrix",
        "",
        markdown_table(best_confusion),
        "",
        "## Interpretation",
        "",
        "This is a cross-sectional multi-omics subtype task, not a natural temporal prediction task. "
        "The Liquid/CfC model is therefore interpreted as a cross-modality state-update module: each omics layer "
        "updates a hidden state, and the final state predicts subtype. Strong early-fusion baselines remain essential.",
        "",
        "If Liquid/CfC does not clearly beat MLP/GRU/LSTM/TCN, the result should be written as a model comparison and "
        "representation-learning result rather than as a claim that liquid networks are universally superior.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(PROCESSED_DIR / "brca_multimodal_impact468_table.csv"))
    parser.add_argument("--splits", default=str(PROCESSED_DIR / "brca_subtype_splits_70_15_15.csv"))
    parser.add_argument("--results-csv", default=str(PROCESSED_DIR / "brca_subtype_model_results.csv"))
    parser.add_argument("--summary-csv", default=str(PROCESSED_DIR / "brca_subtype_model_summary.csv"))
    parser.add_argument("--predictions-csv", default=str(PROCESSED_DIR / "brca_subtype_model_predictions.csv"))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "tcga_brca_subtype_baselines_vs_liquid_report_v0.md"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    device = torch.device(args.device)

    data, modality_cols, label_encoder = load_data(Path(args.table), Path(args.splits))
    train_df = data[data["split"].eq("train")].copy()
    preprocessor = MultiOmicsPreprocessor(modality_cols).fit(train_df)
    processed = make_processed_splits(data, preprocessor)
    input_dims = [processed["train"]["modalities"][idx].shape[1] for idx in range(len(MODALITIES))]
    fusion_dim = processed["train"]["fusion"].shape[1]
    num_classes = len(label_encoder.classes_)
    y_train = processed["train"]["y"]

    all_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for seed in seeds:
        set_seed(seed)
        print(f"Seed {seed}: sklearn baselines")
        sklearn_results = fit_sklearn_models(processed["train"]["fusion"], y_train, processed, seed)
        for result in sklearn_results:
            all_rows.extend(rows_from_result(seed, result["model"], result["model_group"], result))
            for split_name, (y_true, proba) in result["predictions"].items():
                prediction_rows.append(
                    {
                        "seed": seed,
                        "model": result["model"],
                        "split": split_name,
                        "sample_ids": processed[split_name]["sampleId"],
                        "y_true": y_true,
                        "proba": proba,
                    }
                )

        loaders = make_loaders(processed, args.batch_size)
        for model_name, model_group, model in model_factories(input_dims, fusion_dim, num_classes):
            set_seed(seed)
            print(f"Seed {seed}: training {model_name}")
            result = train_torch_model(
                model,
                loaders,
                y_train,
                num_classes,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
                patience=args.patience,
            )
            all_rows.extend(rows_from_result(seed, model_name, model_group, result))
            for split_name, (y_true, proba) in result["predictions"].items():
                prediction_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "split": split_name,
                        "sample_ids": processed[split_name]["sampleId"],
                        "y_true": y_true,
                        "proba": proba,
                    }
                )

    results = pd.DataFrame(all_rows)
    summary = summarize_results(results)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    save_predictions(prediction_rows, Path(args.predictions_csv), label_encoder)

    best_model = summary.iloc[0]["model"]
    best_seed = (
        results[(results["split"] == "test") & (results["model"] == best_model)]
        .sort_values("f1_macro", ascending=False)
        .iloc[0]["seed"]
    )
    pred_df = pd.read_csv(args.predictions_csv)
    best_pred = pred_df[(pred_df["model"].eq(best_model)) & (pred_df["seed"].eq(best_seed)) & (pred_df["split"].eq("test"))]
    cm = confusion_matrix(
        best_pred["true_label"],
        best_pred["pred_label"],
        labels=list(label_encoder.classes_),
    )
    best_confusion = pd.DataFrame(cm, index=[f"true__{x}" for x in label_encoder.classes_], columns=[f"pred__{x}" for x in label_encoder.classes_])
    best_confusion.to_csv(PROCESSED_DIR / "brca_subtype_best_model_confusion_matrix.csv")
    (PROCESSED_DIR / "brca_subtype_label_mapping.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(label_encoder.classes_)}, indent=2),
        encoding="utf-8",
    )
    write_report(
        Path(args.report),
        summary,
        results,
        data,
        label_encoder,
        modality_cols,
        best_confusion.reset_index(names="true_label"),
        best_model,
    )
    print(f"Done. Report: {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
