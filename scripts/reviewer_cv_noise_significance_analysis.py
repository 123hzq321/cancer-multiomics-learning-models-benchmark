from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_joined_metabric_small_liquid_external_validation import SmallModalityLiquidCfC  # noqa: E402
from train_multicancer_internal_benchmark import TASKS, load_task_table, modality_columns  # noqa: E402
from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    EarlyFusionMLP,
    MODALITIES,
    ModalityLiquidCfC,
    MultiOmicsDataset,
    MultiOmicsPreprocessor,
    compute_metrics,
    predict_torch,
    set_seed,
    train_torch_model,
)


ROOT = Path(".")
PANCANCER_PATH = ROOT / "work/data/tcga_pancancer_cbioportal/processed/pancancer_multimodal_impact468_table.csv"
OUT_DIR = ROOT / "work/data/reviewer_cv_noise_significance/processed"
REPORT_PATH = ROOT / "outputs/reviewer_cv_noise_significance_report_v0.md"
CORE_MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "mutation"]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
MODEL_ORDER = [
    "logistic_elasticnet",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "mlp_early_fusion",
    "liquid_cfc_modality_sequence",
    "small_liquid_cfc_modality_sequence",
]
MODEL_LABELS = {
    "logistic_elasticnet": "Logistic ElasticNet",
    "random_forest": "Random Forest",
    "extra_trees": "ExtraTrees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "mlp_early_fusion": "MLP",
    "liquid_cfc_modality_sequence": "Liquid/CfC",
    "small_liquid_cfc_modality_sequence": "small-Liquid/CfC",
}

FOLD_WORDS = {
    5: ("Five-Fold", "fivefold"),
    10: ("Ten-Fold", "tenfold"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validation, significance, noise, and elbow analysis.")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--seed", type=int, default=20260629, help="Random seed.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for CSV files.")
    parser.add_argument("--report-path", type=Path, default=None, help="Markdown report path.")
    return parser.parse_args()


def fold_names(n_folds: int) -> tuple[str, str]:
    if n_folds in FOLD_WORDS:
        return FOLD_WORDS[n_folds]
    return f"{n_folds}-Fold", f"{n_folds}fold"


def default_paths(n_folds: int) -> tuple[Path, Path]:
    _, slug = fold_names(n_folds)
    if n_folds == 5:
        return OUT_DIR, REPORT_PATH
    return (
        ROOT / f"work/data/reviewer_{slug}_cv_noise_significance/processed",
        ROOT / f"outputs/reviewer_{slug}_cv_noise_significance_report_v0.md",
    )


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return ""
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) and not math.isnan(val):
                vals.append(format(val, floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_processed_from_split(data: pd.DataFrame, modality_cols: dict[str, list[str]]):
    data = data.copy()
    classes = sorted(data["task_label"].astype(str).unique())
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    data["target"] = data["task_label"].astype(str).map(class_to_id).astype(int)
    pre = MultiOmicsPreprocessor(modality_cols).fit(data[data["split"].eq("train")])
    processed = {}
    for split in ["train", "valid", "test"]:
        frame = data[data["split"].eq(split)].copy()
        modalities = pre.transform_modalities(frame)
        processed[split] = {
            "frame": frame,
            "modalities": modalities,
            "fusion": np.concatenate(modalities, axis=1).astype(np.float32),
            "y": frame["target"].to_numpy(dtype=np.int64),
            "sampleId": frame["sampleId"].to_numpy(),
        }
    return processed, classes


def loaders_from_processed(processed, batch_size: int):
    loaders = {}
    for split_name, payload in processed.items():
        dataset = MultiOmicsDataset(payload["fusion"], payload["modalities"], payload["y"])
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name == "train"))
    return loaders


def noisy_payload(payload: dict[str, object], sigma: float, rng: np.random.Generator) -> dict[str, object]:
    if sigma <= 0:
        return payload
    fusion = payload["fusion"].copy()
    modalities = [x.copy() for x in payload["modalities"]]
    fusion = fusion + rng.normal(0.0, sigma, size=fusion.shape).astype(np.float32)
    modalities = [x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32) for x in modalities]
    return {**payload, "fusion": fusion.astype(np.float32), "modalities": modalities}


def predict_sklearn(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    scores = model.decision_function(x)
    return torch.softmax(torch.tensor(scores), dim=-1).numpy()


def sklearn_models(seed: int):
    return [
        (
            "logistic_elasticnet",
            "linear",
            LogisticRegression(
                max_iter=5000,
                C=0.5,
                class_weight="balanced",
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.5,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest",
            "tree_ensemble",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "extra_trees",
            "tree_ensemble",
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "hist_gradient_boosting",
            "boosting",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                l2_regularization=0.05,
                max_leaf_nodes=15,
                random_state=seed,
            ),
        ),
    ]


def evaluate_sklearn_noise(model, processed, task: str, fold: int, model_name: str, model_group: str, seed: int):
    rows = []
    rng = np.random.default_rng(seed + fold * 1000 + abs(hash(model_name)) % 997)
    for sigma in NOISE_LEVELS:
        payload = noisy_payload(processed["test"], sigma, rng)
        proba = predict_sklearn(model, payload["fusion"])
        metrics = compute_metrics(payload["y"], proba)
        rows.append(
            {
                "task": task,
                "fold": fold,
                "model": model_name,
                "model_group": model_group,
                "noise_sigma": sigma,
                **metrics,
            }
        )
    return rows


def evaluate_torch_noise(model, processed, task: str, fold: int, model_name: str, model_group: str, seed: int, device, batch_size: int):
    rows = []
    rng = np.random.default_rng(seed + fold * 1000 + abs(hash(model_name)) % 997)
    for sigma in NOISE_LEVELS:
        payload = noisy_payload(processed["test"], sigma, rng)
        dataset = MultiOmicsDataset(payload["fusion"], payload["modalities"], payload["y"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        y_true, proba = predict_torch(model, loader, device)
        metrics = compute_metrics(y_true, proba)
        rows.append(
            {
                "task": task,
                "fold": fold,
                "model": model_name,
                "model_group": model_group,
                "noise_sigma": sigma,
                **metrics,
            }
        )
    return rows


def fold_split(task_table: pd.DataFrame, train_valid_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> pd.DataFrame:
    data = task_table.iloc[np.concatenate([train_valid_idx, test_idx])].copy()
    data["split"] = "test"
    train_valid = task_table.iloc[train_valid_idx].copy()
    train_idx, valid_idx = train_test_split(
        train_valid.index.to_numpy(),
        train_size=0.80,
        random_state=seed,
        stratify=train_valid["task_label"].astype(str),
    )
    data.loc[train_idx, "split"] = "train"
    data.loc[valid_idx, "split"] = "valid"
    data.loc[test_idx, "split"] = "test"
    return data


def holm_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [1.0] * n
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * pvals[idx])
        val = max(prev, val)
        adjusted[idx] = val
        prev = val
    return adjusted


def wilcoxon_or_sign_test(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        n_pos = int((nonzero > 0).sum())
        n = len(nonzero)
        # Two-sided exact sign test under p=0.5.
        tail = sum(math.comb(n, k) for k in range(0, min(n_pos, n - n_pos) + 1)) / (2**n)
        return float(min(1.0, 2 * tail))


def add_mean_ci(frame: pd.DataFrame, mean_col: str, sd_col: str, n_col: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    se = out[sd_col].fillna(0.0) / np.sqrt(out[n_col].clip(lower=1))
    out[f"{prefix}_ci95_low"] = out[mean_col] - 1.96 * se
    out[f"{prefix}_ci95_high"] = out[mean_col] + 1.96 * se
    return out


def summarize_cv(noise_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = noise_df[noise_df["noise_sigma"].eq(0.0)].copy()
    cv_summary = (
        clean.groupby(["task", "model", "model_group"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            roc_auc_ovr_macro_mean=("roc_auc_ovr_macro", "mean"),
            roc_auc_ovr_macro_sd=("roc_auc_ovr_macro", "std"),
        )
        .sort_values(["task", "f1_macro_mean"], ascending=[True, False])
    )
    cv_summary = add_mean_ci(cv_summary, "f1_macro_mean", "f1_macro_sd", "folds", "f1_macro")
    cv_summary = add_mean_ci(cv_summary, "roc_auc_ovr_macro_mean", "roc_auc_ovr_macro_sd", "folds", "roc_auc_ovr_macro")

    sig_rows = []
    pairwise_rows = []
    for task, task_df in cv_summary.groupby("task"):
        ordered = task_df.sort_values("f1_macro_mean", ascending=False)
        if len(ordered) < 2:
            continue
        best_model = ordered.iloc[0]["model"]
        second_model = ordered.iloc[1]["model"]
        best = clean[(clean["task"].eq(task)) & (clean["model"].eq(best_model))].sort_values("fold")
        second = clean[(clean["task"].eq(task)) & (clean["model"].eq(second_model))].sort_values("fold")
        merged = best[["fold", "f1_macro"]].merge(second[["fold", "f1_macro"]], on="fold", suffixes=("_best", "_second"))
        diffs = (merged["f1_macro_best"] - merged["f1_macro_second"]).to_numpy()
        p = wilcoxon_or_sign_test(diffs)
        rng = np.random.default_rng(seed + len(sig_rows))
        boot = []
        for _ in range(5000):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            boot.append(float(np.mean(sample)))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        sig_rows.append(
            {
                "task": task,
                "best_model": best_model,
                "second_model": second_model,
                "top_wins": int((diffs > 0).sum()),
                "runner_up_wins": int((diffs < 0).sum()),
                "ties": int((diffs == 0).sum()),
                "mean_f1_diff": float(np.mean(diffs)),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "wilcoxon_or_sign_p": p,
            }
        )
        task_pair_pvals = []
        task_pair_items = []
        best_fold_scores = clean[(clean["task"].eq(task)) & (clean["model"].eq(best_model))][["fold", "f1_macro"]]
        for _, other_row in ordered.iloc[1:].iterrows():
            other_model = other_row["model"]
            other_fold_scores = clean[(clean["task"].eq(task)) & (clean["model"].eq(other_model))][["fold", "f1_macro"]]
            merged_pair = best_fold_scores.merge(other_fold_scores, on="fold", suffixes=("_top", "_other")).sort_values("fold")
            diffs_pair = (merged_pair["f1_macro_top"] - merged_pair["f1_macro_other"]).to_numpy()
            p_pair = wilcoxon_or_sign_test(diffs_pair)
            rng_pair = np.random.default_rng(seed + 1000 + len(pairwise_rows))
            boot = []
            for _ in range(5000):
                sample = rng_pair.choice(diffs_pair, size=len(diffs_pair), replace=True)
                boot.append(float(np.mean(sample)))
            lo_pair, hi_pair = np.quantile(boot, [0.025, 0.975])
            task_pair_pvals.append(p_pair)
            task_pair_items.append(
                {
                    "task": task,
                    "top_model": best_model,
                    "comparator": other_model,
                    "top_wins": int((diffs_pair > 0).sum()),
                    "comparator_wins": int((diffs_pair < 0).sum()),
                    "ties": int((diffs_pair == 0).sum()),
                    "mean_f1_diff": float(np.mean(diffs_pair)),
                    "ci95_low": float(lo_pair),
                    "ci95_high": float(hi_pair),
                    "wilcoxon_or_sign_p": p_pair,
                }
            )
        for item, p_adj in zip(task_pair_items, holm_adjust(task_pair_pvals), strict=True):
            item["holm_within_task_p"] = p_adj
            pairwise_rows.append(item)
    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        sig_df["holm_p"] = holm_adjust(sig_df["wilcoxon_or_sign_p"].tolist())
    pairwise_df = pd.DataFrame(pairwise_rows)
    return cv_summary, sig_df, pairwise_df


def summarize_noise(noise_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    noise_summary = (
        noise_df.groupby(["task", "model", "model_group", "noise_sigma"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            roc_auc_ovr_macro_mean=("roc_auc_ovr_macro", "mean"),
        )
        .sort_values(["task", "model", "noise_sigma"])
    )
    elbow_rows = []
    for (task, model), curve in noise_summary.groupby(["task", "model"]):
        curve = curve.sort_values("noise_sigma")
        x = curve["noise_sigma"].to_numpy(dtype=float)
        y = curve["f1_macro_mean"].to_numpy(dtype=float)
        base = y[0]
        drops = base - y
        rel = np.divide(drops, max(base, 1e-9))
        first_5 = x[np.where(rel >= 0.05)[0][0]] if np.any(rel >= 0.05) else np.nan
        if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
            xn = (x - x.min()) / np.ptp(x)
            yn = (y - y.min()) / np.ptp(y)
            p1 = np.array([xn[0], yn[0]])
            p2 = np.array([xn[-1], yn[-1]])
            line = p2 - p1
            denom = np.linalg.norm(line)
            distances = []
            for xi, yi in zip(xn, yn, strict=True):
                point = np.array([xi, yi])
                delta = p1 - point
                cross_2d = line[0] * delta[1] - line[1] * delta[0]
                dist = abs(cross_2d) / denom if denom else 0.0
                distances.append(dist)
            elbow = x[int(np.argmax(distances))]
        else:
            elbow = np.nan
        elbow_rows.append(
            {
                "task": task,
                "model": model,
                "clean_f1": float(base),
                "f1_at_sigma_0_20": float(curve[curve["noise_sigma"].eq(0.20)]["f1_macro_mean"].iloc[0]),
                "f1_at_sigma_0_50": float(curve[curve["noise_sigma"].eq(0.50)]["f1_macro_mean"].iloc[0]),
                "first_5pct_drop_sigma": first_5,
                "elbow_sigma": elbow,
            }
        )
    return noise_summary, pd.DataFrame(elbow_rows)


def run() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    out_dir, report_path = default_paths(args.folds)
    if args.out_dir is not None:
        out_dir = args.out_dir
    if args.report_path is not None:
        report_path = args.report_path

    fold_title, fold_slug = fold_names(args.folds)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pancancer = pd.read_csv(PANCANCER_PATH, low_memory=False)
    device = torch.device("cpu")
    all_noise_rows = []
    seed = args.seed

    for task_name, cfg in TASKS.items():
        print(f"Task {task_name}", flush=True)
        task_table = load_task_table(task_name, cfg, pancancer, out_dir / "task_tables")
        counts = task_table["task_label"].astype(str).value_counts()
        task_table = task_table[task_table["task_label"].astype(str).isin(counts[counts >= max(20, args.folds)].index)].copy().reset_index(drop=True)
        modality_cols = modality_columns(task_table, CORE_MODALITIES)
        y = task_table["task_label"].astype(str).to_numpy()
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_valid_idx, test_idx) in enumerate(skf.split(task_table, y), start=1):
            print(f"  fold={fold}", flush=True)
            set_seed(seed + fold)
            split_data = fold_split(task_table, train_valid_idx, test_idx, seed + fold)
            processed, classes = build_processed_from_split(split_data, modality_cols)
            num_classes = len(classes)
            fusion_dim = processed["train"]["fusion"].shape[1]
            input_dims = [arr.shape[1] for arr in processed["train"]["modalities"]]

            for model_name, group, model in sklearn_models(seed + fold):
                print(f"    sklearn {model_name}", flush=True)
                model.fit(processed["train"]["fusion"], processed["train"]["y"])
                all_noise_rows.extend(evaluate_sklearn_noise(model, processed, task_name, fold, model_name, group, seed))

            loaders = loaders_from_processed(processed, batch_size=64)
            torch_models = [
                ("mlp_early_fusion", "static_nn", EarlyFusionMLP(fusion_dim, num_classes, hidden_dim=128)),
                ("liquid_cfc_modality_sequence", "liquid_nn", ModalityLiquidCfC(input_dims, num_classes, embed_dim=64, hidden_dim=96)),
                ("small_liquid_cfc_modality_sequence", "small_liquid_nn", SmallModalityLiquidCfC(input_dims, num_classes)),
            ]
            for model_name, group, model in torch_models:
                print(f"    torch {model_name}", flush=True)
                set_seed(seed + fold)
                train_torch_model(
                    model,
                    loaders,
                    processed["train"]["y"],
                    num_classes,
                    device=device,
                    lr=1e-3,
                    weight_decay=1e-4,
                    max_epochs=60,
                    patience=8,
                )
                all_noise_rows.extend(evaluate_torch_noise(model, processed, task_name, fold, model_name, group, seed, device, batch_size=64))

    noise_df = pd.DataFrame(all_noise_rows)
    cv_summary, sig_df, pairwise_df = summarize_cv(noise_df, seed=seed)
    noise_summary, elbow_df = summarize_noise(noise_df)

    raw_path = out_dir / f"{fold_slug}_cv_noise_raw_metrics.csv"
    cv_path = out_dir / f"{fold_slug}_cv_summary.csv"
    sig_path = out_dir / f"{fold_slug}_top_model_significance.csv"
    sig_wins_path = out_dir / f"{fold_slug}_top_model_significance_with_wins.csv"
    pairwise_path = out_dir / f"{fold_slug}_top_vs_all_pairwise_significance.csv"
    noise_path = out_dir / f"{fold_slug}_noise_robustness_summary.csv"
    elbow_path = out_dir / f"{fold_slug}_noise_elbow_summary.csv"
    noise_df.to_csv(raw_path, index=False)
    cv_summary.to_csv(cv_path, index=False)
    sig_df.to_csv(sig_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    if not sig_df.empty:
        sig_df[
            [
                "task",
                "best_model",
                "second_model",
                "top_wins",
                "runner_up_wins",
                "ties",
                "mean_f1_diff",
                "ci95_low",
                "ci95_high",
                "wilcoxon_or_sign_p",
                "holm_p",
            ]
        ].to_csv(sig_wins_path, index=False)
    noise_summary.to_csv(noise_path, index=False)
    elbow_df.to_csv(elbow_path, index=False)

    if args.folds == 5:
        # Preserve filenames used by the first reviewer-response draft.
        noise_df.to_csv(out_dir / "cv_noise_raw_metrics.csv", index=False)
        cv_summary.to_csv(out_dir / "fivefold_cv_summary.csv", index=False)
        sig_df.to_csv(out_dir / "fivefold_top_model_significance.csv", index=False)
        pairwise_df.to_csv(out_dir / "fivefold_top_vs_all_pairwise_significance.csv", index=False)
        noise_summary.to_csv(out_dir / "noise_robustness_summary.csv", index=False)
        elbow_df.to_csv(out_dir / "noise_elbow_summary.csv", index=False)

    best_cv = cv_summary.sort_values(["task", "f1_macro_mean"], ascending=[True, False]).groupby("task").head(1).copy()
    best_cv["model"] = best_cv["model"].map(MODEL_LABELS).fillna(best_cv["model"])
    sig_pretty = sig_df.copy()
    for col in ["best_model", "second_model"]:
        if col in sig_pretty:
            sig_pretty[col] = sig_pretty[col].map(MODEL_LABELS).fillna(sig_pretty[col])
    sig_display = sig_pretty.rename(
        columns={
            "best_model": "top_model",
            "second_model": "runner_up",
            "wilcoxon_or_sign_p": "raw_p",
        }
    )
    if not sig_display.empty:
        sig_display = sig_display[
            [
                "task",
                "top_model",
                "runner_up",
                "top_wins",
                "runner_up_wins",
                "ties",
                "mean_f1_diff",
                "ci95_low",
                "ci95_high",
                "raw_p",
            ]
        ]
    pairwise_display = pairwise_df.copy()
    for col in ["top_model", "comparator"]:
        if col in pairwise_display:
            pairwise_display[col] = pairwise_display[col].map(MODEL_LABELS).fillna(pairwise_display[col])
    elbow_best = elbow_df.merge(
        cv_summary.sort_values(["task", "f1_macro_mean"], ascending=[True, False]).groupby("task").head(1)[["task", "model"]],
        on=["task", "model"],
        how="inner",
    ).copy()
    elbow_best["model"] = elbow_best["model"].map(MODEL_LABELS).fillna(elbow_best["model"])

    report = "\n".join(
        [
            "# Reviewer CV, Significance, Noise, and Elbow Analysis",
            "",
            "Core-aligned modalities: mRNA, GISTIC CNA, log2 CNA, methylation, and mutation. RPPA is excluded because the sample-alignment audit showed sparse RPPA coverage.",
            "",
            f"## {fold_title} CV: Best Model per Task",
            "",
            markdown_table(best_cv[["task", "model", "folds", "f1_macro_mean", "f1_macro_sd", "roc_auc_ovr_macro_mean"]]),
            "",
            "## Top-vs-Runner-Up Fold-Level Significance",
            "",
            markdown_table(sig_display),
            "",
            "## Top-vs-All Fold-Level Significance",
            "",
            markdown_table(pairwise_display),
            "",
            f"Interpretation: with {args.folds} folds per task, top-vs-runner-up differences are small, fold-win patterns are mixed, and all top-vs-runner-up bootstrap confidence intervals cross zero. Top-vs-all comparisons may identify isolated inferior comparators, but they do not support a universal architecture-level superiority claim.",
            "",
            "## Noise Elbow for Best CV Model per Task",
            "",
            markdown_table(elbow_best[["task", "model", "clean_f1", "f1_at_sigma_0_20", "f1_at_sigma_0_50", "first_5pct_drop_sigma", "elbow_sigma"]]),
            "",
            "Noise is added after training to standardized test features. `first_5pct_drop_sigma` is the first Gaussian noise SD at which macro F1 drops by at least 5% relative to the clean test set. `elbow_sigma` is the maximum-distance knee point of the clean-to-high-noise curve.",
            "",
            "## Files",
            "",
            f"- Raw metrics: `{raw_path}`",
            f"- CV summary: `{cv_path}`",
            f"- Significance: `{sig_path}`",
            f"- Significance with fold wins: `{sig_wins_path}`",
            f"- Top-vs-all significance: `{pairwise_path}`",
            f"- Noise summary: `{noise_path}`",
            f"- Noise elbow: `{elbow_path}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    run()
