from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_joined_metabric_small_liquid_external_validation import JOINED_DIR  # noqa: E402
from train_metabric_marker_panel_external_validation import OUTPUTS_DIR, parse_ints  # noqa: E402
from train_tcga_brca_multiomics_baselines_vs_liquid import compute_metrics, markdown_table, save_predictions  # noqa: E402


OUT_DIR = Path("work/data/large_scale_baselines/processed")


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def class_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    weights = len(y) / (len(counts) * np.maximum(counts, 1))
    return weights[y]


class MatrixPreprocessor:
    def __init__(self, cols: list[str]):
        self.cols = cols
        try:
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        except TypeError:
            self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> "MatrixPreprocessor":
        x = frame[self.cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        x = self.imputer.fit_transform(x)
        self.scaler.fit(x)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        return x.astype(np.float32)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    joined = pd.read_csv(JOINED_DIR / "tcga_metabric_joined_training_table.csv", low_memory=False)
    scanb = pd.read_csv(JOINED_DIR / "scanb_gse96058_marker_mrna_external_table.csv", low_memory=False)
    smc = pd.read_csv(JOINED_DIR / "smc_2018_marker_external_table.csv", low_memory=False)
    cptac = pd.read_csv(JOINED_DIR / "cptac_2020_marker_external_table.csv", low_memory=False)
    mrna_cols = sorted([col for col in joined.columns if col.startswith("mrna_z__")])
    for frame in [scanb, smc, cptac]:
        for col in mrna_cols:
            if col not in frame.columns:
                frame[col] = np.nan
    return joined, scanb, smc, cptac, mrna_cols


def make_training_scenarios(
    joined: pd.DataFrame,
    scanb: pd.DataFrame,
    smc: pd.DataFrame,
    seed: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    joined_clean = joined[joined["subtype_label"].notna()].copy()
    joined_train = joined_clean[joined_clean["joint_split"].eq("train")].copy()
    joined_valid = joined_clean[joined_clean["joint_split"].eq("valid")].copy()

    expanded_parts = []
    for source, frame in [("TCGA_METABRIC", joined_clean), ("SCANB", scanb), ("SMC_2018", smc)]:
        tmp = frame[frame["subtype_label"].notna()].copy()
        tmp["training_source"] = source
        expanded_parts.append(tmp)
    expanded = pd.concat(expanded_parts, axis=0, ignore_index=True, sort=False)
    train_idx, valid_idx = train_test_split(
        np.arange(expanded.shape[0]),
        train_size=0.85,
        random_state=seed,
        stratify=expanded["subtype_label"].astype(str),
    )
    expanded_train = expanded.iloc[train_idx].copy()
    expanded_valid = expanded.iloc[valid_idx].copy()
    return {
        "joined_tcga_metabric": {"train": joined_train, "valid": joined_valid},
        "expanded_tcga_metabric_scanb_smc": {"train": expanded_train, "valid": expanded_valid},
    }


def make_models(seed: int) -> list[tuple[str, str, object, bool]]:
    return [
        (
            "logistic_l2",
            "linear",
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed),
            False,
        ),
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
            False,
        ),
        (
            "ridge_classifier",
            "linear",
            RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=seed),
            False,
        ),
        (
            "linear_svm_calibrated",
            "linear_svm",
            CalibratedClassifierCV(
                estimator=LinearSVC(C=0.5, class_weight="balanced", dual="auto", max_iter=10000, random_state=seed),
                cv=3,
            ),
            False,
        ),
        (
            "lda_shrinkage",
            "linear_discriminant",
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            False,
        ),
        ("gaussian_nb", "probabilistic", GaussianNB(), False),
        ("knn_distance", "instance_based", KNeighborsClassifier(n_neighbors=25, weights="distance"), False),
        (
            "random_forest",
            "tree_ensemble",
            RandomForestClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        (
            "extra_trees",
            "tree_ensemble",
            ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        (
            "hist_gradient_boosting",
            "boosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.04,
                l2_regularization=0.05,
                max_leaf_nodes=15,
                random_state=seed,
            ),
            True,
        ),
        (
            "sklearn_mlp",
            "static_nn",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                max_iter=400,
                random_state=seed,
            ),
            False,
        ),
    ]


def predict_proba_any(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    if hasattr(model, "decision_function"):
        return softmax(model.decision_function(x))
    pred = model.predict(x)
    classes = getattr(model, "classes_", np.unique(pred))
    proba = np.zeros((len(pred), len(classes)), dtype=np.float32)
    for idx, label in enumerate(pred):
        class_idx = int(np.where(classes == label)[0][0])
        proba[idx, class_idx] = 1.0
    return proba


def fit_and_eval(
    *,
    scenario_name: str,
    seed: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    external: pd.DataFrame,
    mrna_cols: list[str],
    label_encoder: LabelEncoder,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pre = MatrixPreprocessor(mrna_cols).fit(train)
    frames = {"train": train, "valid": valid, "cptac_external": external}
    arrays = {split: pre.transform(frame) for split, frame in frames.items()}
    ys = {split: label_encoder.transform(frame["subtype_label"].astype(str)) for split, frame in frames.items()}
    sample_ids = {split: frame["sampleId"].astype(str).to_numpy() for split, frame in frames.items()}

    rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for model_name, group, model, needs_weight in make_models(seed):
        print(f"{scenario_name} seed={seed} model={model_name}", flush=True)
        if needs_weight:
            model.fit(arrays["train"], ys["train"], sample_weight=class_sample_weight(ys["train"]))
        else:
            model.fit(arrays["train"], ys["train"])
        for split in ["train", "valid", "cptac_external"]:
            proba = predict_proba_any(model, arrays[split])
            metrics = compute_metrics(ys[split], proba)
            row = {
                "training_scale": scenario_name,
                "seed": seed,
                "model": model_name,
                "model_group": group,
                "split": split,
            }
            row.update(metrics)
            rows.append(row)
            pred_rows.append(
                {
                    "seed": seed,
                    "model": f"{scenario_name}__{model_name}",
                    "split": split,
                    "sample_ids": sample_ids[split],
                    "y_true": ys[split],
                    "proba": proba,
                }
            )
    return rows, pred_rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["training_scale", "split", "model", "model_group"])
        .agg(
            runs=("seed", "nunique"),
            n_mean=("n", "mean"),
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
        .sort_values(["split", "training_scale", "f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=[True, True, False, False])
    )


def write_report(
    path: Path,
    summary: pd.DataFrame,
    scenario_counts: pd.DataFrame,
    best_confusion: pd.DataFrame,
    best_report: pd.DataFrame,
) -> None:
    external_summary = summary[summary["split"].eq("cptac_external")].copy()
    lines = [
        "# Large-Scale mRNA Baseline Expansion v0",
        "",
        "## Scope",
        "",
        "Expand training scale for mRNA marker subtype prediction by adding SCAN-B and SMC 2018 to the TCGA+METABRIC training pool, while keeping CPTAC 2020 as the independent external validation cohort.",
        "",
        "## Training Scale Counts",
        "",
        markdown_table(scenario_counts),
        "",
        "## CPTAC External Summary",
        "",
        markdown_table(external_summary, floatfmt=".4f"),
        "",
        "## Best CPTAC External Confusion Matrix",
        "",
        markdown_table(best_confusion),
        "",
        "## Best CPTAC External Classification Report",
        "",
        markdown_table(best_report, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "This experiment tests whether increased sample size changes the ranking of classical baselines. It is intentionally mRNA-only because SCAN-B and SMC do not provide the same complete CNA/mutation multimodal feature set as CPTAC.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--results-csv", default=str(OUT_DIR / "large_scale_mrna_baseline_results.csv"))
    parser.add_argument("--summary-csv", default=str(OUT_DIR / "large_scale_mrna_baseline_summary.csv"))
    parser.add_argument("--predictions-csv", default=str(OUT_DIR / "large_scale_mrna_baseline_predictions.csv"))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "large_scale_mrna_baseline_report_v0.md"))
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    joined, scanb, smc, cptac, mrna_cols = load_tables()
    cptac = cptac[cptac["subtype_label"].notna()].copy()
    all_train_for_labels = pd.concat([joined, scanb, smc], axis=0, ignore_index=True, sort=False)
    label_encoder = LabelEncoder().fit(all_train_for_labels["subtype_label"].dropna().astype(str))
    cptac = cptac[cptac["subtype_label"].isin(label_encoder.classes_)].copy()

    all_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    scenario_count_rows = []
    for seed in parse_ints(args.seeds):
        scenarios = make_training_scenarios(joined, scanb, smc, seed)
        for scenario_name, payload in scenarios.items():
            if seed == parse_ints(args.seeds)[0]:
                for split_name, frame in payload.items():
                    counts = frame["subtype_label"].value_counts().rename_axis("subtype_label").reset_index(name="n")
                    counts.insert(0, "split", split_name)
                    counts.insert(0, "training_scale", scenario_name)
                    scenario_count_rows.extend(counts.to_dict(orient="records"))
            rows, preds = fit_and_eval(
                scenario_name=scenario_name,
                seed=seed,
                train=payload["train"],
                valid=payload["valid"],
                external=cptac,
                mrna_cols=mrna_cols,
                label_encoder=label_encoder,
            )
            all_rows.extend(rows)
            pred_rows.extend(preds)

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    scenario_counts = pd.DataFrame(scenario_count_rows)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    save_predictions(pred_rows, Path(args.predictions_csv), label_encoder)

    external = summary[summary["split"].eq("cptac_external")].sort_values(["f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=False)
    best = external.iloc[0]
    best_model = f"{best['training_scale']}__{best['model']}"
    best_seed = (
        results[results["split"].eq("cptac_external") & results["training_scale"].eq(best["training_scale"]) & results["model"].eq(best["model"])]
        .sort_values("f1_macro", ascending=False)
        .iloc[0]["seed"]
    )
    pred = pd.read_csv(args.predictions_csv)
    best_pred = pred[pred["split"].eq("cptac_external") & pred["model"].eq(best_model) & pred["seed"].eq(best_seed)]
    labels = list(label_encoder.classes_)
    cm = pd.DataFrame(
        confusion_matrix(best_pred["true_label"], best_pred["pred_label"], labels=labels),
        index=[f"true__{x}" for x in labels],
        columns=[f"pred__{x}" for x in labels],
    )
    cls = pd.DataFrame(classification_report(best_pred["true_label"], best_pred["pred_label"], output_dict=True, zero_division=0)).T.reset_index(names="label")
    cm.to_csv(out_dir / "large_scale_mrna_baseline_best_confusion_matrix.csv")
    cls.to_csv(out_dir / "large_scale_mrna_baseline_best_classification_report.csv", index=False)
    scenario_counts.to_csv(out_dir / "large_scale_mrna_training_counts.csv", index=False)
    write_report(Path(args.report), summary, scenario_counts, cm.reset_index(names="true_label"), cls)
    print(f"Done. Report: {args.report}")
    print(summary[summary["split"].eq("cptac_external")].to_string(index=False))


if __name__ == "__main__":
    main()
