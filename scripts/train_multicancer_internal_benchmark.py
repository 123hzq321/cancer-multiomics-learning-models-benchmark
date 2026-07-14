from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crawl_tcga_brca_multiomics_cbioportal import fetch_clinical, page_get  # noqa: E402
from train_joined_metabric_small_liquid_external_validation import SmallModalityLiquidCfC  # noqa: E402
from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    EarlyFusionMLP,
    MODALITIES,
    ModalityLiquidCfC,
    MultiOmicsPreprocessor,
    compute_metrics,
    make_loaders,
    markdown_table,
    rows_from_result,
    save_predictions,
    set_seed,
    train_torch_model,
)


PROCESSED_DIR = Path("work/data/tcga_pancancer_cbioportal/processed")
OUT_DIR = Path("work/data/multicancer_internal_benchmark/processed")
OUTPUTS_DIR = Path("outputs")


TASKS = {
    "UCEC_molecular_subtype": {
        "study_id": "ucec_tcga_pan_can_atlas_2018",
        "cancer_type": "UCEC",
        "clinical_type": "PATIENT",
        "attrs": ["SUBTYPE"],
        "source_attr": "SUBTYPE",
        "description": "UCEC molecular subtype: CN_HIGH, CN_LOW, MSI, POLE.",
    },
    "COADREAD_molecular_subtype": {
        "study_id": "coadread_tcga_pan_can_atlas_2018",
        "cancer_type": "COADREAD",
        "clinical_type": "PATIENT",
        "attrs": ["SUBTYPE"],
        "source_attr": "SUBTYPE",
        "description": "COADREAD molecular subtype collapsed across colon/rectal labels: CIN, GS, MSI. Rare POLE classes are excluded.",
    },
    "HNSC_HPV_status": {
        "study_id": "hnsc_tcga_pan_can_atlas_2018",
        "cancer_type": "HNSC",
        "clinical_type": "PATIENT",
        "attrs": ["SUBTYPE"],
        "source_attr": "SUBTYPE",
        "description": "HNSC HPV status from PanCancer subtype labels.",
    },
    "KIRC_grade_binary": {
        "study_id": "kirc_tcga_pan_can_atlas_2018",
        "cancer_type": "KIRC",
        "clinical_type": "SAMPLE",
        "attrs": ["GRADE"],
        "source_attr": "GRADE",
        "description": "KIRC histologic grade, G1/G2 versus G3/G4.",
    },
    "PRAD_pathologic_T_stage": {
        "study_id": "prad_tcga_pan_can_atlas_2018",
        "cancer_type": "PRAD",
        "clinical_type": "PATIENT",
        "attrs": ["PATH_T_STAGE"],
        "source_attr": "PATH_T_STAGE",
        "description": "PRAD pathologic T stage, T2 versus T3/T4.",
    },
}


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def fetch_task_clinical(task_name: str, cfg: dict[str, object], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_name}_clinical.csv"
    if path.exists():
        return pd.read_csv(path)
    samples = pd.DataFrame(page_get(f"/studies/{cfg['study_id']}/samples", params={"projection": "SUMMARY"}, page_size=2000))
    primary = samples[samples["sampleType"].eq("Primary Solid Tumor")].copy()
    if primary.empty:
        primary = samples.copy()
    if cfg["clinical_type"] == "PATIENT":
        ids = sorted(primary["patientId"].dropna().unique())
    else:
        ids = primary["sampleId"].tolist()
    clinical = fetch_clinical(str(cfg["study_id"]), ids, list(cfg["attrs"]), str(cfg["clinical_type"]))
    clinical.to_csv(path, index=False)
    return clinical


def map_label(task_name: str, value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if task_name == "UCEC_molecular_subtype":
        mapping = {
            "UCEC_CN_HIGH": "CN_HIGH",
            "UCEC_CN_LOW": "CN_LOW",
            "UCEC_MSI": "MSI",
            "UCEC_POLE": "POLE",
        }
        return mapping.get(text)
    if task_name == "COADREAD_molecular_subtype":
        if text in {"COAD_CIN", "READ_CIN"}:
            return "CIN"
        if text in {"COAD_MSI", "READ_MSI"}:
            return "MSI"
        if text in {"COAD_GS", "READ_GS"}:
            return "GS"
        return None
    if task_name == "HNSC_HPV_status":
        mapping = {"HNSC_HPV+": "HPV_POS", "HNSC_HPV-": "HPV_NEG"}
        return mapping.get(text)
    if task_name == "KIRC_grade_binary":
        if text in {"G1", "G2"}:
            return "LOW_GRADE"
        if text in {"G3", "G4"}:
            return "HIGH_GRADE"
        return None
    if task_name == "PRAD_pathologic_T_stage":
        if text.startswith("T2"):
            return "T2"
        if text.startswith("T3") or text.startswith("T4"):
            return "T3_T4"
        return None
    raise KeyError(task_name)


def load_task_table(task_name: str, cfg: dict[str, object], pancancer: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{task_name}_table.csv"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    clinical = fetch_task_clinical(task_name, cfg, out_dir / "clinical")
    data = pancancer[pancancer["cancer_type"].eq(cfg["cancer_type"])].copy()
    source_attr = str(cfg["source_attr"])
    if cfg["clinical_type"] == "PATIENT":
        clinical = clinical.rename(columns={source_attr: "raw_task_label"})
        data = data.merge(clinical[["patientId", "raw_task_label"]], on="patientId", how="left")
    else:
        clinical = clinical.rename(columns={source_attr: "raw_task_label"})
        data = data.merge(clinical[["sampleId", "raw_task_label"]], on="sampleId", how="left")
    data["task"] = task_name
    data["task_label"] = data["raw_task_label"].map(lambda x: map_label(task_name, x))
    data = data[data["task_label"].notna()].copy()
    data.to_csv(path, index=False)
    return data


def make_task_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    labels = frame["task_label"].astype(str)
    counts = labels.value_counts()
    usable = frame[labels.isin(counts[counts >= 20].index)].copy()
    if usable["task_label"].nunique() < 2:
        raise ValueError("Need at least two classes with >=20 samples.")
    train_ids, temp_ids = train_test_split(
        usable["sampleId"],
        train_size=0.70,
        random_state=seed,
        stratify=usable["task_label"],
    )
    temp = usable.set_index("sampleId").loc[temp_ids].reset_index()
    valid_ids, test_ids = train_test_split(
        temp["sampleId"],
        train_size=0.50,
        random_state=seed,
        stratify=temp["task_label"],
    )
    splits = pd.DataFrame(
        {
            "sampleId": list(train_ids) + list(valid_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["valid"] * len(valid_ids) + ["test"] * len(test_ids),
        }
    )
    return usable.merge(splits, on="sampleId", how="inner")


def modality_columns(frame: pd.DataFrame, modalities: list[str]) -> dict[str, list[str]]:
    cols = {m: sorted([col for col in frame.columns if col.startswith(f"{m}__")]) for m in modalities}
    missing = [m for m, c in cols.items() if not c]
    if missing:
        raise ValueError(f"Missing modality columns: {missing}")
    return cols


def build_processed(data: pd.DataFrame, modality_cols: dict[str, list[str]]) -> tuple[dict[str, dict[str, object]], LabelEncoder]:
    label_encoder = LabelEncoder()
    data = data.copy()
    data["target"] = label_encoder.fit_transform(data["task_label"].astype(str))
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
    return processed, label_encoder


def fit_sklearn_models(processed: dict[str, dict[str, object]], seed: int, num_classes: int) -> list[dict[str, object]]:
    models = [
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
    rows = []
    for model_name, group, model in models:
        print(f"    sklearn {model_name}", flush=True)
        if model_name == "hist_gradient_boosting":
            y = processed["train"]["y"]
            counts = np.bincount(y, minlength=num_classes)
            weights = len(y) / (num_classes * np.maximum(counts, 1))
            model.fit(processed["train"]["fusion"], y, sample_weight=weights[y])
        else:
            model.fit(processed["train"]["fusion"], processed["train"]["y"])
        metrics, predictions = {}, {}
        for split, payload in processed.items():
            proba = model.predict_proba(payload["fusion"])
            metrics[split] = compute_metrics(payload["y"], proba)
            predictions[split] = (payload["y"], proba)
        rows.append(
            {
                "model": model_name,
                "model_group": group,
                "metrics": metrics,
                "predictions": predictions,
                "best_epoch": np.nan,
                "best_validation_score": metrics["valid"]["f1_macro"],
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy()
    return (
        test.groupby(["task", "model", "model_group"])
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
        .sort_values(["task", "f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=[True, False, False])
    )


def write_report(path: Path, summary: pd.DataFrame, task_counts: pd.DataFrame, best_by_task: pd.DataFrame) -> None:
    lines = [
        "# Multi-Cancer Internal Task Benchmark v0",
        "",
        "## Scope",
        "",
        "Extend the BRCA-focused benchmark with cancer-internal tasks from TCGA PanCancer Atlas cohorts.",
        "",
        "Tasks:",
        "",
    ]
    for task_name, cfg in TASKS.items():
        lines.append(f"- `{task_name}`: {cfg['description']}")
    lines.extend(
        [
            "",
            "Models: elastic-net logistic regression, ExtraTrees, HistGradientBoosting, MLP, Liquid/CfC, and small-Liquid/CfC.",
            "",
            "## Task Label Counts",
            "",
            markdown_table(task_counts),
            "",
            "## Test Summary",
            "",
            markdown_table(summary, floatfmt=".4f"),
            "",
            "## Best Model per Task",
            "",
            markdown_table(best_by_task, floatfmt=".4f"),
            "",
            "## Interpretation",
            "",
            "This benchmark is internal-split only. It should be interpreted as a multi-cancer task suite that complements the deeper BRCA external-validation study. External validation per cancer remains a future extension.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(PROCESSED_DIR / "pancancer_multimodal_impact468_table.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "multicancer_internal_benchmark_report_v0.md"))
    parser.add_argument("--results-csv", default=str(OUT_DIR / "multicancer_internal_benchmark_results.csv"))
    parser.add_argument("--summary-csv", default=str(OUT_DIR / "multicancer_internal_benchmark_summary.csv"))
    parser.add_argument("--predictions-csv", default=str(OUT_DIR / "multicancer_internal_benchmark_predictions.csv"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--modalities",
        default=",".join(MODALITIES),
        help="Comma-separated modality prefixes to use. Defaults to the original full multi-omics set.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    device = torch.device(args.device)
    selected_modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    if not selected_modalities:
        raise ValueError("At least one modality must be selected.")

    pancancer = pd.read_csv(args.table, low_memory=False)
    all_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    task_count_rows = []

    for task_name, cfg in TASKS.items():
        print(f"\nTask {task_name}", flush=True)
        task_table = load_task_table(task_name, cfg, pancancer, out_dir)
        task_counts = task_table["task_label"].value_counts().rename_axis("label").reset_index(name="n")
        task_counts.insert(0, "task", task_name)
        task_count_rows.extend(task_counts.to_dict(orient="records"))
        modality_cols = modality_columns(task_table, selected_modalities)
        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            set_seed(seed)
            split_data = make_task_splits(task_table, seed)
            split_data.to_csv(out_dir / f"{task_name}_splits_seed{seed}.csv", index=False)
            processed, label_encoder = build_processed(split_data, modality_cols)
            num_classes = len(label_encoder.classes_)
            fusion_dim = processed["train"]["fusion"].shape[1]
            input_dims = [processed["train"]["modalities"][idx].shape[1] for idx in range(len(selected_modalities))]

            results = fit_sklearn_models(processed, seed, num_classes)
            loaders = make_loaders(processed, args.batch_size)
            torch_models = [
                ("mlp_early_fusion", "static_nn", EarlyFusionMLP(fusion_dim, num_classes, hidden_dim=128)),
                ("liquid_cfc_modality_sequence", "liquid_nn", ModalityLiquidCfC(input_dims, num_classes, embed_dim=64, hidden_dim=96)),
                ("small_liquid_cfc_modality_sequence", "small_liquid_nn", SmallModalityLiquidCfC(input_dims, num_classes)),
            ]
            for model_name, group, model in torch_models:
                print(f"    torch {model_name}", flush=True)
                set_seed(seed)
                result = train_torch_model(
                    model,
                    loaders,
                    processed["train"]["y"],
                    num_classes,
                    device=device,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    max_epochs=args.epochs,
                    patience=args.patience,
                )
                result["model"] = model_name
                result["model_group"] = group
                results.append({"model": model_name, "model_group": group, **result})

            for result in results:
                rows = rows_from_result(seed, result["model"], result["model_group"], result)
                for row in rows:
                    row["task"] = task_name
                    row["classes"] = "|".join(label_encoder.classes_)
                    row["modalities"] = ",".join(selected_modalities)
                all_rows.extend(rows)
                for split, (y_true, proba) in result["predictions"].items():
                    pred_rows.append(
                        {
                            "seed": seed,
                            "model": f"{task_name}__{result['model']}",
                            "split": split,
                            "sample_ids": processed[split]["sampleId"],
                            "y_true": y_true,
                            "proba": proba,
                        }
                    )

            # Save predictions per task/seed because class mappings differ across tasks.
            save_predictions(
                [row for row in pred_rows if row["model"].startswith(f"{task_name}__") and row["seed"] == seed],
                out_dir / f"{task_name}_predictions_seed{seed}.csv",
                label_encoder,
            )

    results_df = pd.DataFrame(all_rows)
    summary = summarize(results_df)
    task_counts_df = pd.DataFrame(task_count_rows)
    best_by_task = summary.sort_values(["task", "f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=[True, False, False]).groupby("task").head(1)
    results_df.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    task_counts_df.to_csv(out_dir / "multicancer_internal_task_label_counts.csv", index=False)
    best_by_task.to_csv(out_dir / "multicancer_internal_best_by_task.csv", index=False)
    write_report(Path(args.report), summary, task_counts_df, best_by_task)
    print(f"Done. Report: {args.report}")
    print(best_by_task.to_string(index=False))


if __name__ == "__main__":
    main()
