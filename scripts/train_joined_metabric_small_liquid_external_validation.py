from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crawl_tcga_brca_multiomics_cbioportal import (  # noqa: E402
    fetch_clinical,
    fetch_molecular_matrix,
    fetch_mutation_matrix,
    page_get,
)
from train_metabric_marker_panel_external_validation import (  # noqa: E402
    BREAST_MARKERS,
    DATA_DIR,
    OUTPUTS_DIR,
    add_prefix,
    map_marker_genes,
    parse_ints,
)
from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    EarlyFusionMLP,
    ModalityLiquidCfC,
    MultiOmicsDataset,
    compute_metrics,
    markdown_table,
    rows_from_result,
    save_predictions,
    set_seed,
    train_torch_model,
)


SCANB_DIR = Path("work/data/gse96058_scanb")
JOINED_DIR = Path("work/data/joined_metabric_external_validation/processed")

CBIO_EXTERNALS = {
    "smc_2018": {
        "study_id": "brca_smc_2018",
        "label_attr": "PAM50_SUBTYPE",
        "label_map": {"LuminalA": "LumA", "LuminalB": "LumB", "Basal": "Basal", "Her2": "Her2", "Normal": "Normal"},
        "profiles": {
            "mrna_z": "brca_smc_2018_mrna_seq_tpm_all_sample_Zscores",
            "mutation": "brca_smc_2018_mutations",
        },
    },
    "cptac_2020": {
        "study_id": "brca_cptac_2020",
        "label_attr": "PAM50",
        "label_map": {"LumA": "LumA", "LumB": "LumB", "Basal": "Basal", "Her2": "Her2", "Normal-like": "Normal"},
        "profiles": {
            "mrna_z": "brca_cptac_2020_mrna_median_Zscores",
            "cna": "brca_cptac_2020_gistic",
            "mutation": "brca_cptac_2020_mutations",
        },
    },
}


class SmallModalityLiquidCfC(ModalityLiquidCfC):
    def __init__(self, input_dims: list[int], num_classes: int):
        super().__init__(input_dims, num_classes, embed_dim=32, hidden_dim=48)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class JoinedPreprocessor:
    def __init__(self, modality_cols: dict[str, list[str]]):
        self.modality_cols = modality_cols
        self.imputers: dict[str, SimpleImputer] = {}
        self.scalers: dict[str, StandardScaler] = {}

    def fit(self, frame: pd.DataFrame) -> "JoinedPreprocessor":
        for modality, cols in self.modality_cols.items():
            x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            try:
                imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            except TypeError:
                imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x = imputer.fit_transform(x)
            scaler.fit(x)
            self.imputers[modality] = imputer
            self.scalers[modality] = scaler
        return self

    def transform(self, frame: pd.DataFrame) -> list[np.ndarray]:
        arrays = []
        for modality, cols in self.modality_cols.items():
            x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            x = self.imputers[modality].transform(x)
            x = self.scalers[modality].transform(x)
            arrays.append(x.astype(np.float32))
        return arrays


def make_joint_training_table(marker_table: pd.DataFrame, seed: int, out_dir: Path) -> pd.DataFrame:
    frame = marker_table[marker_table["split"].isin(["train", "valid", "tcga_test", "metabric_external"])].copy()
    frame = frame[frame["subtype_label"].notna()].copy()
    frame["joint_source"] = np.where(frame["split"].eq("metabric_external"), "METABRIC", "TCGA")
    frame["joint_split"] = "train"
    train_idx, valid_idx = train_test_split(
        frame.index.to_numpy(),
        train_size=0.85,
        random_state=seed,
        stratify=frame["subtype_label"].astype(str),
    )
    frame.loc[valid_idx, "joint_split"] = "valid"
    frame.to_csv(out_dir / "tcga_metabric_joined_training_table.csv", index=False)
    return frame


def cohort_zscore(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    x = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    means = x.mean(axis=0)
    sds = x.std(axis=0).replace(0, np.nan)
    out[feature_cols] = ((x - means) / sds).fillna(0.0).astype(np.float32)
    return out


def extract_scanb_marker_table(out_dir: Path) -> pd.DataFrame:
    out_path = out_dir / "scanb_gse96058_marker_mrna_external_table.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    samp = pyreadr.read_r(str(SCANB_DIR / "GSE96058_sampDesc.rda"))["GSE96058_sampDesc"].copy()
    samp = samp.reset_index(names="geoRow")
    samp = samp[~samp["isRepl"].astype(bool)].copy()
    label_map = {"LumA": "LumA", "LumB": "LumB", "Basal": "Basal", "Her2": "Her2", "Normal": "Normal"}
    samp["subtype_label"] = samp["pam50_subtype"].map(label_map)
    samp = samp[samp["subtype_label"].notna()].copy()

    markers = list(dict.fromkeys(BREAST_MARKERS))
    collected: list[pd.DataFrame] = []
    for idx in range(1, 6):
        object_name = f"GSE96058_geneExpression_sub{idx}"
        path = SCANB_DIR / f"{object_name}.rda"
        print(f"Reading SCAN-B {object_name}")
        expr = pyreadr.read_r(str(path))[object_name]
        present = [gene for gene in markers if gene in expr.index]
        missing = [gene for gene in markers if gene not in expr.index]
        if missing and idx == 1:
            print(f"SCAN-B missing marker genes: {missing}")
        block = expr.loc[present].T
        block.index.name = "title"
        block = block.reindex(columns=markers)
        collected.append(block)
    expression = pd.concat(collected, axis=0)
    expression = expression[~expression.index.duplicated(keep="first")]
    expression = expression.add_prefix("mrna_z__").reset_index()
    table = samp[["title", "geoAcc", "subtype_label", "pam50_subtype", "countTable"]].merge(expression, on="title", how="inner")
    table = table.rename(columns={"title": "sampleId", "geoAcc": "source_sample_id"})
    table["source_study_id"] = "GSE96058_SCANB"
    table["split"] = "scanb_external"
    feature_cols = [col for col in table.columns if col.startswith("mrna_z__")]
    table = cohort_zscore(table, feature_cols)
    table.to_csv(out_path, index=False)
    return table


def read_or_fetch_cbio_external(name: str, genes: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cfg = CBIO_EXTERNALS[name]
    out_path = out_dir / f"{name}_marker_external_table.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    study_id = cfg["study_id"]
    print(f"Fetching {study_id}")
    samples = pd.DataFrame(page_get(f"/studies/{study_id}/samples", params={"projection": "SUMMARY"}, page_size=3000))
    primary = samples[samples["sampleType"].eq("Primary Solid Tumor")].copy()
    if primary.empty:
        primary = samples.copy()
    sample_ids = primary["sampleId"].tolist()
    clinical = fetch_clinical(study_id, sample_ids, [cfg["label_attr"]], "SAMPLE")
    clinical = clinical.rename(columns={cfg["label_attr"]: "raw_subtype"})
    clinical["subtype_label"] = clinical["raw_subtype"].map(cfg["label_map"])
    table = primary[["sampleId", "patientId", "sampleType"]].merge(clinical, on="sampleId", how="left")
    table["source_study_id"] = study_id
    table["split"] = f"{name}_external"

    matrices = {}
    for modality, profile in cfg["profiles"].items():
        cache = out_dir / f"{name}_{modality}_marker_matrix.csv"
        if cache.exists():
            matrix = pd.read_csv(cache, index_col=0).reindex(index=sample_ids, columns=genes["hugoGeneSymbol"].tolist())
        elif modality == "mutation":
            matrix = fetch_mutation_matrix(profile, sample_ids, genes).reindex(index=sample_ids, columns=genes["hugoGeneSymbol"].tolist())
            matrix = matrix.fillna(0).astype(np.int8)
            matrix.to_csv(cache)
        else:
            matrix = fetch_molecular_matrix(profile, sample_ids, genes).reindex(index=sample_ids, columns=genes["hugoGeneSymbol"].tolist())
            matrix.to_csv(cache)
        matrices[modality] = matrix

    features = pd.concat([add_prefix(matrices[m], m) for m in matrices], axis=1).reset_index()
    table = table.merge(features, on="sampleId", how="left")
    table = table[table["subtype_label"].notna()].copy()
    table.to_csv(out_path, index=False)
    return table


def align_external_columns(external: pd.DataFrame, selected: dict[str, list[str]]) -> pd.DataFrame:
    out = external.copy()
    for cols in selected.values():
        for col in cols:
            if col not in out.columns:
                out[col] = np.nan
    return out


def make_processed(
    train_valid: pd.DataFrame,
    externals: dict[str, pd.DataFrame],
    selected: dict[str, list[str]],
    label_encoder: LabelEncoder,
) -> dict[str, dict[str, object]]:
    pre = JoinedPreprocessor(selected).fit(train_valid[train_valid["joint_split"].eq("train")])
    processed: dict[str, dict[str, object]] = {}
    for split_name, frame in [
        ("train", train_valid[train_valid["joint_split"].eq("train")].copy()),
        ("valid", train_valid[train_valid["joint_split"].eq("valid")].copy()),
    ]:
        mods = pre.transform(frame)
        processed[split_name] = {
            "frame": frame,
            "modalities": mods,
            "fusion": np.concatenate(mods, axis=1).astype(np.float32),
            "y": label_encoder.transform(frame["subtype_label"].astype(str)),
            "sampleId": frame["sampleId"].to_numpy(),
        }
    for name, frame in externals.items():
        frame = align_external_columns(frame[frame["subtype_label"].isin(label_encoder.classes_)].copy(), selected)
        mods = pre.transform(frame)
        processed[name] = {
            "frame": frame,
            "modalities": mods,
            "fusion": np.concatenate(mods, axis=1).astype(np.float32),
            "y": label_encoder.transform(frame["subtype_label"].astype(str)),
            "sampleId": frame["sampleId"].to_numpy(),
        }
    return processed


def loader_dict(processed: dict[str, dict[str, object]], batch_size: int) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            MultiOmicsDataset(payload["fusion"], payload["modalities"], payload["y"]),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        for split, payload in processed.items()
    }


def fit_sklearn(processed: dict[str, dict[str, object]], seed: int, model_prefix: str) -> list[dict[str, object]]:
    models = [
        (
            f"{model_prefix}logistic",
            "linear_baseline",
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed),
        ),
        (
            f"{model_prefix}extra_trees",
            "tree_baseline",
            ExtraTreesClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed),
        ),
    ]
    rows = []
    for name, group, model in models:
        model.fit(processed["train"]["fusion"], processed["train"]["y"])
        metrics, predictions = {}, {}
        for split, payload in processed.items():
            proba = model.predict_proba(payload["fusion"])
            metrics[split] = compute_metrics(payload["y"], proba)
            predictions[split] = (payload["y"], proba)
        rows.append(
            {
                "model": name,
                "model_group": group,
                "metrics": metrics,
                "predictions": predictions,
                "best_epoch": np.nan,
                "best_validation_score": metrics["valid"]["roc_auc_ovr_macro"],
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for split in sorted(results["split"].unique()):
        tmp = results[results["split"].eq(split)].copy()
        s = (
            tmp.groupby(["feature_set", "model", "model_group"])
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
        )
        s.insert(0, "evaluation_split", split)
        frames.append(s)
    return pd.concat(frames, axis=0, ignore_index=True).sort_values(
        ["evaluation_split", "f1_macro_mean", "roc_auc_ovr_macro_mean"],
        ascending=[True, False, False],
    )


def add_feature_set(rows: list[dict[str, object]], feature_set: str, param_count: int | float = np.nan) -> list[dict[str, object]]:
    for row in rows:
        row["feature_set"] = feature_set
        row["trainable_parameters"] = param_count
    return rows


def write_report(
    path: Path,
    summary: pd.DataFrame,
    train_valid: pd.DataFrame,
    externals: dict[str, pd.DataFrame],
    params: pd.DataFrame,
    best_model: str,
    best_split: str,
    best_confusion: pd.DataFrame,
    best_report: pd.DataFrame,
) -> None:
    train_counts = train_valid.groupby(["joint_split", "joint_source", "subtype_label"]).size().reset_index(name="n")
    external_counts = pd.concat(
        [
            frame["subtype_label"].value_counts().rename_axis("subtype_label").reset_index(name="n").assign(external=name)
            for name, frame in externals.items()
        ],
        axis=0,
        ignore_index=True,
    )
    external_summary = summary[summary["evaluation_split"].str.endswith("_external")].copy()
    lines = [
        "# TCGA+METABRIC Joined Training with New External Validation v0",
        "",
        "## Scope",
        "",
        "METABRIC is added to the training/validation pool. Independent external validation is performed on newly added cohorts.",
        "",
        "- Joined training pool: TCGA BRCA + METABRIC marker panel samples.",
        "- New external cohorts: SMC 2018, CPTAC 2020, and GSE96058/SCAN-B.",
        "- Main feature sets: marker mRNA and marker multimodal expression-last.",
        "- Models include original Liquid/CfC and small-Liquid/CfC.",
        "",
        "## Trainable Parameters",
        "",
        markdown_table(params),
        "",
        "## Joined Training Pool Counts",
        "",
        markdown_table(train_counts),
        "",
        "## External Cohort Counts",
        "",
        markdown_table(external_counts),
        "",
        "## External Summary",
        "",
        markdown_table(external_summary, floatfmt=".4f"),
        "",
        f"## Best External Model: `{best_model}` on `{best_split}`",
        "",
        "### Confusion Matrix",
        "",
        markdown_table(best_confusion),
        "",
        "### Classification Report",
        "",
        markdown_table(best_report, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "This analysis answers whether increasing the training data by adding METABRIC and shrinking the Liquid/CfC model improves external transfer. "
        "Because SCAN-B and SMC are mRNA-only external validations, multimodal conclusions should rely mainly on CPTAC and internal validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(JOINED_DIR))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "tcga_metabric_joined_small_liquid_external_validation_report_v0.md"))
    parser.add_argument("--summary-csv", default=str(JOINED_DIR / "joined_small_liquid_external_model_summary.csv"))
    parser.add_argument("--results-csv", default=str(JOINED_DIR / "joined_small_liquid_external_model_results.csv"))
    parser.add_argument("--predictions-csv", default=str(JOINED_DIR / "joined_small_liquid_external_model_predictions.csv"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    device = torch.device(args.device)

    marker_table = pd.read_csv(DATA_DIR / "breast_marker_panel" / "combined_breast_marker_external_table.csv", low_memory=False)
    train_valid = make_joint_training_table(marker_table, args.split_seed, out_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_valid["subtype_label"].astype(str))

    genes = map_marker_genes(DATA_DIR / "breast_marker_panel")
    externals = {
        "smc_2018_external": read_or_fetch_cbio_external("smc_2018", genes, out_dir),
        "cptac_2020_external": read_or_fetch_cbio_external("cptac_2020", genes, out_dir),
        "scanb_external": extract_scanb_marker_table(out_dir),
    }

    all_mrna_cols = [col for col in marker_table.columns if col.startswith("mrna_z__")]
    all_cna_cols = [col for col in marker_table.columns if col.startswith("cna__")]
    all_mut_cols = [col for col in marker_table.columns if col.startswith("mutation__")]
    feature_sets = {
        "joined_marker_mrna": {
            "selected": {"mrna_z": all_mrna_cols},
            "external_names": ["smc_2018_external", "cptac_2020_external", "scanb_external"],
        },
        "joined_marker_multimodal_expression_last": {
            "selected": {"cna": all_cna_cols, "mutation": all_mut_cols, "mrna_z": all_mrna_cols},
            "external_names": ["cptac_2020_external"],
        },
    }

    all_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []

    for feature_set, config in feature_sets.items():
        selected = config["selected"]
        selected_externals = {name: externals[name] for name in config["external_names"]}
        processed = make_processed(train_valid, selected_externals, selected, label_encoder)
        input_dims = [processed["train"]["modalities"][idx].shape[1] for idx in range(len(selected))]
        fusion_dim = processed["train"]["fusion"].shape[1]
        loaders = loader_dict(processed, args.batch_size)

        model_param_examples = [
            (f"{feature_set}__mlp", EarlyFusionMLP(fusion_dim, len(label_encoder.classes_), hidden_dim=128)),
            (f"{feature_set}__liquid_cfc", ModalityLiquidCfC(input_dims, len(label_encoder.classes_), embed_dim=64, hidden_dim=96)),
            (f"{feature_set}__small_liquid_cfc", SmallModalityLiquidCfC(input_dims, len(label_encoder.classes_))),
        ]
        for model_name, model in model_param_examples:
            param_rows.append({"feature_set": feature_set, "model": model_name, "trainable_parameters": count_parameters(model)})

        for seed in seeds:
            set_seed(seed)
            print(f"{feature_set} seed={seed} sklearn")
            for result in fit_sklearn(processed, seed, model_prefix=f"{feature_set}__"):
                rows = add_feature_set(rows_from_result(seed, result["model"], result["model_group"], result), feature_set)
                all_rows.extend(rows)
                for split, (y_true, proba) in result["predictions"].items():
                    prediction_rows.append({"seed": seed, "model": result["model"], "split": split, "sample_ids": processed[split]["sampleId"], "y_true": y_true, "proba": proba})

            torch_models = [
                (f"{feature_set}__mlp", "static_nn", EarlyFusionMLP(fusion_dim, len(label_encoder.classes_), hidden_dim=128)),
                (f"{feature_set}__liquid_cfc", "liquid_nn", ModalityLiquidCfC(input_dims, len(label_encoder.classes_), embed_dim=64, hidden_dim=96)),
                (f"{feature_set}__small_liquid_cfc", "small_liquid_nn", SmallModalityLiquidCfC(input_dims, len(label_encoder.classes_))),
            ]
            for model_name, group, model in torch_models:
                set_seed(seed)
                param_count = count_parameters(model)
                print(f"{feature_set} seed={seed} {model_name} params={param_count}")
                result = train_torch_model(
                    model,
                    loaders,
                    processed["train"]["y"],
                    len(label_encoder.classes_),
                    device=device,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    max_epochs=args.epochs,
                    patience=args.patience,
                )
                rows = add_feature_set(rows_from_result(seed, model_name, group, result), feature_set, param_count)
                all_rows.extend(rows)
                for split, (y_true, proba) in result["predictions"].items():
                    prediction_rows.append({"seed": seed, "model": model_name, "split": split, "sample_ids": processed[split]["sampleId"], "y_true": y_true, "proba": proba})

    results = pd.DataFrame(all_rows)
    params = pd.DataFrame(param_rows).drop_duplicates()
    summary = summarize(results)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    save_predictions(prediction_rows, Path(args.predictions_csv), label_encoder)

    external = summary[summary["evaluation_split"].str.endswith("_external")].sort_values(
        ["f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=False
    )
    best = external.iloc[0]
    best_model = str(best["model"])
    best_split = str(best["evaluation_split"])
    best_seed = (
        results[results["split"].eq(best_split) & results["model"].eq(best_model)]
        .sort_values("f1_macro", ascending=False)
        .iloc[0]["seed"]
    )
    pred = pd.read_csv(args.predictions_csv)
    best_pred = pred[pred["split"].eq(best_split) & pred["model"].eq(best_model) & pred["seed"].eq(best_seed)]
    labels = list(label_encoder.classes_)
    cm = pd.DataFrame(
        confusion_matrix(best_pred["true_label"], best_pred["pred_label"], labels=labels),
        index=[f"true__{x}" for x in labels],
        columns=[f"pred__{x}" for x in labels],
    )
    cls = pd.DataFrame(classification_report(best_pred["true_label"], best_pred["pred_label"], output_dict=True, zero_division=0)).T.reset_index(names="label")
    cm.to_csv(out_dir / "joined_small_liquid_best_external_confusion_matrix.csv")
    cls.to_csv(out_dir / "joined_small_liquid_best_external_classification_report.csv", index=False)
    params.to_csv(out_dir / "joined_small_liquid_parameter_counts.csv", index=False)
    write_report(Path(args.report), summary, train_valid, externals, params, best_model, best_split, cm.reset_index(names="true_label"), cls)
    print(f"Done. Report: {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
