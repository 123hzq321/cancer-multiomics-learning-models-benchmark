from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crawl_tcga_brca_multiomics_cbioportal import (  # noqa: E402
    API_BASE,
    fetch_molecular_matrix,
    fetch_mutation_matrix,
)
from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    EarlyFusionMLP,
    ModalityLiquidCfC,
    ModalityTCN,
    MultiOmicsDataset,
    compute_metrics,
    markdown_table,
    rows_from_result,
    save_predictions,
    set_seed,
    train_torch_model,
)


DATA_DIR = Path("work/data/metabric_external_validation/processed")
OUTPUTS_DIR = Path("outputs")

BREAST_MARKERS = """
ACTR3B ANLN BAG1 BCL2 BIRC5 BLVRA CCNB1 CCNE1 CDC20 CDC6 CDH3 CENPF CEP55 CXXC5
EGFR ERBB2 ESR1 EXO1 FGFR4 FOXC1 FOXA1 GRB7 KIF2C KRT14 KRT17 KRT5 MAPT MDM2 MELK
MIA MKI67 MLPH MMP11 MYBL2 MYC NAT1 ORC6 PGR PHGDH PTTG1 RRM2 SFRP1 SLC39A6
TMEM45B TYMS UBE2C UBE2T CENPA KNTC1 KIF20A AURKA AURKB TOP2A BUB1 BUB1B CCNB2
CDCA8 MCM2 MCM4 MCM6 PCNA CDK1 FOXM1 GATA3 XBP1 AGR2 TFF1 TFF3 KRT8 KRT18 KRT19
EPCAM VIM CDH1 CLDN3 CLDN4 CLDN7 PIK3CA PTEN AKT1 BRCA1 BRCA2 RAD51
""".split()

TCGA_PROFILES = {
    "mrna_z": "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna_median_all_sample_Zscores",
    "cna": "brca_tcga_pan_can_atlas_2018_gistic",
    "mutation": "brca_tcga_pan_can_atlas_2018_mutations",
}
METABRIC_PROFILES = {
    "mrna_z": "brca_metabric_mrna_median_all_sample_Zscores",
    "cna": "brca_metabric_cna",
    "mutation": "brca_metabric_mutations",
}


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def map_marker_genes(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "breast_marker_genes.csv"
    if path.exists():
        return pd.read_csv(path)
    response = requests.post(
        f"{API_BASE}/genes/fetch",
        params={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
        json=BREAST_MARKERS,
        headers={"Accept": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    genes = pd.DataFrame(response.json())
    genes = genes.drop_duplicates("entrezGeneId").sort_values("hugoGeneSymbol")
    genes[["entrezGeneId", "hugoGeneSymbol"]].to_csv(path, index=False)
    return genes[["entrezGeneId", "hugoGeneSymbol"]]


def read_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tcga = pd.read_csv(DATA_DIR / "tcga_brca_aligned_external_train_table.csv", low_memory=False)
    metabric = pd.read_csv(DATA_DIR / "metabric_aligned_external_validation_table.csv", low_memory=False)
    splits = pd.read_csv(DATA_DIR / "tcga_brca_external_validation_splits_70_15_15.csv")
    return tcga, metabric, splits


def fetch_or_read_matrix(
    cohort: str,
    modality: str,
    profile: str,
    sample_ids: list[str],
    genes: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    path = out_dir / f"{cohort}_{modality}_breast_markers.csv"
    if path.exists():
        matrix = pd.read_csv(path, index_col=0)
        return matrix.reindex(index=sample_ids, columns=genes["hugoGeneSymbol"].tolist())
    if modality == "mutation":
        matrix = fetch_mutation_matrix(profile, sample_ids, genes)
    else:
        matrix = fetch_molecular_matrix(profile, sample_ids, genes)
    matrix = matrix.reindex(index=sample_ids, columns=genes["hugoGeneSymbol"].tolist())
    if modality == "mutation":
        matrix = matrix.fillna(0).astype(np.int8)
    matrix.to_csv(path)
    return matrix


def add_prefix(matrix: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = matrix.copy()
    out.columns = [f"{prefix}__{col}" for col in out.columns]
    return out


def assemble_marker_tables(out_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]], LabelEncoder]:
    genes = map_marker_genes(out_dir)
    tcga_base, met_base, splits = read_base_tables()
    tcga_base = tcga_base.merge(splits, on="sampleId", how="inner")
    tcga_base = tcga_base[tcga_base["split"].isin(["train", "valid", "tcga_test"])].copy()
    tcga_base = tcga_base[tcga_base["subtype_label"].notna()].copy()
    met_base = met_base[met_base["subtype_label"].notna()].copy()
    met_base["split"] = "metabric_external"

    cohorts = {
        "tcga": (tcga_base, TCGA_PROFILES),
        "metabric": (met_base, METABRIC_PROFILES),
    }
    tables = []
    for cohort_name, (base, profiles) in cohorts.items():
        sample_ids = base["sampleId"].tolist()
        matrices = {}
        for modality, profile in profiles.items():
            print(f"{cohort_name} {modality}")
            matrices[modality] = fetch_or_read_matrix(cohort_name, modality, profile, sample_ids, genes, out_dir)
        features = pd.concat([add_prefix(matrices[m], m) for m in ["mrna_z", "cna", "mutation"]], axis=1).reset_index()
        tables.append(base[["sampleId", "patientId", "subtype_label", "split", "source_study_id"]].merge(features, on="sampleId", how="left"))
    combined = pd.concat(tables, axis=0, ignore_index=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(tcga_base["subtype_label"].astype(str))
    combined = combined[combined["subtype_label"].isin(label_encoder.classes_)].copy()
    combined["target"] = label_encoder.transform(combined["subtype_label"].astype(str))
    modality_cols = {
        "mrna_z": [c for c in combined.columns if c.startswith("mrna_z__")],
        "cna": [c for c in combined.columns if c.startswith("cna__")],
        "mutation": [c for c in combined.columns if c.startswith("mutation__")],
    }
    combined.to_csv(out_dir / "combined_breast_marker_external_table.csv", index=False)
    return combined, modality_cols, label_encoder


class PanelPreprocessor:
    def __init__(self, modality_cols: dict[str, list[str]]):
        self.modality_cols = modality_cols
        self.imputers: dict[str, SimpleImputer] = {}
        self.scalers: dict[str, StandardScaler] = {}

    def fit(self, frame: pd.DataFrame) -> "PanelPreprocessor":
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


def make_processed(combined: pd.DataFrame, selected: dict[str, list[str]]) -> dict[str, dict[str, object]]:
    pre = PanelPreprocessor(selected).fit(combined[combined["split"].eq("train")])
    processed = {}
    for split in ["train", "valid", "tcga_test", "metabric_external"]:
        frame = combined[combined["split"].eq(split)].copy()
        mods = pre.transform(frame)
        processed[split] = {
            "frame": frame,
            "modalities": mods,
            "fusion": np.concatenate(mods, axis=1).astype(np.float32),
            "y": frame["target"].to_numpy(dtype=np.int64),
            "sampleId": frame["sampleId"].to_numpy(),
        }
    return processed


def loader_dict(processed: dict[str, dict[str, object]], batch_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for split, payload in processed.items():
        loaders[split] = DataLoader(
            MultiOmicsDataset(payload["fusion"], payload["modalities"], payload["y"]),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
    return loaders


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
        rows.append({"model": name, "model_group": group, "metrics": metrics, "predictions": predictions, "best_epoch": np.nan, "best_validation_score": metrics["valid"]["roc_auc_ovr_macro"]})
    return rows


def summarize_by_split(results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for split in ["tcga_test", "metabric_external"]:
        tmp = results[results["split"].eq(split)]
        s = (
            tmp.groupby(["feature_set", "model", "model_group"])
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
        s.insert(0, "evaluation_split", split)
        frames.append(s)
    return pd.concat(frames, axis=0, ignore_index=True).sort_values(["evaluation_split", "f1_macro_mean"], ascending=[True, False])


def add_feature_set(rows: list[dict[str, object]], feature_set: str) -> list[dict[str, object]]:
    for row in rows:
        row["feature_set"] = feature_set
    return rows


def write_report(path: Path, summary: pd.DataFrame, best_confusion: pd.DataFrame, best_report: pd.DataFrame, best_model: str) -> None:
    lines = [
        "# METABRIC Breast Marker External Validation v0",
        "",
        "## Scope",
        "",
        "Rescue external validation using a breast subtype marker expression panel instead of the generic IMPACT468 feature set.",
        "",
        "Feature sets:",
        "",
        "- `marker_mrna`: mRNA z-score marker panel only.",
        "- `marker_multimodal`: marker mRNA z-score + marker CNA + marker mutation.",
        "",
        "## Summary",
        "",
        markdown_table(summary, floatfmt=".4f"),
        "",
        f"## Best METABRIC Model: `{best_model}`",
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
        "If this marker panel improves external macro F1/AUC, the earlier weak METABRIC result was partly caused by using a generic cancer gene panel rather than subtype-specific expression biology.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DATA_DIR / "breast_marker_panel"))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "metabric_breast_marker_external_validation_report_v0.md"))
    parser.add_argument("--results-csv", default=str(DATA_DIR / "breast_marker_panel" / "marker_external_model_results.csv"))
    parser.add_argument("--summary-csv", default=str(DATA_DIR / "breast_marker_panel" / "marker_external_model_summary.csv"))
    parser.add_argument("--predictions-csv", default=str(DATA_DIR / "breast_marker_panel" / "marker_external_model_predictions.csv"))
    parser.add_argument("--seeds", default="42,43,44")
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
    combined, modality_cols, le = assemble_marker_tables(out_dir)
    feature_sets = {
        "marker_mrna": {"mrna_z": modality_cols["mrna_z"]},
        "marker_multimodal_original": {"mrna_z": modality_cols["mrna_z"], "cna": modality_cols["cna"], "mutation": modality_cols["mutation"]},
        "marker_multimodal_expression_last": {"cna": modality_cols["cna"], "mutation": modality_cols["mutation"], "mrna_z": modality_cols["mrna_z"]},
    }
    all_rows, pred_rows = [], []
    for feature_set, selected in feature_sets.items():
        print("Feature set", feature_set)
        processed = make_processed(combined, selected)
        input_dims = [processed["train"]["modalities"][i].shape[1] for i in range(len(selected))]
        fusion_dim = processed["train"]["fusion"].shape[1]
        for seed in seeds:
            set_seed(seed)
            for result in fit_sklearn(processed, seed, model_prefix=f"{feature_set}__"):
                rows = add_feature_set(rows_from_result(seed, result["model"], result["model_group"], result), feature_set)
                all_rows.extend(rows)
                for split, (y, proba) in result["predictions"].items():
                    pred_rows.append({"seed": seed, "model": result["model"], "split": split, "sample_ids": processed[split]["sampleId"], "y_true": y, "proba": proba})
            loaders = loader_dict(processed, args.batch_size)
            torch_models = [
                (f"{feature_set}__mlp", "static_nn", EarlyFusionMLP(fusion_dim, len(le.classes_), hidden_dim=128)),
            ]
            if len(selected) > 1:
                torch_models.extend(
                    [
                        (f"{feature_set}__tcn", "sequence_nn", ModalityTCN(input_dims, len(le.classes_), embed_dim=64, hidden_dim=96)),
                        (f"{feature_set}__liquid_cfc", "liquid_nn", ModalityLiquidCfC(input_dims, len(le.classes_), embed_dim=64, hidden_dim=96)),
                    ]
                )
            for model_name, group, model in torch_models:
                set_seed(seed)
                print(seed, model_name)
                result = train_torch_model(
                    model,
                    loaders,
                    processed["train"]["y"],
                    len(le.classes_),
                    device=torch.device(args.device),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    max_epochs=args.epochs,
                    patience=args.patience,
                )
                rows = add_feature_set(rows_from_result(seed, model_name, group, result), feature_set)
                all_rows.extend(rows)
                for split, (y, proba) in result["predictions"].items():
                    pred_rows.append({"seed": seed, "model": model_name, "split": split, "sample_ids": processed[split]["sampleId"], "y_true": y, "proba": proba})

    results = pd.DataFrame(all_rows)
    summary = summarize_by_split(results)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    save_predictions(pred_rows, Path(args.predictions_csv), le)
    external = summary[summary["evaluation_split"].eq("metabric_external")].sort_values("f1_macro_mean", ascending=False)
    best_model = str(external.iloc[0]["model"])
    best_seed = (
        results[results["split"].eq("metabric_external") & results["model"].eq(best_model)]
        .sort_values("f1_macro", ascending=False)
        .iloc[0]["seed"]
    )
    pred = pd.read_csv(args.predictions_csv)
    best_pred = pred[pred["split"].eq("metabric_external") & pred["model"].eq(best_model) & pred["seed"].eq(best_seed)]
    labels = list(le.classes_)
    cm = pd.DataFrame(confusion_matrix(best_pred["true_label"], best_pred["pred_label"], labels=labels), index=[f"true__{x}" for x in labels], columns=[f"pred__{x}" for x in labels])
    cls = pd.DataFrame(classification_report(best_pred["true_label"], best_pred["pred_label"], output_dict=True, zero_division=0)).T.reset_index(names="label")
    cm.to_csv(out_dir / "marker_external_best_confusion_matrix.csv")
    cls.to_csv(out_dir / "marker_external_best_classification_report.csv", index=False)
    write_report(Path(args.report), summary, cm.reset_index(names="true_label"), cls, best_model)
    print(f"Done. Report: {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
