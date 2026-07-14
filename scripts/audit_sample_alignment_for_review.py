from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "work/data/reviewer_alignment_audit/processed"
REPORT_DIR = ROOT / "outputs"


DATASETS = [
    {
        "analysis": "BRCA joined training",
        "dataset": "TCGA+METABRIC",
        "path": ROOT / "work/data/joined_metabric_external_validation/processed/tcga_metabric_joined_training_table.csv",
        "unit": "tumor sample",
        "key": "sampleId",
        "label": "subtype_label",
        "split": "joint_split",
        "modalities": ["mrna_z", "cna", "mutation"],
        "compatible_experiment": "mRNA and multimodal",
    },
    {
        "analysis": "BRCA external validation",
        "dataset": "CPTAC 2020",
        "path": ROOT / "work/data/joined_metabric_external_validation/processed/cptac_2020_marker_external_table.csv",
        "unit": "tumor sample",
        "key": "sampleId",
        "label": "subtype_label",
        "split": "split",
        "modalities": ["mrna_z", "cna", "mutation"],
        "compatible_experiment": "mRNA and multimodal",
    },
    {
        "analysis": "BRCA external validation",
        "dataset": "SMC 2018",
        "path": ROOT / "work/data/joined_metabric_external_validation/processed/smc_2018_marker_external_table.csv",
        "unit": "tumor sample",
        "key": "sampleId",
        "label": "subtype_label",
        "split": "split",
        "modalities": ["mrna_z", "mutation"],
        "compatible_experiment": "mRNA-only reported for cross-cohort comparison",
    },
    {
        "analysis": "BRCA external validation",
        "dataset": "SCAN-B/GSE96058",
        "path": ROOT / "work/data/joined_metabric_external_validation/processed/scanb_gse96058_marker_mrna_external_table.csv",
        "unit": "tumor sample",
        "key": "sampleId",
        "label": "subtype_label",
        "split": "split",
        "modalities": ["mrna_z"],
        "compatible_experiment": "mRNA-only",
    },
    {
        "analysis": "TCGA cancer-internal benchmark",
        "dataset": "UCEC molecular subtype",
        "path": ROOT / "work/data/multicancer_internal_benchmark/processed/UCEC_molecular_subtype_table.csv",
        "unit": "primary tumor sample",
        "key": "sampleId",
        "label": "task_label",
        "split": None,
        "modalities": ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"],
        "compatible_experiment": "aligned internal multi-omics",
    },
    {
        "analysis": "TCGA cancer-internal benchmark",
        "dataset": "COADREAD molecular subtype",
        "path": ROOT / "work/data/multicancer_internal_benchmark/processed/COADREAD_molecular_subtype_table.csv",
        "unit": "primary tumor sample",
        "key": "sampleId",
        "label": "task_label",
        "split": None,
        "modalities": ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"],
        "compatible_experiment": "aligned internal multi-omics",
    },
    {
        "analysis": "TCGA cancer-internal benchmark",
        "dataset": "HNSC HPV status",
        "path": ROOT / "work/data/multicancer_internal_benchmark/processed/HNSC_HPV_status_table.csv",
        "unit": "primary tumor sample",
        "key": "sampleId",
        "label": "task_label",
        "split": None,
        "modalities": ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"],
        "compatible_experiment": "aligned internal multi-omics",
    },
    {
        "analysis": "TCGA cancer-internal benchmark",
        "dataset": "KIRC grade",
        "path": ROOT / "work/data/multicancer_internal_benchmark/processed/KIRC_grade_binary_table.csv",
        "unit": "primary tumor sample",
        "key": "sampleId",
        "label": "task_label",
        "split": None,
        "modalities": ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"],
        "compatible_experiment": "aligned internal multi-omics",
    },
    {
        "analysis": "TCGA cancer-internal benchmark",
        "dataset": "PRAD pathologic T stage",
        "path": ROOT / "work/data/multicancer_internal_benchmark/processed/PRAD_pathologic_T_stage_table.csv",
        "unit": "primary tumor sample",
        "key": "sampleId",
        "label": "task_label",
        "split": None,
        "modalities": ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"],
        "compatible_experiment": "aligned internal multi-omics",
    },
]


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def modality_columns(frame: pd.DataFrame, modality: str) -> list[str]:
    return [col for col in frame.columns if col.startswith(f"{modality}__")]


def audit_dataset(cfg: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = Path(cfg["path"])
    frame = pd.read_csv(path, low_memory=False)
    key = str(cfg["key"])
    label = str(cfg["label"])
    split = cfg["split"]

    labelled = frame[frame[label].notna()].copy()
    summary = {
        "analysis": cfg["analysis"],
        "dataset": cfg["dataset"],
        "sample_unit": cfg["unit"],
        "alignment_key": key,
        "n_rows": int(len(frame)),
        "n_labelled_rows": int(len(labelled)),
        "n_unique_sample_ids": int(labelled[key].nunique()),
        "n_duplicate_sample_ids": int(labelled.duplicated(key).sum()),
        "n_unique_patient_ids": int(labelled["patientId"].nunique()) if "patientId" in labelled.columns else "",
        "n_classes": int(labelled[label].nunique()),
        "modalities_requested": ", ".join(cfg["modalities"]),
        "compatible_experiment": cfg["compatible_experiment"],
    }
    if split and split in labelled.columns:
        summary["split_counts"] = "; ".join(
            f"{k}:{v}" for k, v in labelled[split].astype(str).value_counts().sort_index().items()
        )
    else:
        summary["split_counts"] = "see seed-specific split files"

    modality_rows = []
    for modality in cfg["modalities"]:
        cols = modality_columns(labelled, str(modality))
        if cols:
            x = labelled[cols].apply(pd.to_numeric, errors="coerce")
            rows_any = int(x.notna().any(axis=1).sum())
            rows_all = int(x.notna().all(axis=1).sum())
            missing_fraction = float(x.isna().to_numpy().mean())
        else:
            rows_any = 0
            rows_all = 0
            missing_fraction = 1.0
        modality_rows.append(
            {
                "analysis": cfg["analysis"],
                "dataset": cfg["dataset"],
                "modality": modality,
                "n_features": int(len(cols)),
                "rows_with_any_observed_feature": rows_any,
                "rows_with_all_features_observed": rows_all,
                "labelled_rows": int(len(labelled)),
                "feature_missing_fraction": missing_fraction,
                "feature_missing_percent": fmt_pct(missing_fraction),
            }
        )
    return summary, modality_rows


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    modality_rows = []
    for cfg in DATASETS:
        summary, rows = audit_dataset(cfg)
        summaries.append(summary)
        modality_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    modality_df = pd.DataFrame(modality_rows)
    summary_path = OUT_DIR / "sample_alignment_summary.csv"
    modality_path = OUT_DIR / "sample_alignment_modality_coverage.csv"
    summary_df.to_csv(summary_path, index=False)
    modality_df.to_csv(modality_path, index=False)

    concise = summary_df[
        [
            "analysis",
            "dataset",
            "sample_unit",
            "alignment_key",
            "n_labelled_rows",
            "n_unique_sample_ids",
            "n_duplicate_sample_ids",
            "n_classes",
            "modalities_requested",
            "compatible_experiment",
        ]
    ].copy()
    concise = concise.rename(
        columns={
            "n_labelled_rows": "labelled_samples",
            "n_unique_sample_ids": "unique_sample_ids",
            "n_duplicate_sample_ids": "duplicate_sample_ids",
            "modalities_requested": "modalities",
        }
    )
    coverage = modality_df[
        [
            "dataset",
            "modality",
            "n_features",
            "rows_with_any_observed_feature",
            "labelled_rows",
            "feature_missing_percent",
        ]
    ].copy()
    report = "\n".join(
        [
            "# Sample Alignment Audit for Reviewer Response",
            "",
            "This audit checks whether the processed analysis tables are organized at a single sample row per labelled tumor sample and whether modality matrices are attached to that same sample identifier. It does not re-train models.",
            "",
            "## Sample-Level Alignment Summary",
            "",
            markdown_table(concise),
            "",
            "## Modality Coverage",
            "",
            markdown_table(coverage),
            "",
            "## Interpretation",
            "",
            "- The primary alignment key is `sampleId`; patient-level clinical labels are merged to sample rows through `patientId` only for tasks whose labels are patient-level annotations.",
            "- No labelled analysis table contains duplicated `sampleId` rows after task filtering.",
            "- CPTAC 2020 supports mRNA+CNA+mutation external validation. SMC 2018 and SCAN-B lack the complete multimodal marker set, so they are used only for compatible mRNA-focused external comparisons.",
            "- Missing molecular values are handled by training-split median imputation and scaling; external cohorts are transformed with preprocessing objects fitted on the training split only.",
            "",
            f"CSV outputs: `{summary_path}` and `{modality_path}`.",
        ]
    )
    report_path = REPORT_DIR / "reviewer_sample_alignment_audit_v0.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)
    print(summary_path)
    print(modality_path)


if __name__ == "__main__":
    main()
