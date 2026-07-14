from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crawl_tcga_brca_multiomics_cbioportal import (  # noqa: E402
    API_BASE,
    GENE_PANEL_ID,
    fetch_clinical,
    fetch_molecular_matrix,
    fetch_mutation_matrix,
    get_gene_panel,
    page_get,
    request_json,
)
from train_tcga_brca_multiomics_baselines_vs_liquid import markdown_table  # noqa: E402


TCGA_STUDY = "brca_tcga_pan_can_atlas_2018"
METABRIC_STUDY = "brca_metabric"

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
MODALITIES = ["mrna_z", "cna", "mutation"]

TCGA_LABEL_MAP = {
    "BRCA_LumA": "LumA",
    "BRCA_LumB": "LumB",
    "BRCA_Basal": "Basal",
    "BRCA_Her2": "Her2",
    "BRCA_Normal": "Normal",
}
METABRIC_LABEL_MAP = {
    "LumA": "LumA",
    "LumB": "LumB",
    "Basal": "Basal",
    "Her2": "Her2",
    "Normal": "Normal",
}


def add_prefix(matrix: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = matrix.copy()
    out.columns = [f"{prefix}__{col}" for col in out.columns]
    return out


def read_matrix(path: Path, samples: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.reindex(index=samples)
    matrix.index.name = "sampleId"
    return matrix


def reindex_genes(matrix: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    return matrix.reindex(columns=genes["hugoGeneSymbol"].tolist())


def get_samples(study_id: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{study_id}_samples.csv"
    if path.exists():
        return pd.read_csv(path)
    samples = pd.DataFrame(page_get(f"/studies/{study_id}/samples", params={"projection": "SUMMARY"}, page_size=3000))
    samples.to_csv(path, index=False)
    return samples


def crawl_cohort(
    *,
    study_id: str,
    profiles: dict[str, str],
    label_attr: str,
    label_map: dict[str, str],
    cohort_dir: Path,
    raw_dir: Path,
    genes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cohort_dir.mkdir(parents=True, exist_ok=True)
    samples = get_samples(study_id, raw_dir)
    primary = samples[samples["sampleType"].eq("Primary Solid Tumor")].copy()
    if primary.empty:
        primary = samples.copy()
    sample_ids = primary["sampleId"].tolist()
    patient_ids = sorted(primary["patientId"].dropna().unique())
    clinical = fetch_clinical(study_id, patient_ids, [label_attr, "OS_STATUS", "OS_MONTHS"], "PATIENT")
    clinical = clinical.rename(columns={label_attr: "raw_subtype"})
    clinical["subtype_label"] = clinical["raw_subtype"].map(label_map)
    sample_table = primary[["sampleId", "patientId", "sampleType"]].merge(clinical, on="patientId", how="left")
    sample_table["source_study_id"] = study_id

    matrices: dict[str, pd.DataFrame] = {}
    for modality in ["mrna_z", "cna"]:
        path = cohort_dir / f"{study_id}_{modality}_impact468_matrix.csv"
        if path.exists():
            print(f"  loading {study_id} {modality}")
            matrix = read_matrix(path, sample_ids)
        else:
            print(f"  fetching {study_id} {modality}")
            matrix = fetch_molecular_matrix(profiles[modality], sample_ids, genes)
            matrix.to_csv(path)
        matrices[modality] = reindex_genes(matrix, genes)

    mut_path = cohort_dir / f"{study_id}_mutation_impact468_matrix.csv"
    if mut_path.exists():
        print(f"  loading {study_id} mutation")
        mutation = read_matrix(mut_path, sample_ids)
    else:
        print(f"  fetching {study_id} mutation")
        mutation = fetch_mutation_matrix(profiles["mutation"], sample_ids, genes)
        mutation.to_csv(mut_path)
    matrices["mutation"] = reindex_genes(mutation, genes).fillna(0).astype(np.int8)
    return sample_table, matrices


def assemble(sample_table: pd.DataFrame, matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    features = pd.concat([add_prefix(matrices[m], m) for m in MODALITIES], axis=1).reset_index()
    table = sample_table.merge(features, on="sampleId", how="left")
    return table


def make_tcga_splits(table: pd.DataFrame, processed_dir: Path, seed: int) -> pd.DataFrame:
    labeled = table[table["subtype_label"].notna()].copy()
    train_ids, temp_ids = train_test_split(
        labeled["sampleId"],
        train_size=0.70,
        random_state=seed,
        stratify=labeled["subtype_label"],
    )
    temp = labeled.set_index("sampleId").loc[temp_ids].reset_index()
    valid_ids, test_ids = train_test_split(
        temp["sampleId"],
        train_size=0.50,
        random_state=seed,
        stratify=temp["subtype_label"],
    )
    splits = pd.DataFrame(
        {
            "sampleId": list(train_ids) + list(valid_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["valid"] * len(valid_ids) + ["tcga_test"] * len(test_ids),
        }
    )
    splits = table[["sampleId"]].merge(splits, on="sampleId", how="left")
    splits["split"] = splits["split"].fillna("unused")
    splits.to_csv(processed_dir / "tcga_brca_external_validation_splits_70_15_15.csv", index=False)
    return splits


def write_report(
    path: Path,
    tcga: pd.DataFrame,
    metabric: pd.DataFrame,
    splits: pd.DataFrame,
    genes: pd.DataFrame,
) -> None:
    tcga_counts = tcga["subtype_label"].fillna("NA").value_counts().rename_axis("subtype").reset_index(name="n")
    met_counts = metabric["subtype_label"].fillna("NA").value_counts().rename_axis("subtype").reset_index(name="n")
    split_counts = splits["split"].value_counts().rename_axis("split").reset_index(name="n")
    lines = [
        "# METABRIC External Validation Crawl v0",
        "",
        "## Scope",
        "",
        "Build aligned TCGA-BRCA training and METABRIC external validation tables.",
        "",
        "- Training cohort: TCGA BRCA PanCancer Atlas 2018.",
        "- External validation cohort: METABRIC.",
        "- Aligned modalities: mRNA z-score, discrete CNA, mutation binary.",
        f"- Gene panel: `{GENE_PANEL_ID}` with {genes.shape[0]} genes.",
        "- Target: LumA, LumB, Basal, Her2, Normal subtype labels.",
        "",
        "## TCGA Label Counts",
        "",
        markdown_table(tcga_counts),
        "",
        "## METABRIC Label Counts",
        "",
        markdown_table(met_counts),
        "",
        "## TCGA Split Counts",
        "",
        markdown_table(split_counts),
        "",
        "## Output Files",
        "",
        "- `work/data/metabric_external_validation/processed/tcga_brca_aligned_external_train_table.csv`",
        "- `work/data/metabric_external_validation/processed/metabric_aligned_external_validation_table.csv`",
        "- `work/data/metabric_external_validation/processed/tcga_brca_external_validation_splits_70_15_15.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="work/data/metabric_external_validation")
    parser.add_argument("--report", default="outputs/metabric_external_validation_crawl_report_v0.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    cohort_dir = processed_dir / "cohorts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    genes = get_gene_panel(GENE_PANEL_ID)
    genes.to_csv(raw_dir / "impact468_genes.csv", index=False)

    print("Crawling TCGA aligned cohort")
    tcga_samples, tcga_matrices = crawl_cohort(
        study_id=TCGA_STUDY,
        profiles=TCGA_PROFILES,
        label_attr="SUBTYPE",
        label_map=TCGA_LABEL_MAP,
        cohort_dir=cohort_dir / TCGA_STUDY,
        raw_dir=raw_dir,
        genes=genes,
    )
    tcga_table = assemble(tcga_samples, tcga_matrices)
    tcga_table.to_csv(processed_dir / "tcga_brca_aligned_external_train_table.csv", index=False)
    splits = make_tcga_splits(tcga_table, processed_dir, args.seed)

    print("Crawling METABRIC external cohort")
    met_samples, met_matrices = crawl_cohort(
        study_id=METABRIC_STUDY,
        profiles=METABRIC_PROFILES,
        label_attr="CLAUDIN_SUBTYPE",
        label_map=METABRIC_LABEL_MAP,
        cohort_dir=cohort_dir / METABRIC_STUDY,
        raw_dir=raw_dir,
        genes=genes,
    )
    met_table = assemble(met_samples, met_matrices)
    met_table.to_csv(processed_dir / "metabric_aligned_external_validation_table.csv", index=False)

    write_report(Path(args.report), tcga_table, met_table, splits, genes)
    print(f"Done. Report: {args.report}")


if __name__ == "__main__":
    main()
