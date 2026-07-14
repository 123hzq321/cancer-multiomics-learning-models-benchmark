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
    fetch_methylation_matrix,
    fetch_molecular_matrix,
    fetch_mutation_matrix,
    get_gene_panel,
    load_methylation_probe_map,
    markdown_table,
    page_get,
    request_json,
)


DEFAULT_STUDIES = [
    "brca_tcga_pan_can_atlas_2018",
    "luad_tcga_pan_can_atlas_2018",
    "lusc_tcga_pan_can_atlas_2018",
    "coadread_tcga_pan_can_atlas_2018",
    "gbm_tcga_pan_can_atlas_2018",
    "hnsc_tcga_pan_can_atlas_2018",
    "kirc_tcga_pan_can_atlas_2018",
    "prad_tcga_pan_can_atlas_2018",
    "ucec_tcga_pan_can_atlas_2018",
    "ov_tcga_pan_can_atlas_2018",
]

MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"]


def cancer_code(study_id: str) -> str:
    return study_id.replace("_tcga_pan_can_atlas_2018", "").upper()


def list_profiles(study_id: str) -> dict[str, str]:
    profiles = request_json("GET", f"{API_BASE}/studies/{study_id}/molecular-profiles")
    ids = {item["molecularProfileId"] for item in profiles}
    expected = {
        "mrna": f"{study_id}_rna_seq_v2_mrna",
        "gistic": f"{study_id}_gistic",
        "log2cna": f"{study_id}_log2CNA",
        "methylation": f"{study_id}_methylation_hm27_hm450_merge",
        "rppa": f"{study_id}_rppa",
        "mutation": f"{study_id}_mutations",
    }
    if expected["methylation"] not in ids:
        expected["methylation"] = f"{study_id}_methylation_hm450"
    missing = [name for name, profile_id in expected.items() if profile_id not in ids]
    if missing:
        raise ValueError(f"{study_id} missing profiles: {missing}")
    return expected


def read_matrix(path: Path, sample_ids: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.reindex(index=sample_ids)
    matrix.index.name = "sampleId"
    return matrix


def reindex_gene_matrix(matrix: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    return matrix.reindex(columns=genes["hugoGeneSymbol"].tolist())


def reindex_methylation_matrix(matrix: pd.DataFrame, probe_map: pd.DataFrame) -> pd.DataFrame:
    return matrix.reindex(columns=probe_map["hugoGeneSymbol"].tolist())


def add_prefix(matrix: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = matrix.copy()
    out.columns = [f"{prefix}__{col}" for col in out.columns]
    return out


def crawl_study(
    study_id: str,
    study_dir: Path,
    genes: pd.DataFrame,
    probe_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    study_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nStudy {study_id}")
    profiles = list_profiles(study_id)
    samples = pd.DataFrame(
        page_get(f"/studies/{study_id}/samples", params={"projection": "SUMMARY"}, page_size=2000)
    )
    samples.to_csv(study_dir / "samples.csv", index=False)
    if samples.empty:
        raise ValueError(f"No samples for {study_id}")
    primary = samples[samples["sampleType"].eq("Primary Solid Tumor")].copy()
    if primary.empty:
        primary = samples.copy()
    primary["cancer_type"] = cancer_code(study_id)
    primary["source_study_id"] = study_id
    sample_ids = primary["sampleId"].tolist()
    print(f"  samples={len(sample_ids)}")

    matrices: dict[str, pd.DataFrame] = {}
    for modality in ["mrna", "gistic", "log2cna", "rppa"]:
        path = study_dir / f"{modality}_impact468_matrix.csv"
        if path.exists():
            print(f"  loading {modality}")
            matrix = read_matrix(path, sample_ids)
        else:
            print(f"  fetching {modality}")
            matrix = fetch_molecular_matrix(profiles[modality], sample_ids, genes)
            matrix.to_csv(path)
        matrices[modality] = reindex_gene_matrix(matrix, genes)

    methylation_path = study_dir / "methylation_impact468_matrix.csv"
    if methylation_path.exists():
        print("  loading methylation")
        methylation = read_matrix(methylation_path, sample_ids)
    else:
        print("  fetching methylation")
        methylation = fetch_methylation_matrix(profiles["methylation"], sample_ids, probe_map)
        methylation.to_csv(methylation_path)
    matrices["methylation"] = reindex_methylation_matrix(methylation, probe_map)

    mutation_path = study_dir / "mutation_impact468_matrix.csv"
    if mutation_path.exists():
        print("  loading mutation")
        mutation = read_matrix(mutation_path, sample_ids)
    else:
        print("  fetching mutation")
        mutation = fetch_mutation_matrix(profiles["mutation"], sample_ids, genes)
        mutation.to_csv(mutation_path)
    matrices["mutation"] = reindex_gene_matrix(mutation, genes).fillna(0).astype(np.int8)

    return primary[["sampleId", "patientId", "sampleType", "source_study_id", "cancer_type"]], matrices


def make_splits(table: pd.DataFrame, processed_dir: Path, seed: int) -> pd.DataFrame:
    labeled = table.copy()
    counts = labeled["cancer_type"].value_counts()
    labeled = labeled[labeled["cancer_type"].isin(counts[counts >= 20].index)].copy()
    train_ids, temp_ids = train_test_split(
        labeled["sampleId"],
        train_size=0.70,
        random_state=seed,
        stratify=labeled["cancer_type"],
    )
    temp = labeled.set_index("sampleId").loc[temp_ids].reset_index()
    valid_ids, test_ids = train_test_split(
        temp["sampleId"],
        train_size=0.50,
        random_state=seed,
        stratify=temp["cancer_type"],
    )
    splits = pd.DataFrame(
        {
            "sampleId": list(train_ids) + list(valid_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["valid"] * len(valid_ids) + ["test"] * len(test_ids),
        }
    )
    splits = table[["sampleId"]].merge(splits, on="sampleId", how="left")
    splits["split"] = splits["split"].fillna("unused")
    splits.to_csv(processed_dir / "pancancer_cancer_type_splits_70_15_15.csv", index=False)
    return splits


def write_report(
    report_path: Path,
    combined: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    splits: pd.DataFrame,
    genes: pd.DataFrame,
    probe_map: pd.DataFrame,
    studies: list[str],
) -> None:
    sample_counts = combined["cancer_type"].value_counts().rename_axis("cancer_type").reset_index(name="n")
    split_counts = splits["split"].value_counts().rename_axis("split").reset_index(name="n")
    modality_rows = []
    for modality, matrix in matrices.items():
        modality_rows.append(
            {
                "modality": modality,
                "samples": matrix.shape[0],
                "features": matrix.shape[1],
                "observed_fraction": float(matrix.notna().to_numpy().mean()) if matrix.size else 0.0,
            }
        )
    modality_summary = pd.DataFrame(modality_rows)
    lines = [
        "# TCGA PanCancer Multi-omics Crawl Report v0",
        "",
        "## Source",
        "",
        "- cBioPortal public API: https://www.cbioportal.org/api",
        f"- Studies: {', '.join(studies)}",
        f"- Gene panel: `{GENE_PANEL_ID}`",
        "- Target for first multi-cancer task: `cancer_type`.",
        "",
        "## Dataset Summary",
        "",
        f"- Samples retained: {combined.shape[0]}",
        f"- Cancer types: {combined['cancer_type'].nunique()}",
        f"- Genes requested: {genes.shape[0]}",
        f"- Methylation probe-mapped genes: {probe_map.shape[0]}",
        f"- Final modeling table columns: {combined.shape[1]}",
        "",
        "## Samples by Cancer Type",
        "",
        markdown_table(sample_counts),
        "",
        "## Modality Matrices",
        "",
        markdown_table(modality_summary, floatfmt=".4f"),
        "",
        "## Split Counts",
        "",
        markdown_table(split_counts),
        "",
        "## Output Files",
        "",
        "- `work/data/tcga_pancancer_cbioportal/processed/pancancer_multimodal_impact468_table.csv`",
        "- `work/data/tcga_pancancer_cbioportal/processed/pancancer_cancer_type_splits_70_15_15.csv`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="work/data/tcga_pancancer_cbioportal")
    parser.add_argument("--report", default="outputs/tcga_pancancer_multiomics_crawl_report_v0.md")
    parser.add_argument("--studies", default=",".join(DEFAULT_STUDIES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    per_study_dir = processed_dir / "per_study"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    studies = [item.strip() for item in args.studies.split(",") if item.strip()]
    genes = get_gene_panel(GENE_PANEL_ID)
    genes.to_csv(raw_dir / "impact468_genes.csv", index=False)
    # Use one PanCancer methylation metadata profile to select a consistent probe per gene.
    probe_map_path = processed_dir / "pancancer_methylation_probe_map_impact468.csv"
    if probe_map_path.exists():
        probe_map = pd.read_csv(probe_map_path)
    else:
        probe_map = load_methylation_probe_map(
            "brca_tcga_pan_can_atlas_2018_methylation_hm27_hm450_merge",
            genes,
        )
        probe_map.to_csv(probe_map_path, index=False)
    print(f"Genes={genes.shape[0]} methylation_probes={probe_map.shape[0]}")

    sample_frames: list[pd.DataFrame] = []
    modality_frames: dict[str, list[pd.DataFrame]] = {modality: [] for modality in MODALITIES}
    for study_id in studies:
        samples, matrices = crawl_study(study_id, per_study_dir / study_id, genes, probe_map)
        sample_frames.append(samples)
        for modality in MODALITIES:
            modality_frames[modality].append(matrices[modality])

    sample_table = pd.concat(sample_frames, axis=0, ignore_index=True)
    sample_table.to_csv(processed_dir / "pancancer_sample_table.csv", index=False)
    combined_matrices: dict[str, pd.DataFrame] = {}
    for modality, frames in modality_frames.items():
        matrix = pd.concat(frames, axis=0)
        matrix = matrix[~matrix.index.duplicated(keep="first")]
        matrix.to_csv(processed_dir / f"pancancer_{modality}_impact468_matrix.csv")
        combined_matrices[modality] = matrix

    features = pd.concat([add_prefix(combined_matrices[m], m) for m in MODALITIES], axis=1).reset_index()
    combined = sample_table.merge(features, on="sampleId", how="left")
    combined.to_csv(processed_dir / "pancancer_multimodal_impact468_table.csv", index=False)
    splits = make_splits(combined, processed_dir, args.seed)
    write_report(Path(args.report), combined, combined_matrices, splits, genes, probe_map, studies)
    print(f"Done. Report: {args.report}")


if __name__ == "__main__":
    main()
