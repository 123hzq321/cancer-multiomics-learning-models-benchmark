from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split


API_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga_pan_can_atlas_2018"
GENE_PANEL_ID = "IMPACT468"

PROFILES = {
    "mrna": "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
    "gistic": "brca_tcga_pan_can_atlas_2018_gistic",
    "log2cna": "brca_tcga_pan_can_atlas_2018_log2CNA",
    "rppa": "brca_tcga_pan_can_atlas_2018_rppa",
    "methylation": "brca_tcga_pan_can_atlas_2018_methylation_hm450",
    "mutation": "brca_tcga_pan_can_atlas_2018_mutations",
}

PATIENT_CLINICAL_ATTRS = [
    "SUBTYPE",
    "AGE",
    "SEX",
    "OS_STATUS",
    "OS_MONTHS",
    "DFS_STATUS",
    "DFS_MONTHS",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "PATH_T_STAGE",
    "PATH_N_STAGE",
    "PATH_M_STAGE",
]

SAMPLE_CLINICAL_ATTRS = [
    "CANCER_TYPE",
    "CANCER_TYPE_DETAILED",
    "ANEUPLOIDY_SCORE",
    "FRACTION_GENOME_ALTERED",
    "MUTATION_COUNT",
    "TMB_NONSYNONYMOUS",
]


def request_json(
    method: str,
    url: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    retries: int = 5,
    sleep: float = 1.0,
) -> object:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                json=json_body,
                params=params,
                headers={"Accept": "application/json", "User-Agent": "codex-tcga-crawl/1.0"},
                timeout=120,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            if not response.text:
                return []
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(sleep * (2**attempt))
    raise RuntimeError(f"Request failed after {retries} retries: {url}") from last_error


def chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def page_get(path: str, *, page_size: int = 2000, params: dict | None = None) -> list[dict]:
    all_rows: list[dict] = []
    page_number = 0
    while True:
        page_params = dict(params or {})
        page_params.update({"pageSize": page_size, "pageNumber": page_number})
        rows = request_json("GET", f"{API_BASE}{path}", params=page_params)
        if not rows:
            break
        if not isinstance(rows, list):
            raise TypeError(f"Expected list for {path}, got {type(rows)}")
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page_number += 1
    return all_rows


def pivot_clinical(rows: list[dict], id_col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    wide = (
        frame.pivot_table(index=id_col, columns="clinicalAttributeId", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return wide


def fetch_clinical(study_id: str, ids: list[str], attrs: list[str], clinical_type: str) -> pd.DataFrame:
    id_col = "patientId" if clinical_type == "PATIENT" else "sampleId"
    rows: list[dict] = []
    for id_chunk in chunks(ids, 300):
        body = {"attributeIds": attrs, "ids": id_chunk}
        data = request_json(
            "POST",
            f"{API_BASE}/studies/{study_id}/clinical-data/fetch",
            params={"clinicalDataType": clinical_type, "projection": "SUMMARY"},
            json_body=body,
        )
        rows.extend(data)
    return pivot_clinical(rows, id_col)


def get_gene_panel(panel_id: str) -> pd.DataFrame:
    data = request_json(
        "GET",
        f"{API_BASE}/gene-panels/{panel_id}",
        params={"projection": "DETAILED"},
    )
    genes = pd.DataFrame(data["genes"])
    genes = genes.sort_values(["hugoGeneSymbol", "entrezGeneId"]).drop_duplicates("entrezGeneId")
    return genes[["entrezGeneId", "hugoGeneSymbol"]].reset_index(drop=True)


def fetch_molecular_matrix(
    profile_id: str,
    samples: list[str],
    genes: pd.DataFrame,
    *,
    gene_chunk_size: int = 80,
) -> pd.DataFrame:
    rows: list[dict] = []
    entrez_ids = genes["entrezGeneId"].astype(int).tolist()
    for i, gene_chunk in enumerate(chunks(entrez_ids, gene_chunk_size), start=1):
        body = {"sampleIds": samples, "entrezGeneIds": gene_chunk}
        data = request_json(
            "POST",
            f"{API_BASE}/molecular-profiles/{profile_id}/molecular-data/fetch",
            params={"projection": "DETAILED"},
            json_body=body,
        )
        rows.extend(data)
        print(f"  {profile_id}: chunk {i}/{math.ceil(len(entrez_ids) / gene_chunk_size)} rows={len(data)}")
    if not rows:
        return pd.DataFrame(index=samples)
    frame = pd.DataFrame(rows)
    frame["hugoGeneSymbol"] = frame["gene"].map(lambda x: x.get("hugoGeneSymbol") if isinstance(x, dict) else None)
    matrix = frame.pivot_table(index="sampleId", columns="hugoGeneSymbol", values="value", aggfunc="first")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.reindex(index=samples)
    matrix.index.name = "sampleId"
    return matrix


def load_methylation_probe_map(profile_id: str, genes: pd.DataFrame) -> pd.DataFrame:
    meta = request_json(
        "GET",
        f"{API_BASE}/generic-assay-meta/{profile_id}",
        params={"projection": "DETAILED", "pageSize": 1000000, "pageNumber": 0},
    )
    meta_frame = pd.DataFrame(meta)
    target_symbols = set(genes["hugoGeneSymbol"].astype(str))
    records: list[dict] = []
    for row in meta_frame.itertuples(index=False):
        props = getattr(row, "genericEntityMetaProperties", {}) or {}
        names = str(props.get("NAME", "")).replace(",", ";").split(";")
        description = str(props.get("DESCRIPTION", ""))
        transcript_id = str(props.get("TRANSCRIPT_ID", ""))
        for name in names:
            symbol = name.strip()
            if symbol in target_symbols:
                desc_upper = description.upper()
                score = 0
                if "TSS200" in desc_upper:
                    score += 5
                if "TSS1500" in desc_upper:
                    score += 4
                if "1STEXON" in desc_upper:
                    score += 3
                if "5'UTR" in desc_upper or "5UTR" in desc_upper:
                    score += 2
                if transcript_id and transcript_id != "NA":
                    score += 1
                records.append(
                    {
                        "hugoGeneSymbol": symbol,
                        "stableId": row.stableId,
                        "description": description,
                        "transcriptId": transcript_id,
                        "selection_score": score,
                    }
                )
    probe_map = pd.DataFrame(records)
    if probe_map.empty:
        return probe_map
    probe_map = (
        probe_map.sort_values(["hugoGeneSymbol", "selection_score", "stableId"], ascending=[True, False, True])
        .drop_duplicates("hugoGeneSymbol")
        .reset_index(drop=True)
    )
    return probe_map


def fetch_methylation_matrix(
    profile_id: str,
    samples: list[str],
    probe_map: pd.DataFrame,
    *,
    probe_chunk_size: int = 80,
) -> pd.DataFrame:
    if probe_map.empty:
        return pd.DataFrame(index=samples)
    rows: list[dict] = []
    stable_ids = probe_map["stableId"].tolist()
    for i, stable_chunk in enumerate(chunks(stable_ids, probe_chunk_size), start=1):
        body = {"sampleIds": samples, "genericAssayStableIds": stable_chunk}
        data = request_json(
            "POST",
            f"{API_BASE}/generic_assay_data/{profile_id}/fetch",
            params={"projection": "DETAILED"},
            json_body=body,
        )
        rows.extend(data)
        print(f"  {profile_id}: chunk {i}/{math.ceil(len(stable_ids) / probe_chunk_size)} rows={len(data)}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(index=samples)
    matrix = frame.pivot_table(index="sampleId", columns="genericAssayStableId", values="value", aggfunc="first")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    stable_to_gene = dict(zip(probe_map["stableId"], probe_map["hugoGeneSymbol"], strict=False))
    matrix = matrix.rename(columns=stable_to_gene)
    matrix = matrix.reindex(index=samples)
    matrix.index.name = "sampleId"
    return matrix


def fetch_mutation_matrix(
    profile_id: str,
    samples: list[str],
    genes: pd.DataFrame,
    *,
    gene_chunk_size: int = 80,
) -> pd.DataFrame:
    rows: list[dict] = []
    entrez_ids = genes["entrezGeneId"].astype(int).tolist()
    for i, gene_chunk in enumerate(chunks(entrez_ids, gene_chunk_size), start=1):
        body = {"sampleIds": samples, "entrezGeneIds": gene_chunk}
        data = request_json(
            "POST",
            f"{API_BASE}/molecular-profiles/{profile_id}/mutations/fetch",
            params={"projection": "SUMMARY", "pageSize": 10000000, "pageNumber": 0},
            json_body=body,
        )
        rows.extend(data)
        print(f"  {profile_id}: chunk {i}/{math.ceil(len(entrez_ids) / gene_chunk_size)} mutations={len(data)}")
    matrix = pd.DataFrame(0, index=samples, columns=genes["hugoGeneSymbol"].tolist(), dtype=np.int8)
    if rows:
        frame = pd.DataFrame(rows)
        gene_lookup = dict(zip(genes["entrezGeneId"], genes["hugoGeneSymbol"], strict=False))
        frame["hugoGeneSymbol"] = frame["entrezGeneId"].map(gene_lookup)
        for sample_id, gene in frame[["sampleId", "hugoGeneSymbol"]].dropna().drop_duplicates().itertuples(index=False):
            if sample_id in matrix.index and gene in matrix.columns:
                matrix.loc[sample_id, gene] = 1
    matrix.index.name = "sampleId"
    return matrix


def add_prefix(matrix: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = matrix.copy()
    renamed.columns = [f"{prefix}__{col}" for col in renamed.columns]
    return renamed


def read_matrix(path: Path, samples: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.reindex(index=samples)
    matrix.index.name = "sampleId"
    return matrix


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


def make_subtype_splits(model_table: pd.DataFrame, out_dir: Path, seed: int = 42) -> pd.DataFrame:
    table = model_table[model_table["SUBTYPE"].notna()].copy()
    table = table[~table["SUBTYPE"].astype(str).str.contains("NA|Unknown|Not", case=False, regex=True)]
    counts = table["SUBTYPE"].value_counts()
    keep = counts[counts >= 10].index
    table = table[table["SUBTYPE"].isin(keep)].copy()
    if table.empty or table["SUBTYPE"].nunique() < 2:
        split_frame = pd.DataFrame({"sampleId": model_table["sampleId"], "split": "unsplit"})
        split_frame.to_csv(out_dir / "brca_subtype_splits_70_15_15.csv", index=False)
        return split_frame

    train_ids, temp_ids = train_test_split(
        table["sampleId"],
        train_size=0.70,
        random_state=seed,
        stratify=table["SUBTYPE"],
    )
    temp = table.set_index("sampleId").loc[temp_ids].reset_index()
    valid_ids, test_ids = train_test_split(
        temp["sampleId"],
        train_size=0.50,
        random_state=seed,
        stratify=temp["SUBTYPE"],
    )
    split_frame = pd.DataFrame(
        {
            "sampleId": list(train_ids) + list(valid_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["valid"] * len(valid_ids) + ["test"] * len(test_ids),
        }
    )
    split_frame = model_table[["sampleId"]].merge(split_frame, on="sampleId", how="left")
    split_frame["split"] = split_frame["split"].fillna("unused")
    split_frame.to_csv(out_dir / "brca_subtype_splits_70_15_15.csv", index=False)
    return split_frame


def write_report(
    report_path: Path,
    samples: pd.DataFrame,
    genes: pd.DataFrame,
    probe_map: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    model_table: pd.DataFrame,
    split_frame: pd.DataFrame,
) -> None:
    subtype_counts = model_table["SUBTYPE"].fillna("NA").value_counts().rename_axis("SUBTYPE").reset_index(name="n")
    split_counts = split_frame["split"].value_counts().rename_axis("split").reset_index(name="n")
    modality_rows = []
    for name, matrix in matrices.items():
        modality_rows.append(
            {
                "modality": name,
                "samples": matrix.shape[0],
                "features": matrix.shape[1],
                "observed_fraction": float(matrix.notna().to_numpy().mean()) if matrix.size else 0.0,
            }
        )
    modality_summary = pd.DataFrame(modality_rows)

    lines = [
        "# TCGA BRCA Multi-omics Crawl Report v0",
        "",
        "## Source",
        "",
        f"- Study: `{STUDY_ID}` from cBioPortal public API.",
        f"- Gene panel: `{GENE_PANEL_ID}` cancer gene panel.",
        "- Profiles: mRNA RSEM, GISTIC CNA, log2 CNA, HM450 methylation, RPPA, mutation.",
        "- API root: https://www.cbioportal.org/api",
        "",
        "## Dataset Summary",
        "",
        f"- Samples discovered: {len(samples)}",
        f"- Primary tumor samples retained: {model_table.shape[0]}",
        f"- Cancer genes requested: {genes.shape[0]}",
        f"- Methylation genes with selected HM450 probe: {probe_map.shape[0]}",
        f"- Final modeling table shape: {model_table.shape[0]} samples x {model_table.shape[1]} columns",
        "",
        "## Modality Matrices",
        "",
        markdown_table(modality_summary, floatfmt=".4f"),
        "",
        "## SUBTYPE Counts",
        "",
        markdown_table(subtype_counts),
        "",
        "## Split Counts",
        "",
        markdown_table(split_counts),
        "",
        "## Output Files",
        "",
        "- `work/data/tcga_brca_cbioportal/processed/brca_mrna_impact468_matrix.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_gistic_impact468_matrix.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_log2cna_impact468_matrix.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_methylation_impact468_matrix.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_rppa_impact468_matrix.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_mutation_impact468_matrix.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_multimodal_impact468_table.csv`",
        "- `work/data/tcga_brca_cbioportal/processed/brca_subtype_splits_70_15_15.csv`",
        "",
        "## Modeling Note",
        "",
        "This first crawl is designed for breast cancer subtype prediction and multi-omics fusion experiments. "
        "The liquid/CfC model can treat modalities as a short ordered sequence of omics views, while MLP, logistic, "
        "random forest, LSTM/GRU, and TCN should remain baselines.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="work/data/tcga_brca_cbioportal")
    parser.add_argument("--report", default="outputs/tcga_brca_multiomics_crawl_report_v0.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    print("Fetching samples...")
    samples = pd.DataFrame(
        page_get(f"/studies/{STUDY_ID}/samples", params={"projection": "SUMMARY"}, page_size=2000)
    )
    samples.to_csv(raw_dir / "brca_samples.csv", index=False)
    primary_samples = samples[samples["sampleType"].eq("Primary Solid Tumor")].copy()
    sample_ids = primary_samples["sampleId"].tolist()
    patient_ids = sorted(primary_samples["patientId"].dropna().unique())
    print(f"Samples: {samples.shape[0]} total, {len(sample_ids)} primary tumor")

    print("Fetching clinical data...")
    patient_clinical = fetch_clinical(STUDY_ID, patient_ids, PATIENT_CLINICAL_ATTRS, "PATIENT")
    sample_clinical = fetch_clinical(STUDY_ID, sample_ids, SAMPLE_CLINICAL_ATTRS, "SAMPLE")
    patient_clinical.to_csv(raw_dir / "brca_patient_clinical_selected.csv", index=False)
    sample_clinical.to_csv(raw_dir / "brca_sample_clinical_selected.csv", index=False)

    print("Fetching gene panel...")
    genes = get_gene_panel(GENE_PANEL_ID)
    genes.to_csv(raw_dir / "impact468_genes.csv", index=False)

    matrices: dict[str, pd.DataFrame] = {}
    for modality in ["mrna", "gistic", "log2cna", "rppa"]:
        print(f"Fetching {modality}...")
        matrix_path = processed_dir / f"brca_{modality}_impact468_matrix.csv"
        if matrix_path.exists():
            print(f"  loading existing {matrix_path}")
            matrix = read_matrix(matrix_path, sample_ids)
        else:
            matrix = fetch_molecular_matrix(PROFILES[modality], sample_ids, genes)
            matrix.to_csv(matrix_path)
        matrices[modality] = matrix

    print("Building methylation probe map...")
    probe_map_path = processed_dir / "brca_methylation_probe_map_impact468.csv"
    if probe_map_path.exists():
        print(f"  loading existing {probe_map_path}")
        probe_map = pd.read_csv(probe_map_path)
    else:
        probe_map = load_methylation_probe_map(PROFILES["methylation"], genes)
        probe_map.to_csv(probe_map_path, index=False)
    print(f"Methylation probes selected: {probe_map.shape[0]}")

    print("Fetching methylation...")
    methylation_path = processed_dir / "brca_methylation_impact468_matrix.csv"
    if methylation_path.exists():
        print(f"  loading existing {methylation_path}")
        methylation_matrix = read_matrix(methylation_path, sample_ids)
    else:
        methylation_matrix = fetch_methylation_matrix(PROFILES["methylation"], sample_ids, probe_map)
        methylation_matrix.to_csv(methylation_path)
    matrices["methylation"] = methylation_matrix

    print("Fetching mutations...")
    mutation_path = processed_dir / "brca_mutation_impact468_matrix.csv"
    if mutation_path.exists():
        print(f"  loading existing {mutation_path}")
        mutation_matrix = read_matrix(mutation_path, sample_ids)
    else:
        mutation_matrix = fetch_mutation_matrix(PROFILES["mutation"], sample_ids, genes)
        mutation_matrix.to_csv(mutation_path)
    matrices["mutation"] = mutation_matrix

    print("Assembling multimodal table...")
    clinical = primary_samples[["sampleId", "patientId", "sampleType"]].merge(
        patient_clinical, on="patientId", how="left"
    )
    clinical = clinical.merge(sample_clinical, on="sampleId", how="left")
    feature_blocks = [
        add_prefix(matrices["mrna"], "mrna"),
        add_prefix(matrices["gistic"], "gistic"),
        add_prefix(matrices["log2cna"], "log2cna"),
        add_prefix(matrices["methylation"], "methylation"),
        add_prefix(matrices["rppa"], "rppa"),
        add_prefix(matrices["mutation"], "mutation"),
    ]
    features = pd.concat(feature_blocks, axis=1).reset_index()
    model_table = clinical.merge(features, on="sampleId", how="left")
    model_table.to_csv(processed_dir / "brca_multimodal_impact468_table.csv", index=False)

    split_frame = make_subtype_splits(model_table, processed_dir, args.seed)
    write_report(
        Path(args.report),
        samples,
        genes,
        probe_map,
        matrices,
        model_table,
        split_frame,
    )
    print(f"Done. Report: {args.report}")


if __name__ == "__main__":
    main()
