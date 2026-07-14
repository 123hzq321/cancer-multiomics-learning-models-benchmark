from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_metabric_marker_panel_external_validation import (  # noqa: E402
    DATA_DIR,
    OUTPUTS_DIR,
    add_feature_set,
    assemble_marker_tables,
    fit_sklearn,
    loader_dict,
    make_processed,
    parse_ints,
    summarize_by_split,
)
from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    EarlyFusionMLP,
    EarlyFusionResMLP,
    ModalityAttentionFusion,
    ModalityDeepSets,
    ModalityGRU,
    ModalityGatedLateFusion,
    ModalityLSTM,
    ModalityLiquidCfC,
    ModalityTCN,
    ModalityTransformer,
    markdown_table,
    rows_from_result,
    save_predictions,
    set_seed,
    train_torch_model,
)


def extended_torch_models(
    feature_set: str,
    input_dims: list[int],
    fusion_dim: int,
    num_classes: int,
) -> list[tuple[str, str, torch.nn.Module]]:
    prefix = f"{feature_set}__"
    models: list[tuple[str, str, torch.nn.Module]] = [
        (f"{prefix}mlp", "static_nn", EarlyFusionMLP(fusion_dim, num_classes, hidden_dim=128)),
        (f"{prefix}resmlp", "static_nn", EarlyFusionResMLP(fusion_dim, num_classes, hidden_dim=128, blocks=3)),
        (f"{prefix}deepsets", "set_nn", ModalityDeepSets(input_dims, num_classes, embed_dim=64, hidden_dim=128)),
        (f"{prefix}attention_fusion", "attention_nn", ModalityAttentionFusion(input_dims, num_classes, embed_dim=64, hidden_dim=128)),
        (f"{prefix}transformer", "attention_nn", ModalityTransformer(input_dims, num_classes, embed_dim=64, hidden_dim=128)),
        (f"{prefix}gated_late_fusion", "late_fusion_nn", ModalityGatedLateFusion(input_dims, num_classes, embed_dim=64)),
    ]
    if len(input_dims) > 1:
        models.extend(
            [
                (f"{prefix}gru", "sequence_nn", ModalityGRU(input_dims, num_classes, embed_dim=64, hidden_dim=96)),
                (f"{prefix}lstm", "sequence_nn", ModalityLSTM(input_dims, num_classes, embed_dim=64, hidden_dim=96)),
                (f"{prefix}tcn", "sequence_nn", ModalityTCN(input_dims, num_classes, embed_dim=64, hidden_dim=96)),
                (f"{prefix}liquid_cfc", "liquid_nn", ModalityLiquidCfC(input_dims, num_classes, embed_dim=64, hidden_dim=96)),
            ]
        )
    return models


def write_report(
    path: Path,
    summary: pd.DataFrame,
    best_model: str,
    best_confusion: pd.DataFrame,
    best_report: pd.DataFrame,
    seeds: list[int],
) -> None:
    external = summary[summary["evaluation_split"].eq("metabric_external")].copy()
    external = external.sort_values(["f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=False)
    tcga = summary[summary["evaluation_split"].eq("tcga_test")].copy()
    tcga = tcga.sort_values(["f1_macro_mean", "roc_auc_ovr_macro_mean"], ascending=False)
    lines = [
        "# METABRIC Marker Panel Extended Neural Network Validation v0",
        "",
        "## Scope",
        "",
        "Extended neural baselines for TCGA-trained BRCA subtype prediction with METABRIC external validation.",
        "",
        f"- Seeds: `{','.join(map(str, seeds))}`",
        "- Feature sets: `marker_mrna` and `marker_multimodal_expression_last`.",
        "- Added NN families: residual MLP, DeepSets-style pooling, attention fusion, Transformer encoder, gated late fusion, GRU, LSTM, TCN, and Liquid/CfC.",
        "- Model selection: early stopping on TCGA validation split only; METABRIC is external hold-out.",
        "",
        "## METABRIC External Summary",
        "",
        markdown_table(external, floatfmt=".4f"),
        "",
        "## TCGA Internal Test Summary",
        "",
        markdown_table(tcga, floatfmt=".4f"),
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
        "This run tests whether the Liquid/CfC conclusion is robust after adding stronger neural alternatives. "
        "If mRNA-only MLP/ResMLP remains best externally, the publishable message should emphasize that BRCA subtype transfer is expression-marker dominated. "
        "If a multimodal attention/Transformer/Liquid model wins, the message can shift toward cross-omics fusion.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DATA_DIR / "breast_marker_panel"))
    parser.add_argument("--report", default=str(OUTPUTS_DIR / "metabric_breast_marker_extended_nn_validation_report_v0.md"))
    parser.add_argument("--results-csv", default=str(DATA_DIR / "breast_marker_panel" / "marker_external_extended_nn_model_results.csv"))
    parser.add_argument("--summary-csv", default=str(DATA_DIR / "breast_marker_panel" / "marker_external_extended_nn_model_summary.csv"))
    parser.add_argument("--predictions-csv", default=str(DATA_DIR / "breast_marker_panel" / "marker_external_extended_nn_model_predictions.csv"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    device = torch.device(args.device)

    combined, modality_cols, label_encoder = assemble_marker_tables(out_dir)
    feature_sets = {
        "marker_mrna": {"mrna_z": modality_cols["mrna_z"]},
        "marker_multimodal_expression_last": {
            "cna": modality_cols["cna"],
            "mutation": modality_cols["mutation"],
            "mrna_z": modality_cols["mrna_z"],
        },
    }

    all_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for feature_set, selected in feature_sets.items():
        print(f"Feature set: {feature_set}")
        processed = make_processed(combined, selected)
        input_dims = [processed["train"]["modalities"][idx].shape[1] for idx in range(len(selected))]
        fusion_dim = processed["train"]["fusion"].shape[1]
        loaders = loader_dict(processed, args.batch_size)

        for seed in seeds:
            set_seed(seed)
            print(f"  seed={seed} sklearn")
            for result in fit_sklearn(processed, seed, model_prefix=f"{feature_set}__"):
                rows = add_feature_set(rows_from_result(seed, result["model"], result["model_group"], result), feature_set)
                all_rows.extend(rows)
                for split, (y_true, proba) in result["predictions"].items():
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "model": result["model"],
                            "split": split,
                            "sample_ids": processed[split]["sampleId"],
                            "y_true": y_true,
                            "proba": proba,
                        }
                    )

            for model_name, group, model in extended_torch_models(feature_set, input_dims, fusion_dim, len(label_encoder.classes_)):
                set_seed(seed)
                print(f"  seed={seed} {model_name}")
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
                rows = add_feature_set(rows_from_result(seed, model_name, group, result), feature_set)
                all_rows.extend(rows)
                for split, (y_true, proba) in result["predictions"].items():
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "split": split,
                            "sample_ids": processed[split]["sampleId"],
                            "y_true": y_true,
                            "proba": proba,
                        }
                    )

    results = pd.DataFrame(all_rows)
    summary = summarize_by_split(results)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    save_predictions(prediction_rows, Path(args.predictions_csv), label_encoder)

    external = summary[summary["evaluation_split"].eq("metabric_external")].sort_values(
        ["f1_macro_mean", "roc_auc_ovr_macro_mean"],
        ascending=False,
    )
    best_model = str(external.iloc[0]["model"])
    best_seed = (
        results[results["split"].eq("metabric_external") & results["model"].eq(best_model)]
        .sort_values("f1_macro", ascending=False)
        .iloc[0]["seed"]
    )
    pred = pd.read_csv(args.predictions_csv)
    best_pred = pred[pred["split"].eq("metabric_external") & pred["model"].eq(best_model) & pred["seed"].eq(best_seed)]
    labels = list(label_encoder.classes_)
    cm = pd.DataFrame(
        confusion_matrix(best_pred["true_label"], best_pred["pred_label"], labels=labels),
        index=[f"true__{x}" for x in labels],
        columns=[f"pred__{x}" for x in labels],
    )
    cls = pd.DataFrame(classification_report(best_pred["true_label"], best_pred["pred_label"], output_dict=True, zero_division=0)).T.reset_index(names="label")
    cm.to_csv(out_dir / "marker_external_extended_nn_best_confusion_matrix.csv")
    cls.to_csv(out_dir / "marker_external_extended_nn_best_classification_report.csv", index=False)
    write_report(Path(args.report), summary, best_model, cm.reset_index(names="true_label"), cls, seeds)
    print(f"Done. Report: {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
