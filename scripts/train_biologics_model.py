"""Train and validate biologics pharmacology AI/ML models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biologics_pharmacology import (
    rank_candidates,
    run_grouped_cross_validation,
    train_final_model,
    validate_dataset_schema,
)
from biologics_pharmacology.schema import feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grouped, multi-output biologics models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/synthetic_biologics.csv"),
        help="Path to training data CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/biologics_multitask_model.joblib"),
        help="Path for serialized model artifacts.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top candidates to show.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data_path)
    validate_dataset_schema(data)

    metrics = run_grouped_cross_validation(data, n_splits=args.splits)
    metrics_summary = (
        metrics.groupby("target")[["mae", "r2"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\nGrouped cross-validation performance:")
    print(metrics_summary)

    artifacts = train_final_model(data)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, args.model_path)
    print(f"\nSaved trained model artifacts to: {args.model_path}")

    shortlisted = rank_candidates(
        artifacts=artifacts,
        candidates=data[feature_columns() + ["molecule_id"]].copy(),
        top_k=args.top_k,
    )
    print("\nTop candidates for early development decisions:")
    print(
        shortlisted[
            [
                "molecule_id",
                "pred_clinical_pk_half_life_day",
                "pred_clinical_pd_response_pct",
                "pred_severe_ae_rate_pct",
                "decision_score",
            ]
        ].round(3)
    )


if __name__ == "__main__":
    main()
