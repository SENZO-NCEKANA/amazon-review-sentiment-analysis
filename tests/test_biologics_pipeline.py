"""Basic tests for biologics pharmacology pipeline."""

from __future__ import annotations

from biologics_pharmacology.decision_support import rank_candidates
from biologics_pharmacology.modeling import (
    run_grouped_cross_validation,
    train_final_model,
)
from biologics_pharmacology.schema import feature_columns
from scripts.generate_synthetic_biologics_data import generate_dataset


def test_grouped_cross_validation_returns_metrics() -> None:
    data = generate_dataset(n_samples=120, seed=7)
    metrics = run_grouped_cross_validation(data, n_splits=4)

    assert not metrics.empty
    assert set(metrics.columns) == {"fold", "target", "mae", "r2"}
    assert metrics["fold"].nunique() == 4
    assert metrics["target"].nunique() == 3


def test_rank_candidates_outputs_requested_top_k() -> None:
    data = generate_dataset(n_samples=150, seed=21)
    artifacts = train_final_model(data)

    ranked = rank_candidates(
        artifacts=artifacts,
        candidates=data[feature_columns() + ["molecule_id"]].copy(),
        top_k=15,
    )

    assert len(ranked) == 15
    assert ranked["decision_score"].is_monotonic_decreasing
    assert "pred_clinical_pd_response_pct" in ranked.columns
