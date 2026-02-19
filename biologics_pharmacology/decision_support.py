"""Decision-support utilities for candidate prioritization."""

from __future__ import annotations

import pandas as pd

from .modeling import BiologicsModelArtifacts


def _minmax_scale(values: pd.Series) -> pd.Series:
    """Scale values to [0, 1] with a safe fallback."""
    min_value = float(values.min())
    max_value = float(values.max())
    span = max_value - min_value
    if span == 0:
        return pd.Series(0.5, index=values.index)
    return (values - min_value) / span


def rank_candidates(
    artifacts: BiologicsModelArtifacts,
    candidates: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Predict clinical pharmacology outcomes and rank candidate biologics.

    A simple multi-objective score combines efficacy, PK durability,
    and safety risk into one sortable value.
    """
    required = set(artifacts.feature_columns)
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(
            "Candidates are missing required feature columns: "
            + ", ".join(missing)
        )

    predictions = pd.DataFrame(
        artifacts.pipeline.predict(candidates[artifacts.feature_columns]),
        columns=artifacts.target_columns,
        index=candidates.index,
    )

    scored = candidates.copy()
    for column in artifacts.target_columns:
        scored[f"pred_{column}"] = predictions[column]

    pd_score = _minmax_scale(scored["pred_clinical_pd_response_pct"])
    pk_score = _minmax_scale(scored["pred_clinical_pk_half_life_day"])
    safety_score = 1.0 - _minmax_scale(scored["pred_severe_ae_rate_pct"])
    scored["decision_score"] = (
        0.5 * pd_score + 0.3 * pk_score + 0.2 * safety_score
    )

    return scored.sort_values("decision_score", ascending=False).head(top_k)
