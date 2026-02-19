"""Dataset schema definitions for biologics pharmacology modeling."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

NUMERIC_FEATURES = [
    "molecular_weight_kda",
    "isoelectric_point",
    "hydrophobicity_index",
    "aggregation_propensity",
    "fc_receptor_binding_ratio",
    "binding_affinity_nm",
    "in_vitro_potency_ec50_nm",
    "preclinical_clearance_ml_day_kg",
    "preclinical_half_life_day",
    "preclinical_efficacy_pct",
    "ada_incidence_pct",
    "dose_mg_kg",
    "patient_baseline_biomarker",
]

CATEGORICAL_FEATURES = [
    "modality",
    "therapeutic_area",
    "administration_route",
    "development_stage",
    "species_model",
]

TARGET_COLUMNS = [
    "clinical_pk_half_life_day",
    "clinical_pd_response_pct",
    "severe_ae_rate_pct",
]

GROUP_COLUMN = "molecule_family"
ID_COLUMN = "molecule_id"


def required_columns() -> list[str]:
    """Return all columns required for training and evaluation."""
    return (
        [ID_COLUMN, GROUP_COLUMN]
        + CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
        + TARGET_COLUMNS
    )


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Raise an error when the dataframe is missing required columns."""
    missing = sorted(set(required_columns()) - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )


def feature_columns() -> list[str]:
    """Return the full list of model feature columns."""
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES


def ensure_non_empty(df: pd.DataFrame, name: str) -> None:
    """Raise a helpful error when expected datasets are empty."""
    if df.empty:
        raise ValueError(f"{name} is empty. Provide at least one row.")


def ensure_unique_groups(groups: Iterable[str], min_groups: int = 2) -> None:
    """Ensure grouped validation has enough independent molecule families."""
    count = len(set(groups))
    if count < min_groups:
        raise ValueError(
            f"Need at least {min_groups} unique groups, found {count}."
        )
