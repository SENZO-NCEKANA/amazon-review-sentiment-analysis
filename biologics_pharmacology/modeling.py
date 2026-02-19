"""Model training and grouped validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .schema import (
    CATEGORICAL_FEATURES,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMNS,
    ensure_non_empty,
    ensure_unique_groups,
    feature_columns,
    validate_dataset_schema,
)


@dataclass
class BiologicsModelArtifacts:
    """Serializable container for trained model objects and metadata."""

    pipeline: Pipeline
    feature_columns: list[str]
    target_columns: list[str]
    group_column: str


def build_pipeline() -> Pipeline:
    """Build a tabular multi-output regression pipeline."""
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    regressor = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def run_grouped_cross_validation(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Perform group-aware CV to prevent molecule family leakage.

    Returns one row per fold and target with MAE and R2 metrics.
    """
    validate_dataset_schema(df)
    ensure_non_empty(df, "Training dataframe")
    ensure_unique_groups(df[GROUP_COLUMN].tolist(), min_groups=2)

    n_unique_groups = df[GROUP_COLUMN].nunique()
    effective_splits = min(n_splits, n_unique_groups)
    if effective_splits < 2:
        raise ValueError("Grouped cross-validation requires at least 2 folds.")

    cv = GroupKFold(n_splits=effective_splits)
    X = df[feature_columns()]
    y = df[TARGET_COLUMNS]
    groups = df[GROUP_COLUMN]
    base_pipeline = build_pipeline()
    records: list[dict[str, float | int | str]] = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):
        model = clone(base_pipeline)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = pd.DataFrame(
            model.predict(X_test),
            columns=TARGET_COLUMNS,
            index=y_test.index,
        )

        for target in TARGET_COLUMNS:
            records.append(
                {
                    "fold": fold,
                    "target": target,
                    "mae": float(
                        mean_absolute_error(
                            y_test[target],
                            predictions[target],
                        )
                    ),
                    "r2": float(
                        r2_score(
                            y_test[target],
                            predictions[target],
                        )
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


def train_final_model(df: pd.DataFrame) -> BiologicsModelArtifacts:
    """Train the final model on all available data."""
    validate_dataset_schema(df)
    ensure_non_empty(df, "Training dataframe")

    model = build_pipeline()
    model.fit(df[feature_columns()], df[TARGET_COLUMNS])

    return BiologicsModelArtifacts(
        pipeline=model,
        feature_columns=feature_columns(),
        target_columns=TARGET_COLUMNS,
        group_column=GROUP_COLUMN,
    )
