"""Generate synthetic biologics pharmacology dataset for demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Generate synthetic structure, preclinical, and clinical records."""
    rng = np.random.default_rng(seed)

    modalities = ["mAb", "bispecific", "ADC", "fusion_protein", "peptide"]
    therapeutic_areas = [
        "oncology",
        "immunology",
        "neuroscience",
        "respiratory",
    ]
    routes = ["IV", "SC", "IM"]
    stages = ["discovery", "preclinical", "phase1", "phase2"]
    species_models = ["mouse", "rat", "NHP"]

    df = pd.DataFrame(
        {
            "molecule_id": [f"MOL-{i:04d}" for i in range(n_samples)],
            "molecule_family": [
                f"FAM-{i:03d}" for i in rng.integers(0, max(2, n_samples // 8), n_samples)
            ],
            "modality": rng.choice(
                modalities,
                size=n_samples,
                p=[0.38, 0.2, 0.18, 0.14, 0.1],
            ),
            "therapeutic_area": rng.choice(therapeutic_areas, size=n_samples),
            "administration_route": rng.choice(routes, size=n_samples, p=[0.6, 0.35, 0.05]),
            "development_stage": rng.choice(stages, size=n_samples, p=[0.35, 0.35, 0.2, 0.1]),
            "species_model": rng.choice(species_models, size=n_samples, p=[0.5, 0.25, 0.25]),
            "molecular_weight_kda": rng.normal(95, 26, n_samples).clip(12, 220),
            "isoelectric_point": rng.normal(7.2, 1.0, n_samples).clip(4.5, 10.5),
            "hydrophobicity_index": rng.normal(0.1, 0.9, n_samples).clip(-2.5, 2.5),
            "aggregation_propensity": rng.beta(2.1, 5.2, n_samples),
            "fc_receptor_binding_ratio": rng.lognormal(mean=0.0, sigma=0.38, size=n_samples),
            "binding_affinity_nm": rng.lognormal(mean=2.1, sigma=0.65, size=n_samples).clip(0.1, 250),
            "in_vitro_potency_ec50_nm": rng.lognormal(mean=2.3, sigma=0.7, size=n_samples).clip(0.2, 350),
            "preclinical_clearance_ml_day_kg": rng.normal(8.5, 3.2, n_samples).clip(1.0, 20),
            "preclinical_half_life_day": rng.normal(6.0, 2.4, n_samples).clip(0.5, 18),
            "preclinical_efficacy_pct": rng.normal(58, 16, n_samples).clip(5, 95),
            "ada_incidence_pct": rng.normal(14, 7.5, n_samples).clip(0, 55),
            "dose_mg_kg": rng.normal(6.5, 2.2, n_samples).clip(0.2, 20),
            "patient_baseline_biomarker": rng.normal(1.0, 0.4, n_samples).clip(0.1, 3.0),
        }
    )

    modality_effect = {
        "mAb": 4.0,
        "bispecific": 2.0,
        "ADC": -1.0,
        "fusion_protein": 3.0,
        "peptide": -3.5,
    }
    stage_effect = {
        "discovery": -2.0,
        "preclinical": -0.5,
        "phase1": 1.0,
        "phase2": 2.5,
    }

    clinical_pk_half_life = (
        3.0
        + 0.45 * df["preclinical_half_life_day"]
        + 0.03 * df["molecular_weight_kda"]
        - 0.62 * df["preclinical_clearance_ml_day_kg"]
        - 2.3 * df["aggregation_propensity"]
        + 1.8 * np.log(df["fc_receptor_binding_ratio"])
        + df["modality"].map(modality_effect)
        + rng.normal(0, 1.8, n_samples)
    ).clip(0.2, 35)

    clinical_pd_response = (
        18
        + 0.55 * df["preclinical_efficacy_pct"]
        - 0.12 * df["in_vitro_potency_ec50_nm"]
        - 0.06 * df["binding_affinity_nm"]
        + 0.8 * df["dose_mg_kg"]
        - 2.7 * df["hydrophobicity_index"]
        + df["development_stage"].map(stage_effect)
        + rng.normal(0, 6.5, n_samples)
    ).clip(0, 100)

    severe_ae_rate = (
        1.5
        + 0.33 * df["aggregation_propensity"] * 100 / 10
        + 0.08 * df["ada_incidence_pct"]
        + 0.45 * df["dose_mg_kg"]
        - 0.35 * df["patient_baseline_biomarker"]
        + np.where(df["modality"] == "ADC", 4.5, 0.0)
        + rng.normal(0, 1.5, n_samples)
    ).clip(0.1, 40)

    df["clinical_pk_half_life_day"] = clinical_pk_half_life
    df["clinical_pd_response_pct"] = clinical_pd_response
    df["severe_ae_rate_pct"] = severe_ae_rate

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic biologics pharmacology data."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/synthetic_biologics.csv"),
        help="Where to write the generated CSV.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of synthetic rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(n_samples=args.samples, seed=args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_path, index=False)
    print(f"Saved {len(dataset)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
