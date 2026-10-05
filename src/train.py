"""
Training entrypoint.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --rebuild-data

MLflow tracking will be added in Week 3; this version prints metrics and
saves the fitted model + feature list to models/ via joblib so later weeks
have something concrete to wrap in a registry/serving layer.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import PROCESSED_DATA_DIR, build_dataset, load_dataset
from src.model import FEATURE_NAMES, build_baseline_model, featurize_dataset

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str, rebuild_data: bool = False):
    logging.basicConfig(level=logging.INFO)
    config = load_config(config_path)

    dataset_path = PROCESSED_DATA_DIR / "dataset.npz"
    if rebuild_data or not dataset_path.exists():
        logger.info("Building dataset (this downloads real GWOSC strain data)...")
        build_dataset(
            detector=config["data"]["detector"],
            duration=config["data"]["segment_duration_sec"],
            sample_rate=config["data"]["sample_rate"],
            n_per_glitch_class=config["data"].get("n_per_glitch_class", 5),
            output_path=dataset_path,
        )

    X_raw, y_raw, gps_times, sources = load_dataset(dataset_path)
    logger.info("Loaded dataset: %d examples, classes=%s", len(y_raw), set(y_raw))

    X = featurize_dataset(X_raw, sample_rate=config["data"]["sample_rate"])

    # Fail fast with a clear message if any feature is non-finite, rather
    # than letting it reach sklearn's much less specific "Input X contains
    # NaN" error. build_dataset() already filters non-finite raw strain
    # segments at ingestion (see src/data.py), so this should not trigger
    # in normal operation -- it's a guard against a *different* bad
    # dataset.npz (e.g. one built before that fix) rather than a fix in
    # itself.
    if not np.all(np.isfinite(X)):
        n_bad_rows = int(np.sum(~np.isfinite(X).all(axis=1)))
        raise ValueError(
            f"{n_bad_rows} of {len(X)} feature rows contain NaN/Inf after "
            "featurization. This dataset.npz may predate the non-finite-"
            "segment filter in build_dataset() -- rerun with --rebuild-data "
            "to regenerate it."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.get("val_size", 0.2),
        random_state=config.get("seed", 42),
        stratify=y if len(set(y)) > 1 else None,
    )

    model = build_baseline_model(**config.get("model_params", {}))
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    metrics = {"accuracy": accuracy_score(y_val, preds)}

    # f1 / roc_auc need both classes present in y_val to be meaningful
    if len(set(y_val)) > 1:
        probs = model.predict_proba(X_val)[:, 1]
        metrics["f1"] = f1_score(y_val, preds)
        metrics["roc_auc"] = roc_auc_score(y_val, probs)

    logger.info("Validation metrics: %s", metrics)
    print(classification_report(y_val, preds, target_names=label_encoder.classes_))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_names": FEATURE_NAMES,
            "config": config,
            "metrics": metrics,
        },
        MODELS_DIR / "baseline_model.joblib",
    )
    logger.info("Saved model to %s", MODELS_DIR / "baseline_model.joblib")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Force re-download/rebuild of the dataset even if a cached one exists.",
    )
    args = parser.parse_args()
    main(args.config, rebuild_data=args.rebuild_data)
