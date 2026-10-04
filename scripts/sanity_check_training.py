"""
Offline sanity check for the training pipeline, using synthetic data in
place of real GWOSC/Gravity Spy downloads.

This does NOT validate the network-dependent parts of src/data.py
(download_strain_segment, load_glitch_labels, build_dataset) — those need
real network access to GWOSC/Zenodo, which this sandbox doesn't have.
What it DOES validate, against the real code in src/train.py and
src/model.py: dataset loading, feature extraction, label encoding,
train/val split, model fitting, metric computation, and model saving.

It writes a synthetic dataset directly to the same path src/train.py
expects (data/processed/dataset.npz) so that `python -m src.train` with
`--rebuild-data` omitted will pick it up and skip the (unavailable)
network download step.

Usage:
    python scripts/sanity_check_training.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import PROCESSED_DATA_DIR, SAMPLE_RATE_HZ, SEGMENT_DURATION_SEC
from src.train import main as train_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_PER_CLASS = 15
SEGMENT_LEN = SEGMENT_DURATION_SEC * SAMPLE_RATE_HZ


def make_synthetic_glitch(rng: np.random.Generator) -> np.ndarray:
    """Noise + a sharp, high-amplitude transient burst, like a real glitch."""
    seg = rng.normal(scale=1.0, size=SEGMENT_LEN).astype(np.float32)
    burst_center = rng.integers(SEGMENT_LEN // 4, 3 * SEGMENT_LEN // 4)
    burst_width = rng.integers(50, 300)
    t = np.arange(SEGMENT_LEN)
    envelope = np.exp(-0.5 * ((t - burst_center) / burst_width) ** 2)
    burst = rng.normal(scale=15.0, size=SEGMENT_LEN) * envelope
    return (seg + burst).astype(np.float32)


def make_synthetic_signal(rng: np.random.Generator) -> np.ndarray:
    """Quiet broadband noise only, no transient — stand-in for a 'clean' segment."""
    return rng.normal(scale=1.0, size=SEGMENT_LEN).astype(np.float32)


def build_synthetic_dataset() -> Path:
    rng = np.random.default_rng(42)

    X, y, gps_times, sources = [], [], [], []
    glitch_classes = ["Blip", "Koi_Fish", "Whistle"]

    for i in range(N_PER_CLASS):
        X.append(make_synthetic_glitch(rng))
        y.append("glitch")
        gps_times.append(1_126_000_000.0 + i)
        sources.append(glitch_classes[i % len(glitch_classes)])

    for i in range(N_PER_CLASS):
        X.append(make_synthetic_signal(rng))
        y.append("signal")
        gps_times.append(1_187_000_000.0 + i)
        sources.append("signal")

    X = np.stack(X)
    y = np.array(y)
    gps_times = np.array(gps_times)
    sources = np.array(sources, dtype=object)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "dataset.npz"
    np.savez_compressed(output_path, X=X, y=y, gps_times=gps_times, sources=sources)
    logger.info("Wrote synthetic dataset: %d examples -> %s", len(y), output_path)
    return output_path


def main():
    logger.info("=== Building synthetic dataset (standing in for GWOSC/Gravity Spy) ===")
    build_synthetic_dataset()

    logger.info("=== Running real src.train.main() against synthetic data ===")
    config_path = str(Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml")
    metrics = train_main(config_path, rebuild_data=False)

    logger.info("=== Sanity check result ===")
    logger.info("Metrics: %s", metrics)

    if metrics.get("accuracy", 0) < 0.8:
        logger.warning(
            "Accuracy is lower than expected for a trivially separable synthetic "
            "task — worth double-checking the feature/training pipeline."
        )
    else:
        logger.info(
            "PASS: training pipeline runs end-to-end and recovers the (deliberately "
            "easy) synthetic signal/glitch distinction."
        )


if __name__ == "__main__":
    main()
