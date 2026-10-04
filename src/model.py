"""
Feature extraction + model definition for the GW glitch classifier.

Week 1 goal: get a simple, honest baseline working — hand-engineered
features from the raw strain segment, fed into a RandomForest. A
spectrogram + CNN model is a reasonable Week 2+ upgrade once the pipeline
around it (tests, tracking, CI, serving) is solid end-to-end.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from sklearn.ensemble import RandomForestClassifier

from src.data import SAMPLE_RATE_HZ


def extract_features(segment: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """
    Extract a small set of time- and frequency-domain features from a raw
    strain segment. Deliberately simple for a first baseline — swap for
    learned spectrogram features later.

    Returns a 1D feature vector.
    """
    seg = segment.astype(np.float64)
    seg = seg - seg.mean()

    # Time-domain stats
    rms = float(np.sqrt(np.mean(seg**2)))
    peak = float(np.max(np.abs(seg)))
    crest_factor = peak / rms if rms > 0 else 0.0
    kurtosis = float(_kurtosis(seg))
    zero_crossing_rate = float(np.mean(np.diff(np.sign(seg)) != 0))

    # Frequency-domain: power in a few bands relevant to LIGO noise/signal content
    freqs, psd = sp_signal.welch(seg, fs=sample_rate, nperseg=min(1024, len(seg)))
    band_powers = [
        _band_power(freqs, psd, lo, hi)
        for lo, hi in [(10, 50), (50, 150), (150, 500), (500, 1500)]
    ]
    peak_freq = float(freqs[np.argmax(psd)])

    return np.array(
        [rms, peak, crest_factor, kurtosis, zero_crossing_rate, peak_freq, *band_powers],
        dtype=np.float32,
    )


def _kurtosis(x: np.ndarray) -> float:
    std = x.std()
    if std == 0:
        return 0.0
    return float(np.mean((x - x.mean()) ** 4) / std**4 - 3.0)


_trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in NumPy 2.0+


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(_trapz(psd[mask], freqs[mask])) if mask.any() else 0.0


def featurize_dataset(X_raw: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Apply extract_features() to every row of a raw strain dataset."""
    return np.stack([extract_features(row, sample_rate) for row in X_raw])


def build_baseline_model(**kwargs) -> RandomForestClassifier:
    """Construct the baseline classifier."""
    defaults = dict(n_estimators=200, max_depth=None, random_state=42, class_weight="balanced")
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


FEATURE_NAMES = [
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "zero_crossing_rate",
    "peak_freq",
    "band_power_10_50hz",
    "band_power_50_150hz",
    "band_power_150_500hz",
    "band_power_500_1500hz",
]
