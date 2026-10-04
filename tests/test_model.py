"""
Unit tests for src/model.py — feature extraction, which is pure and fast
(no network), so it's fully tested from Week 1 on. Data-download tests
live in test_data.py and are marked slow/skipped by default.
"""

import numpy as np
import pytest

from src.model import FEATURE_NAMES, build_baseline_model, extract_features, featurize_dataset


@pytest.fixture
def fake_segment():
    rng = np.random.default_rng(0)
    # 4 seconds at 4096 Hz, like a real GWOSC H1 segment
    return rng.normal(size=4 * 4096).astype(np.float32)


def test_extract_features_returns_expected_length(fake_segment):
    feats = extract_features(fake_segment, sample_rate=4096)
    assert feats.shape == (len(FEATURE_NAMES),)


def test_extract_features_no_nans_or_infs(fake_segment):
    feats = extract_features(fake_segment, sample_rate=4096)
    assert np.all(np.isfinite(feats))


def test_extract_features_handles_constant_segment():
    # Degenerate input (std = 0) shouldn't raise or produce NaNs
    segment = np.zeros(4 * 4096, dtype=np.float32)
    feats = extract_features(segment, sample_rate=4096)
    assert np.all(np.isfinite(feats))


def test_featurize_dataset_shape(fake_segment):
    X_raw = np.stack([fake_segment, fake_segment * 2.0])
    X = featurize_dataset(X_raw, sample_rate=4096)
    assert X.shape == (2, len(FEATURE_NAMES))


def test_build_baseline_model_is_fittable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, len(FEATURE_NAMES)))
    y = np.array([0, 1] * 10)
    model = build_baseline_model(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})
