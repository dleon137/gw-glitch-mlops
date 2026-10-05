"""
Regression test for the NaN-in-strain-segment bug found on 2026-10-04:
a real GWOSC fetch against live data produced a RandomForestClassifier
crash (`ValueError: Input X contains NaN`) from Gravity Spy glitch
segments that contained non-finite strain samples within an otherwise
correctly-sized window (likely detector lock-loss / data-quality gaps).

Covers both layers of the fix:
1. build_dataset() filters non-finite raw segments at ingestion
   (see src/data.py) -- not covered here directly since it requires a
   live download; see the manual verification in the fix commit instead.
2. train.py's fail-fast check catches non-finite *features* with a clear
   error message, as a guard against a stale/bad dataset.npz.
"""

import numpy as np
import pytest

from src.model import featurize_dataset


def test_nan_in_segment_produces_nonfinite_features():
    """
    Confirms a NaN gap in raw strain actually propagates into the feature
    vector -- the premise the train.py fail-fast guard depends on.
    """
    rng = np.random.default_rng(0)
    seg = rng.normal(size=4 * 4096).astype(np.float32)
    seg[1000:1010] = np.nan

    X = featurize_dataset(np.stack([seg]), sample_rate=4096)
    assert not np.all(np.isfinite(X))


def test_clean_segment_produces_finite_features():
    """Sanity counterpart: a clean segment should never trip the guard."""
    rng = np.random.default_rng(0)
    seg = rng.normal(size=4 * 4096).astype(np.float32)

    X = featurize_dataset(np.stack([seg]), sample_rate=4096)
    assert np.all(np.isfinite(X))


def test_train_main_raises_clear_error_on_nonfinite_features(tmp_path, monkeypatch):
    """
    If a dataset.npz with non-finite features somehow reaches train.main(),
    it should fail with our specific, actionable error -- not sklearn's
    generic one several stack frames deep.
    """
    import src.train as train_mod

    rng = np.random.default_rng(0)
    seg_bad = rng.normal(size=4 * 4096).astype(np.float32)
    seg_bad[0:5] = np.nan
    seg_good = rng.normal(size=4 * 4096).astype(np.float32)

    X_raw = np.stack([seg_bad, seg_good, seg_good, seg_bad])
    y_raw = np.array(["glitch", "signal", "glitch", "signal"])

    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        X=X_raw,
        y=y_raw,
        gps_times=np.arange(4, dtype=float),
        sources=np.array(["x"] * 4, dtype=object),
    )

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "seed: 42\nval_size: 0.5\ndata:\n  detector: H1\n  segment_duration_sec: 4\n"
        "  sample_rate: 4096\n  n_per_glitch_class: 5\nmodel_params:\n  n_estimators: 5\n"
    )

    # train.main() derives its dataset path from PROCESSED_DATA_DIR, so
    # pointing that at tmp_path (where we just wrote dataset.npz) lets the
    # real load_dataset() run unmodified -- no need to fake its return value.
    monkeypatch.setattr(train_mod, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="non-finite|NaN/Inf"):
        train_mod.main(str(config_path), rebuild_data=False)
