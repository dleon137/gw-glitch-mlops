"""
Unit tests for src/data.py.

Most of this module hits the network (GWOSC, Zenodo), so those tests are
marked and skipped by default in CI. Run them explicitly with:
    pytest tests/test_data.py -m network
"""

import numpy as np
import pytest

from src.data import PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLE_RATE_HZ, download_strain_segment, load_glitch_labels


def test_data_dirs_exist():
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()


@pytest.mark.network
def test_download_strain_segment_returns_expected_length():
    segment = download_strain_segment(gps_start=1126259460, duration=4, detector="H1")
    assert isinstance(segment, np.ndarray)
    # Allow a little slack for resampling edge effects
    assert abs(len(segment) - 4 * SAMPLE_RATE_HZ) < 10


@pytest.mark.network
def test_glitch_labels_have_expected_columns():
    df = load_glitch_labels(detector="H1", n_per_class=2)
    assert {"peak_time", "ifo", "label"}.issubset(df.columns)
    assert len(df) > 0
