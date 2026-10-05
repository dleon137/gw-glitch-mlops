"""
Regression test for the Gravity Spy column-collision bug found on
2026-10-04: the raw CSV has its own integer-GPS `peak_time` column,
distinct from `event_time` (full-precision float GPS time). An earlier
version of load_glitch_labels() renamed event_time -> peak_time directly,
silently colliding with the existing column.

This test uses a small fixture shaped exactly like a real Gravity Spy row
(see tests/fixtures/gravityspy_sample.csv) so it runs offline and still
catches a regression if the rename logic is reintroduced.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data import load_glitch_labels

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gravityspy_sample.csv"


@pytest.fixture
def mock_gravity_spy(monkeypatch):
    raw = pd.read_csv(FIXTURE_PATH)

    def _fake_download(detector="H1", force=False):
        return raw[raw["ifo"] == detector].copy()

    monkeypatch.setattr("src.data._download_gravity_spy_metadata", _fake_download)


def test_load_glitch_labels_uses_event_time_not_raw_peak_time(mock_gravity_spy):
    """
    peak_time in the returned frame must come from event_time (float,
    full precision), never from the raw integer-GPS peak_time column --
    regression guard for the collision bug.
    """
    raw = pd.read_csv(FIXTURE_PATH)
    df = load_glitch_labels(detector="H1", min_confidence=0.0, n_per_class=10)

    assert not df.empty
    # peak_time values should match event_time's fractional-second precision,
    # not the truncated integer-GPS peak_time column from the raw file.
    for _, row in df.iterrows():
        matching_raw = raw[(raw["ifo"] == "H1") & (raw["ml_label"] == row["label"])]
        assert any(
            abs(float(row["peak_time"]) - et) < 1e-6 for et in matching_raw["event_time"]
        ), "peak_time does not match any event_time value -- collision regression?"


def test_load_glitch_labels_respects_min_confidence(mock_gravity_spy):
    df = load_glitch_labels(detector="H1", min_confidence=0.999, n_per_class=10)
    assert len(df) <= 1  # fixture has exactly one row >= 0.999 confidence


def test_load_glitch_labels_columns(mock_gravity_spy):
    df = load_glitch_labels(detector="H1", min_confidence=0.0, n_per_class=10)
    assert list(df.columns) == ["peak_time", "ifo", "label"]
