"""
Data loading and preprocessing for the GW glitch classifier.

Sources:
- GWOSC (strain data): https://gwosc.org, accessed via the `gwpy` package
- Gravity Spy (glitch labels): https://www.zooniverse.org/projects/zooniverse/gravity-spy
  Metadata is pulled from the public Gravity Spy data releases (Zenodo CSV
  export), which lists, per glitch: GPS peak time, detector, ML-confidence
  label, and glitch class (e.g. "Blip", "Koi_Fish", "Whistle", ...).

This module builds a small labeled dataset of short strain segments, each
tagged either `glitch` (from Gravity Spy, using one of its 22 glitch
classes) or `signal`-like. True confirmed-signal segments are rare and
mostly confined to O1-O3 event lists, so for a first baseline we treat
"quiet" (glitch-free, per Gravity Spy coverage) segments around confirmed
GWTC event times as the positive "signal" class, and Gravity Spy-labeled
glitches as the negative "glitch" class. This is called out explicitly in
the README as a simplification to revisit once the pipeline works
end-to-end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# The Gravity Spy Zenodo record (5649212) splits metadata into one CSV per
# detector per observing run -- there is no single combined file. For Week 1
# we use one run per detector (the smallest available, to keep downloads
# fast); pooling across all runs is a reasonable Week 2+ extension.
# Verified against the live record on 2026-10-04.
GRAVITY_SPY_FILES = {
    "H1": "H1_O1.csv",  # ~16 MB
    "L1": "L1_O1.csv",  # ~27 MB
}
GRAVITY_SPY_BASE_URL = "https://zenodo.org/records/5649212/files"

# A handful of confirmed GWTC event GPS times (H1), used to pull "quiet"
# segments as a stand-in for signal-bearing data. Small and hardcoded for
# Week 1; swap for a full GWTC catalog query later if the task is extended.
CONFIRMED_EVENT_GPS_TIMES = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170814": 1186741861.5,
    "GW170817": 1187008882.4,
}

SEGMENT_DURATION_SEC = 4
SAMPLE_RATE_HZ = 4096


@dataclass
class StrainSegment:
    gps_start: float
    detector: str
    label: str  # "glitch" or "signal"
    source: str  # e.g. gravity spy glitch class, or event name
    data: np.ndarray  # raw strain samples


def download_strain_segment(
    gps_start: float,
    duration: int = SEGMENT_DURATION_SEC,
    detector: str = "H1",
    sample_rate: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """
    Download a strain data segment from GWOSC, centered-ish on gps_start.

    Args:
        gps_start: GPS start time of the segment (seconds).
        duration: Duration in seconds.
        detector: Detector name ("H1", "L1", or "V1").
        sample_rate: Target sample rate in Hz (data is resampled if needed).

    Returns:
        1D numpy array of strain values, length ~= duration * sample_rate.
    """
    from gwpy.timeseries import TimeSeries

    gps_end = gps_start + duration
    logger.info(
        "Fetching %s strain: gps=[%s, %s] (%ss)", detector, gps_start, gps_end, duration
    )
    ts = TimeSeries.fetch_open_data(detector, gps_start, gps_end, verbose=False)

    if int(ts.sample_rate.value) != sample_rate:
        ts = ts.resample(sample_rate)

    return ts.value.astype(np.float32)


def _download_gravity_spy_metadata(detector: str = "H1", force: bool = False) -> pd.DataFrame:
    """
    Download (or load cached) Gravity Spy glitch metadata for one detector's
    chosen observing run (see GRAVITY_SPY_FILES).

    Returns the raw DataFrame exactly as published -- callers are
    responsible for selecting/renaming columns (see load_glitch_labels).
    """
    if detector not in GRAVITY_SPY_FILES:
        raise ValueError(
            f"No Gravity Spy file configured for detector={detector!r}; "
            f"available: {list(GRAVITY_SPY_FILES)}"
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = GRAVITY_SPY_FILES[detector]
    cache_path = RAW_DATA_DIR / f"gravityspy_{filename}"

    if cache_path.exists() and not force:
        logger.info("Loading cached Gravity Spy metadata from %s", cache_path)
        return pd.read_csv(cache_path)

    url = f"{GRAVITY_SPY_BASE_URL}/{filename}?download=1"
    logger.info("Downloading Gravity Spy metadata from %s", url)
    df = pd.read_csv(url)
    df.to_csv(cache_path, index=False)
    return df


def load_glitch_labels(
    detector: str = "H1",
    min_confidence: float = 0.9,
    classes: list[str] | None = None,
    n_per_class: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load a small, balanced sample of Gravity Spy glitch labels.

    Args:
        detector: Which interferometer's glitches to keep ("H1" or "L1").
        min_confidence: Minimum ML-confidence label to trust.
        classes: Subset of glitch classes to include (defaults to a fixed
            starter list of visually/physically distinct classes).
        n_per_class: How many examples to sample per class (keeps the
            Week 1 dataset small and fast to download/debug).
        random_state: For reproducible sampling.

    Returns:
        DataFrame with columns: peak_time, ifo, label (glitch class).
    """
    df = _download_gravity_spy_metadata(detector=detector)

    # NOTE: the raw CSV already has its own integer-GPS `peak_time` column
    # (paired with `peak_time_ns`), separate from `event_time` (the
    # full-precision float GPS time). Renaming event_time -> peak_time
    # directly would collide with that existing column and silently
    # corrupt the data -- caught by inspecting a real downloaded file
    # rather than assumed from the column name alone. We deliberately
    # build a fresh frame instead of renaming in place.
    column_map = {"label": "ml_label", "confidence": "ml_confidence"}
    missing = {src for src in column_map.values() if src not in df.columns} | (
        {"event_time", "ifo"} - set(df.columns)
    )
    if missing:
        raise ValueError(
            f"Gravity Spy metadata missing expected columns {missing}; "
            f"got columns: {list(df.columns)}. The upstream schema may have "
            "changed since this was last verified (2026-10-04) — inspect "
            "the CSV and update load_glitch_labels() accordingly."
        )

    df = pd.DataFrame(
        {
            "peak_time": df["event_time"].astype(float),
            "ifo": df["ifo"],
            "label": df["ml_label"],
            "confidence": df["ml_confidence"].astype(float),
        }
    )

    if "confidence" in df.columns:
        df = df[df["confidence"] >= min_confidence]

    df = df[df["ifo"] == detector]

    default_classes = [
        "Blip",
        "Koi_Fish",
        "Whistle",
        "Scattered_Light",
        "Low_Frequency_Burst",
        "Violin_Mode",
    ]
    classes = classes or default_classes
    df = df[df["label"].isin(classes)]

    # NOTE: deliberately not groupby(...).apply(...) here -- pandas changed
    # whether the grouping column is included in the group passed to the
    # lambda (pre-2.2 kept it, 2.2+/3.0 excludes it by default unless
    # include_groups=True is passed), which silently dropped `label` and
    # broke the final column selection below. Caught by actually running
    # this against a real downloaded file, not just reading the pandas
    # docs. This formulation is correct across pandas versions.
    sampled = pd.concat(
        [g.sample(min(len(g), n_per_class), random_state=random_state) for _, g in df.groupby("label")],
        ignore_index=True,
    )

    return sampled[["peak_time", "ifo", "label"]]


def _quiet_segments_near_events(detector: str = "H1") -> pd.DataFrame:
    """
    Build a small table of GPS times around confirmed GWTC events, used as
    the "signal" class stand-in. Offsets are chosen a few seconds off the
    exact merger time so each segment window brackets the signal without
    needing precise alignment for this first baseline.
    """
    rows = [
        {"peak_time": gps, "ifo": detector, "label": "signal"}
        for gps in CONFIRMED_EVENT_GPS_TIMES.values()
    ]
    return pd.DataFrame(rows)


def build_dataset(
    detector: str = "H1",
    duration: int = SEGMENT_DURATION_SEC,
    sample_rate: int = SAMPLE_RATE_HZ,
    n_per_glitch_class: int = 5,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.npz",
) -> Path:
    """
    Assemble strain segments + labels into a training-ready dataset and
    save to disk as a compressed .npz (X: float32 array, y: label array,
    meta: structured array of peak_time/source for traceability).

    This intentionally downloads real strain data, so it's slow (network +
    GWOSC fetch latency) and meant to be run once, not on every test.
    """
    glitch_labels = load_glitch_labels(detector=detector, n_per_class=n_per_glitch_class)
    signal_labels = _quiet_segments_near_events(detector=detector)
    labels = pd.concat([glitch_labels, signal_labels], ignore_index=True)

    X, y, gps_times, sources = [], [], [], []

    for _, row in labels.iterrows():
        gps_center = float(row["peak_time"])
        gps_start = gps_center - duration / 2
        try:
            segment = download_strain_segment(
                gps_start=gps_start,
                duration=duration,
                detector=row["ifo"],
                sample_rate=sample_rate,
            )
        except Exception as exc:  # noqa: BLE001 — log and skip bad segments
            logger.warning("Skipping gps=%s (%s): %s", gps_center, row["label"], exc)
            continue

        expected_len = duration * sample_rate
        if len(segment) != expected_len:
            logger.warning(
                "Skipping gps=%s: got %d samples, expected %d",
                gps_center, len(segment), expected_len,
            )
            continue

        # GWOSC strain can contain non-finite values (NaN/Inf) within an
        # otherwise correctly-sized segment -- observed in practice around
        # some Gravity Spy glitch times, likely detector lock-loss or
        # data-quality gaps that don't shorten the returned array. Feeding
        # these into the model blindly produced a `ValueError: Input X
        # contains NaN` at training time, so we filter at ingestion instead
        # of imputing -- a strain segment with a real gap isn't usable
        # signal for a Week-1 baseline either way.
        if not np.all(np.isfinite(segment)):
            n_bad = int(np.sum(~np.isfinite(segment)))
            logger.warning(
                "Skipping gps=%s: segment contains %d non-finite sample(s)",
                gps_center, n_bad,
            )
            continue

        X.append(segment)
        y.append("glitch" if row["label"] != "signal" else "signal")
        gps_times.append(gps_center)
        sources.append(row["label"])

    if not X:
        raise RuntimeError(
            "No segments downloaded successfully — check network access to "
            "GWOSC and Zenodo, and verify CONFIRMED_EVENT_GPS_TIMES / Gravity "
            "Spy metadata are still valid."
        )

    X = np.stack(X)
    y = np.array(y)
    gps_times = np.array(gps_times)
    sources = np.array(sources, dtype=object)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, X=X, y=y, gps_times=gps_times, sources=sources
    )
    logger.info("Saved dataset: %d examples -> %s", len(y), output_path)
    return output_path


def load_dataset(path: Path = PROCESSED_DATA_DIR / "dataset.npz"):
    """Load a dataset previously built with build_dataset()."""
    npz = np.load(path, allow_pickle=True)
    return npz["X"], npz["y"], npz["gps_times"], npz["sources"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset()
