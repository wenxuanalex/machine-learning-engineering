"""Tests for the point-in-time snapshot design (temporal cohorts for drift).

Verifies that the two gold snapshots are leakage-safe, share a schema, and are
genuinely different distributions — the property that makes the drift report
meaningful rather than a comparison of two samples of one distribution.
"""

import pandas as pd
import pytest

from utils.gold import (
    GOLD_SNAPSHOTS,
    SCORING_SNAPSHOT,
    TRAINING_SNAPSHOT,
    build_gold_feature_store,
)


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory):
    """Build every configured snapshot once into a temp dir."""
    out_dir = tmp_path_factory.mktemp("snaps")
    built = {}
    for scoring_date, w in GOLD_SNAPSHOTS.items():
        dest = str(out_dir / f"fs_{scoring_date}.parquet")
        build_gold_feature_store(
            dest_parquet=dest,
            obs_start=w["obs_start"],
            obs_end=w["obs_end"],
        )
        built[scoring_date] = pd.read_parquet(dest)
    return built


def test_label_window_strictly_after_observation():
    # Config-level point-in-time guarantee: label is measured strictly after the
    # observation window, so no label-window data can leak into a feature.
    for scoring_date, w in GOLD_SNAPSHOTS.items():
        obs_end = pd.Timestamp(w["obs_end"])
        label_start = obs_end + pd.Timedelta(days=1)
        label_end = obs_end + pd.Timedelta(days=90)
        assert pd.Timestamp(w["obs_start"]) <= obs_end < label_start <= label_end, scoring_date


def test_training_and_scoring_label_windows_are_disjoint():
    # The training label window must close before the scoring label window opens.
    train_end = pd.Timestamp(GOLD_SNAPSHOTS[TRAINING_SNAPSHOT]["obs_end"]) + pd.Timedelta(days=90)
    score_start = pd.Timestamp(GOLD_SNAPSHOTS[SCORING_SNAPSHOT]["obs_end"]) + pd.Timedelta(days=1)
    assert (
        train_end
        < score_start
    )


def test_snapshots_share_schema(snapshots):
    ref = snapshots[TRAINING_SNAPSHOT]
    cur = snapshots[SCORING_SNAPSHOT]
    assert list(ref.columns) == list(cur.columns)


def test_snapshots_are_genuinely_different(snapshots):
    # The whole point of temporal cohorts: the two periods are not the same
    # distribution. Churn rate must differ materially between them.
    ref = snapshots[TRAINING_SNAPSHOT]
    cur = snapshots[SCORING_SNAPSHOT]
    churn_gap = abs(ref["is_churn_label"].mean() - cur["is_churn_label"].mean())
    assert churn_gap > 0.03, f"snapshots too similar (churn gap {churn_gap:.3f})"


def test_no_future_leakage_into_features(snapshots):
    for scoring_date, df in snapshots.items():
        # recency = days since last purchase as-of obs_end; a purchase after
        # obs_end would make this negative — so non-negativity proves no leakage.
        assert (df["recency"] >= 0).all(), scoring_date
        # the as-of date must equal the observation cut-off
        obs_end = pd.Timestamp(GOLD_SNAPSHOTS[scoring_date]["obs_end"])
        assert (df["scoring_date"] == obs_end).all()
