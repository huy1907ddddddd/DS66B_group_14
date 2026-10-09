"""Choose an alert threshold using explicit, hypothetical error penalties."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_policy_split(timing):
    """Calibrate on earlier targets; assess on later origins, with a horizon gap."""
    if len(timing) < 10 or not timing.origin.is_monotonic_increasing:
        raise ValueError("Policy development needs ordered chronological observations.")
    boundary = timing.origin.iloc[len(timing) // 2]
    calibration = timing.target_time.lt(boundary)
    assessment = timing.origin.ge(boundary)
    if not calibration.any() or not assessment.any():
        raise ValueError("Both policy periods need observations.")
    assert timing.loc[calibration, "target_time"].max() < timing.loc[assessment, "origin"].min()
    return calibration, assessment, boundary


def threshold_candidates(scores, truth):
    """Count errors at every distinct threshold, treating tied scores together."""
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth)
    if scores.ndim != 1 or truth.shape != scores.shape or not len(scores):
        raise ValueError("Scores and binary labels must be equally sized vectors.")
    if not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("Finite scores and binary labels are required.")
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_truth = scores[order], truth[order].astype(int)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    removed_positives = np.cumsum(sorted_truth)[ends]
    removed_negatives = ends + 1 - removed_positives
    total_positive, total_negative = int(truth.sum()), int((truth == 0).sum())
    # Minimum score includes every sample; just above each group removes that group.
    thresholds = np.r_[sorted_scores[0], np.nextafter(sorted_scores[ends], np.inf)]
    if not np.isfinite(thresholds).all():
        raise ValueError("Scores are too large to represent a finite upper threshold.")
    fn = np.r_[0, removed_positives]
    tn = np.r_[0, removed_negatives]
    return pd.DataFrame({"threshold": thresholds, "tp": total_positive - fn,
                         "fp": total_negative - tn, "fn": fn, "tn": tn})


def choose_policy(candidates, miss_penalty, false_alarm_penalty):
    """Minimize c_miss * FN + c_false_alarm * FP on calibration data only."""
    if not np.isfinite([miss_penalty, false_alarm_penalty]).all() or min(miss_penalty, false_alarm_penalty) <= 0:
        raise ValueError("Both hypothetical penalties must be positive and finite.")
    ranked = candidates.copy()
    ranked["penalty"] = miss_penalty * ranked.fn + false_alarm_penalty * ranked.fp
    # Equal penalties favor fewer false alarms, then fewer misses; deterministic.
    return ranked.sort_values(["penalty", "fp", "fn", "threshold"],
                              kind="stable").iloc[0]


def penalty_total(truth, alerts, miss_penalty, false_alarm_penalty):
    truth, alerts = np.asarray(truth), np.asarray(alerts, dtype=bool)
    misses = int(((truth == 1) & ~alerts).sum())
    false_alarms = int(((truth == 0) & alerts).sum())
    return miss_penalty * misses + false_alarm_penalty * false_alarms
