"""Prepare a transparent, chronological alert-policy development laboratory."""

from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .data import ROOT, ZONES, load_source
from .decision_support import chronological_policy_split, threshold_candidates, choose_policy
from .experiment import alert_metrics, classification_candidates, continuous_scores
from .features import prepare_features
from .quality import evaluate_quality


PENALTY_SCENARIOS = {"Equal penalties": 1, "Miss costs 20 points": 20, "Miss costs 200 points": 200}


def prepare_operations():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    X, y, timing, masks, _ = prepare_features(load_source(), config["forecast_horizon_minutes"] // 10)
    training = X.loc[masks["train"]]
    validation = X.loc[masks["validation"]]
    validation_timing = timing.loc[validation.index]
    calibration, assessment, boundary = chronological_policy_split(validation_timing)
    thresholds = y.loc[masks["train"]].quantile(config["high_demand_training_quantile"])
    # Reuse the train-only Linear model; no original test predictions are inspected.
    bundle = joblib.load(ROOT / "artifacts/models.joblib")
    if bundle["regressor_name"] != "Linear":
        raise ValueError("This reviewed extension expects the existing train-only Linear model.")
    linear = bundle["regressor"]
    forecasts = np.maximum(linear.predict(validation), 0)
    candidates, replay_rows, metric_rows, zone_settings = [], [], [], {}
    for index, zone in enumerate(ZONES):
        print(f"Policy laboratory: {zone}", flush=True)
        truth_train = (y.loc[masks["train"], zone] >= thresholds[zone]).astype(int)
        truth = (y.loc[validation.index, zone] >= thresholds[zone]).astype(int)
        model = classification_candidates()["Logistic"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(training, truth_train)
        scores_by_model = {"Linear score": forecasts[:, index],
                           "Logistic": continuous_scores(model, validation)}
        calibration_labels = truth.loc[calibration].to_numpy()
        assessment_labels = truth.loc[assessment].to_numpy()
        supported = len(np.unique(calibration_labels)) == 2
        zone_settings[zone] = {"high_demand_threshold": float(thresholds[zone]),
                               "calibration_high": int(calibration_labels.sum()),
                               "assessment_high": int(assessment_labels.sum()),
                               "policy_supported": bool(supported)}
        for model_name, scores in scores_by_model.items():
            table = threshold_candidates(scores[calibration.to_numpy()], calibration_labels)
            table["zone"], table["model"] = zone, model_name
            candidates.append(table)
            zone_replay = validation_timing.copy()
            zone_replay["zone"], zone_replay["model"] = zone, model_name
            zone_replay["period"] = np.where(calibration, "calibration",
                                            np.where(assessment, "assessment", "purged"))
            zone_replay["source_row_number"] = validation.index + 2
            zone_replay["score"] = scores
            zone_replay["current"] = validation[zone].to_numpy()
            zone_replay["forecast"] = forecasts[:, index]
            zone_replay["actual"] = y.loc[validation.index, zone].to_numpy()
            zone_replay["actual_high"] = truth.to_numpy()
            zone_replay["high_demand_threshold"] = float(thresholds[zone])
            replay_rows.append(zone_replay.reset_index(drop=True))
            if not supported:
                continue
            for scenario, miss_penalty in PENALTY_SCENARIOS.items():
                selected = choose_policy(table, miss_penalty, 1)
                fixed_cutoff = float(thresholds[zone]) if model_name == "Linear score" else .5
                assessment_scores = scores[assessment.to_numpy()]
                strategies = {
                    "Cost-selected threshold": assessment_scores >= selected.threshold,
                    "Fixed demand threshold / logistic 0.5": assessment_scores >= fixed_cutoff,
                    "Always normal": np.zeros(len(assessment_scores), dtype=bool),
                    "Always alert": np.ones(len(assessment_scores), dtype=bool),
                }
                for strategy, alerts in strategies.items():
                    counts = alert_metrics(assessment_labels, alerts)
                    metric_rows.append({"zone": zone, "model": model_name, "scenario": scenario,
                                        "strategy": strategy, "miss_penalty": miss_penalty,
                                        "false_alarm_penalty": 1,
                                        "chosen_threshold": float(selected.threshold),
                                        "calibration_penalty": float(selected.penalty),
                                        "assessment_penalty": miss_penalty * counts["fn"] + counts["fp"],
                                        **counts})
    pd.concat(candidates, ignore_index=True).to_csv(ROOT / "artifacts/policy_candidates.csv", index=False)
    pd.concat(replay_rows, ignore_index=True).to_csv(ROOT / "artifacts/policy_replay.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(ROOT / "reports/policy_assessment.csv", index=False)
    summary = {"stage": "chronological_policy_development", "boundary": str(boundary),
               "calibration_rows": int(calibration.sum()), "assessment_rows": int(assessment.sum()),
               "purged_rows": int((~calibration & ~assessment).sum()),
               "calibration_last_target": str(validation_timing.loc[calibration, "target_time"].max()),
               "assessment_first_origin": str(validation_timing.loc[assessment, "origin"].min()),
               "test_re_evaluated": False, "previously_inspected_validation": True,
               "units": "hypothetical penalty points per 10-minute sample; not money",
               "zones": zone_settings}
    (ROOT / "artifacts/operations_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return X, y, timing, masks, calibration, assessment, linear


def show_results():
    table = pd.read_csv(ROOT / "reports/policy_assessment.csv")
    selected = table[table.strategy == "Cost-selected threshold"]
    print(selected[["zone", "model", "scenario", "fp", "fn", "assessment_penalty"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-results", action="store_true")
    if parser.parse_args().show_results:
        show_results()
        return
    X, y, timing, masks, _, assessment, linear = prepare_operations()
    # Evaluate the basic gate only on the later development-assessment period.
    # Keep the original test and original benchmark files unchanged.
    assessment_index = assessment.index[assessment]
    bounds, quality_metrics, quality_replay = evaluate_quality(
        linear, X.loc[masks["train"]], X.loc[assessment_index],
        y.loc[assessment_index], timing.loc[assessment_index])
    bounds.to_csv(ROOT / "artifacts/quality_bounds.csv")
    quality_metrics.to_csv(ROOT / "reports/quality_assessment.csv", index=False)
    quality_replay.to_csv(ROOT / "artifacts/quality_replay.csv", index=False)
    show_results()
    print("\nQuality checks (MAE is calculated on available predictions):")
    print(quality_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
