"""Explain train-defined high-demand labels and validation alert thresholds."""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from .data import ROOT, ZONES, load_source
from .experiment import (
    alert_metrics, classification_candidates, continuous_scores,
    fit_operating_threshold, regression_candidates,
)
from .features import prepare_features


FALSE_ALARM_BUDGET = .05


def contribution_table(model, query, zone, origin):
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    standardized = scaler.transform(query)[0]
    coefficients = classifier.coef_[0]
    contributions = standardized * coefficients
    intercept = float(classifier.intercept_[0])
    linear_score = intercept + float(contributions.sum())
    probability_score = float(expit(linear_score))
    np.testing.assert_allclose(probability_score, continuous_scores(model, query)[0], rtol=1e-10, atol=1e-12)
    table = pd.DataFrame({
        "zone": zone, "origin": origin, "feature": query.columns,
        "input_value": query.iloc[0].to_numpy(), "training_mean": scaler.mean_,
        "training_scale": scaler.scale_, "standardized_input": standardized,
        "coefficient": coefficients, "contribution": contributions,
    })
    return table, {"intercept": intercept, "linear_score": linear_score, "probability_score": probability_score}


def confusion_plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for zone_index, zone in enumerate(ZONES[:2]):
        for strategy_index, strategy in enumerate(["Logistic default 0.5", "Logistic calibrated 5%"]):
            row = metrics[(metrics.zone == zone) & (metrics.strategy == strategy)].iloc[0]
            matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            axis = axes[zone_index, strategy_index]
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()))
            for actual in range(2):
                for predicted in range(2):
                    axis.text(predicted, actual, str(matrix[actual, predicted]), ha="center", va="center",
                              color="white" if matrix[actual, predicted] > matrix.max() / 2 else "black")
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Alert"],
                     yticklabels=["Normal", "High"], xlabel="Prediction", ylabel="Actual",
                     title=f"{zone}: {strategy}")
    fig.suptitle("Validation counts: thresholds calibrated on this same validation set")
    fig.savefig(ROOT / "reports/figures/logistic_confusion.png", dpi=150)
    plt.close(fig)


def show_results():
    labels = pd.read_csv(ROOT / "reports/logistic_label_counts.csv")
    metrics = pd.read_csv(ROOT / "reports/logistic_alert_metrics.csv")
    print("High-demand thresholds and observed class counts:")
    print(labels.to_string(index=False))
    print("\nValidation alert results (same set used to calibrate operating thresholds):")
    print(metrics[["zone", "strategy", "tp", "fp", "fn", "tn", "recall", "precision", "false_positive_rate"]].to_string(index=False))
    print("\nEvidence: reports/logistic_real_examples.csv and reports/logistic_contributions.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-results", action="store_true")
    if parser.parse_args().show_results:
        show_results()
        return
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    features, future_demand, timing, masks, _ = prepare_features(
        load_source(), horizon=config["forecast_horizon_minutes"] // 10)
    train_features = features.loc[masks["train"]]
    train_demand = future_demand.loc[masks["train"]]
    validation_features = features.loc[masks["validation"]]
    validation_demand = future_demand.loc[masks["validation"]]
    high_thresholds = train_demand.quantile(config["high_demand_training_quantile"])
    train_labels = (train_demand >= high_thresholds).astype(int)
    validation_labels = (validation_demand >= high_thresholds).astype(int)
    linear = regression_candidates()["Linear"].fit(train_features, train_demand)
    linear_forecasts = np.maximum(linear.predict(validation_features), 0)

    counts, metrics, examples, contributions, curves, settings = [], [], [], [], [], {}
    for zone_index, zone in enumerate(ZONES):
        print(f"Logistic alerts: fitting {zone}", flush=True)
        truth_train = train_labels[zone].to_numpy()
        truth_validation = validation_labels[zone].to_numpy()
        positive_count = int(truth_validation.sum())
        counts.append({
            "zone": zone, "high_demand_threshold": float(high_thresholds[zone]),
            "train_rows": len(truth_train), "train_high": int(truth_train.sum()),
            "train_normal": int((truth_train == 0).sum()),
            "validation_rows": len(truth_validation), "validation_high": positive_count,
            "validation_normal": int((truth_validation == 0).sum()),
        })
        model = classification_candidates()["Logistic"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_features, truth_train)
        scores = continuous_scores(model, validation_features)
        score_cutoff = fit_operating_threshold(scores, truth_validation, FALSE_ALARM_BUDGET)
        forecast_scores = linear_forecasts[:, zone_index]
        forecast_cutoff = fit_operating_threshold(forecast_scores, truth_validation, FALSE_ALARM_BUDGET)
        logistic_decision = (scores >= score_cutoff).astype(int)
        strategies = {
            "Always normal": (np.zeros(len(scores), dtype=int), None),
            "Logistic default 0.5": ((scores >= .5).astype(int), .5),
            "Logistic calibrated 5%": (logistic_decision, score_cutoff),
            "Linear above demand threshold": ((forecast_scores >= high_thresholds[zone]).astype(int), float(high_thresholds[zone])),
            "Linear score calibrated 5%": ((forecast_scores >= forecast_cutoff).astype(int), forecast_cutoff),
        }
        for strategy, (decision, cutoff) in strategies.items():
            result = alert_metrics(truth_validation, decision)
            if "calibrated" in strategy:
                assert result["false_positive_rate"] <= FALSE_ALARM_BUDGET
            metrics.append({"zone": zone, "strategy": strategy, "decision_threshold": cutoff,
                            "status": "development calibration" if positive_count else "no positive validation events; sensitivity unavailable",
                            **result})
        positions = {0: "first chronological sample", 1: "second chronological sample"}
        for actual, predicted, category in [(1, 1, "first true alert"), (0, 1, "first false alarm"),
                                             (1, 0, "first missed high"), (0, 0, "first true normal")]:
            matches = np.flatnonzero((truth_validation == actual) & (logistic_decision == predicted))
            if len(matches):
                position = int(matches[0])
                positions[position] = "; ".join(filter(None, [positions.get(position), category]))
        default_misses = np.flatnonzero((truth_validation == 1) & (scores < .5))
        if len(default_misses):
            position = int(default_misses[0])
            positions[position] = "; ".join(filter(None, [positions.get(position), "first default-0.5 miss"]))
        for position, category in sorted(positions.items()):
            index = validation_features.index[position]
            examples.append({
                "zone": zone, "example_selection": category,
                "origin": str(timing.loc[index, "origin"]), "target_time": str(timing.loc[index, "target_time"]),
                "source_row_number": int(index + 2),
                "temperature": float(features.loc[index, "temperature"]),
                "humidity": float(features.loc[index, "humidity"]),
                "current_demand": float(features.loc[index, zone]),
                "actual_future_demand": float(validation_demand.iloc[position][zone]),
                "high_demand_threshold": float(high_thresholds[zone]), "actual_label": int(truth_validation[position]),
                "logistic_score": float(scores[position]), "logistic_score_cutoff": score_cutoff,
                "logistic_default_alert": int(scores[position] >= .5),
                "logistic_calibrated_alert": int(logistic_decision[position]),
                "linear_forecast": float(forecast_scores[position]),
                "linear_above_high_threshold": int(forecast_scores[position] >= high_thresholds[zone]),
                "linear_calibrated_alert": int(forecast_scores[position] >= forecast_cutoff),
                "zone_sensitivity_validated": bool(positive_count),
            })
        query = validation_features.iloc[[0]]
        table, score_details = contribution_table(model, query, zone, str(timing.loc[query.index[0], "origin"]))
        contributions.append(table)
        settings[zone] = {"high_demand_threshold": float(high_thresholds[zone]),
                          "logistic_cutoff": score_cutoff, "linear_score_cutoff": forecast_cutoff,
                          "validation_positives": positive_count, "iterations": int(model.named_steps["logisticregression"].n_iter_[0]),
                          "first_example": score_details,
                          "operational_status": "development only" if positive_count else "not validated; retain always-normal placeholder"}
        for budget in [0., .01, .02, .05, .10]:
            for strategy, values in [("Logistic", scores), ("Linear score", forecast_scores)]:
                cutoff = fit_operating_threshold(values, truth_validation, budget)
                curves.append({"zone": zone, "strategy": strategy, "target_fpr": budget, "cutoff": cutoff,
                               **alert_metrics(truth_validation, values >= cutoff)})

    metrics = pd.DataFrame(metrics)
    pd.DataFrame(counts).to_csv(ROOT / "reports/logistic_label_counts.csv", index=False)
    metrics.to_csv(ROOT / "reports/logistic_alert_metrics.csv", index=False)
    pd.DataFrame(examples).to_csv(ROOT / "reports/logistic_real_examples.csv", index=False)
    pd.concat(contributions, ignore_index=True).to_csv(ROOT / "reports/logistic_contributions.csv", index=False)
    pd.DataFrame(curves).to_csv(ROOT / "reports/logistic_threshold_curve.csv", index=False)
    summary = {"stage": "logistic_alert_development", "false_alarm_budget": FALSE_ALARM_BUDGET,
               "label_quantile": config["high_demand_training_quantile"], "zones": settings,
               "features": len(features.columns), "train_rows": len(train_features),
               "validation_rows": len(validation_features), "test_evaluated": False,
               "probability_calibration_verified": False,
               "same_validation_used_for_threshold_and_metrics": True,
               "contribution_reconstruction_verified": True}
    (ROOT / "artifacts/logistic_alert_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    confusion_plot(metrics)
    show_results()


if __name__ == "__main__":
    main()
