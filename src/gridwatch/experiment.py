"""A bounded week-10 benchmark with chronological model selection."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Perceptron, Ridge
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .data import ROOT, WEATHER, ZONES, load_source
from .features import prepare_features


def regression_candidates():
    return {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=100)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=10, max_iter=3000, tol=.001)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=2)),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=5, max_features=.8, n_jobs=2, random_state=42),
    }


def classification_candidates():
    return {
        "Always normal": DummyClassifier(strategy="constant", constant=0),
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1500, C=1)),
        "Perceptron": make_pipeline(StandardScaler(), Perceptron(class_weight="balanced", random_state=42, max_iter=1000)),
        "Gaussian NB": make_pipeline(StandardScaler(), GaussianNB()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=25, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=10, class_weight="balanced", n_jobs=2, random_state=42),
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=.1, class_weight="balanced", max_iter=5000, random_state=42, dual="auto")),
    }


def continuous_scores(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    return np.asarray(model.decision_function(X))


def alert_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    positives, negatives = int(tp + fn), int(tn + fp)
    return {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)) if positives else None,
        "f1": float(f1_score(truth, prediction, zero_division=0)) if positives else None,
        "false_positive_rate": float(fp / negatives) if negatives else None,
        "miss_rate": float(fn / positives) if positives else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "positive_events": positives, "negative_events": negatives,
    }


def fit_operating_threshold(scores, truth, false_alarm_budget=.05):
    negatives = np.asarray(scores)[np.asarray(truth) == 0]
    if not len(negatives):
        raise ValueError("Validation requires negative examples for operating-point selection.")
    # Strictly above the selected negative quantile handles ties conservatively.
    threshold = float(np.nextafter(np.quantile(negatives, 1 - false_alarm_budget, method="higher"), np.inf))
    return threshold


def regression_rows(truth, predictions, thresholds, model_name, split, training_seconds=0.0, prediction_seconds=0.0):
    rows = []
    for index, zone in enumerate(ZONES):
        actual = np.asarray(truth)[:, index]
        predicted = predictions[:, index]
        peak = actual >= thresholds[index]
        rows.append({"split": split, "model": model_name, "zone": zone,
                     "mae": float(mean_absolute_error(actual, predicted)),
                     "rmse": float(mean_squared_error(actual, predicted) ** .5),
                     "r2": float(r2_score(actual, predicted)),
                     "peak_mae": float(mean_absolute_error(actual[peak], predicted[peak])) if peak.any() else None,
                     "peak_samples": int(peak.sum()), "train_seconds": training_seconds,
                     "prediction_seconds": prediction_seconds})
    rows.append({"split": split, "model": model_name, "zone": "average",
                 "mae": float(mean_absolute_error(truth, predictions)),
                 "rmse": float(np.mean(np.sqrt(np.mean((np.asarray(truth) - predictions) ** 2, axis=0)))),
                 "r2": float(r2_score(truth, predictions)), "peak_mae": None,
                 "peak_samples": None, "train_seconds": training_seconds, "prediction_seconds": prediction_seconds})
    return rows


def main():
    started = time.perf_counter()
    frame = load_source()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    X, y, timing, masks, manifest = prepare_features(frame, config["forecast_horizon_minutes"] // 10, save_manifest=True)
    train, validation, test = (masks[key] for key in ("train", "validation", "test"))
    thresholds = y.loc[train].quantile(config["high_demand_training_quantile"]).to_numpy()
    alert_truth = (y.to_numpy() >= thresholds).astype(int)
    rows, fitted, validation_predictions = [], {}, {}
    for name in ("Persistence", "Previous day"):
        cols = ZONES if name == "Persistence" else [f"{zone}_daily_reference" for zone in ZONES]
        predictions = X.loc[validation, cols].to_numpy()
        validation_predictions[name] = predictions
        rows += regression_rows(y.loc[validation], predictions, thresholds, name, "validation")
    for name, model in regression_candidates().items():
        begin = time.perf_counter()
        model.fit(X.loc[train], y.loc[train])
        fit_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        predictions = np.maximum(model.predict(X.loc[validation]), 0)
        predict_seconds = time.perf_counter() - begin
        rows += regression_rows(y.loc[validation], predictions, thresholds, name, "validation", fit_seconds, predict_seconds)
        fitted[name], validation_predictions[name] = model, predictions
        print(f"Regression {name}: validation MAE {mean_absolute_error(y.loc[validation], predictions):.2f}; fit {fit_seconds:.1f}s", flush=True)
    validation_table = pd.DataFrame(rows)
    validation_table.to_csv(ROOT / "reports/regression_validation.csv", index=False)
    selected = validation_table.loc[validation_table.zone == "average"].sort_values(["mae", "prediction_seconds"]).iloc[0].model

    # Select classifiers and operating points exclusively on validation.
    alert_rows, alert_models, operating_points, selected_classifiers = [], {}, {}, {}
    validation_alert_status = {}
    train_array, val_array = train.to_numpy(), validation.to_numpy()
    for zone_index, zone in enumerate(ZONES):
        for name, model in classification_candidates().items():
            truth_train, truth_val = alert_truth[train_array, zone_index], alert_truth[val_array, zone_index]
            if len(np.unique(truth_train)) != 2:
                raise ValueError(f"Insufficient classes in training for {zone}")
            begin = time.perf_counter()
            model.fit(X.loc[train], truth_train)
            fit_seconds = time.perf_counter() - begin
            scores = continuous_scores(model, X.loc[validation])
            cutoff = fit_operating_threshold(scores, truth_val)
            metrics = alert_metrics(truth_val, scores >= cutoff)
            alert_rows.append({"split": "validation", "zone": zone, "model": name,
                               "threshold": cutoff, "train_seconds": fit_seconds, **metrics})
            alert_models[(zone, name)], operating_points[(zone, name)] = model, cutoff
        options = pd.DataFrame([row for row in alert_rows if row["zone"] == zone])
        if not alert_truth[val_array, zone_index].sum():
            # No positive validation events: sensitivity cannot be estimated or ranked.
            winner = options[options.model == "Always normal"].iloc[0]
            validation_alert_status[zone] = "Not validated: no positive validation events; always-normal placeholder."
        else:
            winner = options.sort_values(["recall", "precision", "train_seconds"], ascending=[False, False, True], na_position="last").iloc[0]
            validation_alert_status[zone] = "Selected on validation recall at the fixed false-positive budget."
        selected_classifiers[zone] = winner.model
        recall_text = "not estimable" if pd.isna(winner.recall) else f"{winner.recall:.3f}"
        print(f"Alert {zone}: selected {winner.model}; validation recall {recall_text} at FPR {winner.false_positive_rate:.3f}", flush=True)
    pd.DataFrame(alert_rows).to_csv(ROOT / "reports/alerts_validation.csv", index=False)

    # Ablations use validation only and retain exactly the same sample boundaries.
    ablation_rows = []
    if selected in fitted:
        for label, columns in {
            "Without demand history": [name for name in X.columns if not name.startswith("zone_")],
            "Without weather": [name for name in X.columns if name not in WEATHER],
        }.items():
            model = clone(fitted[selected])
            begin = time.perf_counter()
            model.fit(X.loc[train, columns], y.loc[train])
            predictions = np.maximum(model.predict(X.loc[validation, columns]), 0)
            ablation_rows += regression_rows(y.loc[validation], predictions, thresholds, label, "validation", time.perf_counter() - begin)
    ablation_rows += regression_rows(y.loc[validation], validation_predictions[selected], thresholds, "Full selected model", "validation")
    pd.DataFrame(ablation_rows).to_csv(ROOT / "reports/ablation_validation.csv", index=False)

    # Everything above is frozen before the held-out test is evaluated.
    if selected in fitted:
        selected_model = fitted[selected]
        begin = time.perf_counter()
        test_predictions = np.maximum(selected_model.predict(X.loc[test]), 0)
        predict_seconds = time.perf_counter() - begin
    else:
        selected_model = None
        cols = ZONES if selected == "Persistence" else [f"{zone}_daily_reference" for zone in ZONES]
        test_predictions = X.loc[test, cols].to_numpy()
        predict_seconds = 0.0
    test_rows = regression_rows(y.loc[test], test_predictions, thresholds, selected, "test", prediction_seconds=predict_seconds)
    for name, cols in {"Persistence": ZONES, "Previous day": [f"{zone}_daily_reference" for zone in ZONES]}.items():
        if name != selected:
            test_rows += regression_rows(y.loc[test], X.loc[test, cols].to_numpy(), thresholds, name, "test")
    pd.DataFrame(test_rows).to_csv(ROOT / "reports/regression_test.csv", index=False)

    replay = timing.loc[test].copy().reset_index(drop=True)
    test_alert_rows = []
    alert_bundle = {}
    for zone_index, zone in enumerate(ZONES):
        name = selected_classifiers[zone]
        model, cutoff = alert_models[(zone, name)], operating_points[(zone, name)]
        scores = continuous_scores(model, X.loc[test])
        truth = alert_truth[test.to_numpy(), zone_index]
        direct = scores >= cutoff
        # Also tune the forecast-score comparator to the same validation FPR budget.
        reg_cutoff = fit_operating_threshold(validation_predictions[selected][:, zone_index], alert_truth[val_array, zone_index])
        for strategy, predictions in {
            "Direct classifier (validation 5% FPR budget)": direct,
            "Forecast score (validation 5% FPR budget)": test_predictions[:, zone_index] >= reg_cutoff,
            "Forecast above training high-demand threshold": test_predictions[:, zone_index] >= thresholds[zone_index],
        }.items():
            test_alert_rows.append({"split": "test", "zone": zone, "strategy": strategy,
                                    "classifier": name, **alert_metrics(truth, predictions)})
        replay[f"{zone}_current"] = X.loc[test, zone].to_numpy()
        replay[f"{zone}_forecast"] = test_predictions[:, zone_index]
        replay[f"{zone}_actual"] = y.loc[test, zone].to_numpy()
        replay[f"{zone}_high_threshold"] = thresholds[zone_index]
        replay[f"{zone}_alert_score"] = scores
        replay[f"{zone}_alert_threshold"] = cutoff
        replay[f"{zone}_direct_alert"] = direct.astype(int)
        replay[f"{zone}_actual_high"] = truth
        replay[f"{zone}_persistence"] = X.loc[test, zone].to_numpy()
        replay[f"{zone}_daily_reference"] = X.loc[test, f"{zone}_daily_reference"].to_numpy()
        alert_bundle[zone] = {"name": name, "model": model, "threshold": cutoff,
                              "high_demand_threshold": float(thresholds[zone_index])}
    pd.DataFrame(test_alert_rows).to_csv(ROOT / "reports/alerts_test.csv", index=False)
    replay.to_csv(ROOT / "artifacts/demo_data.csv", index=False)
    replay.to_csv(ROOT / "data/processed/test_predictions.csv", index=False)
    feature_range = pd.DataFrame({"min": X.loc[train].min(), "max": X.loc[train].max()})
    feature_range.to_csv(ROOT / "artifacts/training_feature_ranges.csv")
    joblib.dump({"regressor": selected_model, "regressor_name": selected, "alerts": alert_bundle,
                 "features": list(X.columns), "config": config, "training_feature_ranges": feature_range}, ROOT / "artifacts/models.joblib", compress=3)
    summary = {
        "stage": "week_10_initial_benchmark", "selected_regressor": selected,
        "selected_classifiers": selected_classifiers,
        "validation_alert_status": validation_alert_status,
        "high_demand_thresholds": dict(zip(ZONES, map(float, thresholds))),
        "threshold_selection": "Each alert operating point uses validation negatives only, targeting at most 5% validation false-positive rate. Test FPR may differ.",
        "training_rows": int(train.sum()), "validation_rows": int(validation.sum()), "test_rows": int(test.sum()),
        "features": int(X.shape[1]), "elapsed_seconds": time.perf_counter() - started,
        "test_prediction_milliseconds_per_origin": 1000 * predict_seconds / int(test.sum()),
        "regression_test": test_rows, "alert_test": test_alert_rows,
        "limitations": ["One historical year, three zones, source measurement units not confirmed.",
                        "Zone 3 has no positive validation events at its training threshold; its alert is an unvalidated always-normal placeholder.",
                        "One chronological holdout; seasonal changes can invalidate validation operating points.",
                        "No capacity or outage labels; high demand is a research-defined event.",
                        "Initial small configurations, not an exhaustive hyperparameter search.",
                        "Regime adaptation, neural models, PCA and anomaly fallback remain planned.",
                        "Batch prediction timing is not a single-request latency claim.",
                        "Split must be coordinated with other groups using this dataset."],
    }
    (ROOT / "artifacts/experiment_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["selected_regressor", "selected_classifiers", "training_rows", "validation_rows", "test_rows", "elapsed_seconds"]}, indent=2), flush=True)
    make_result_figures(replay, validation_table, pd.DataFrame(test_rows), pd.DataFrame(test_alert_rows))


def make_result_figures(replay, validation_table, test_table, alert_table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    averages = validation_table[validation_table.zone == "average"].sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh(averages.model, averages.mae, color="#2563eb")
    ax.set(title="Model selection uses validation data only", xlabel="Average zone MAE (source units)")
    for i, value in enumerate(averages.mae):
        ax.text(value, i, f"  {value:,.0f}", va="center", fontsize=9)
    ax.margins(x=.15)
    fig.savefig(ROOT / "reports/figures/validation_comparison.png", dpi=160)
    plt.close(fig)
    window = replay.iloc[:432]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for zone, ax in zip(ZONES, axes):
        ax.plot(pd.to_datetime(window.target_time), window[f"{zone}_actual"], label="Observed later", color="#64748b", lw=1.4)
        ax.plot(pd.to_datetime(window.target_time), window[f"{zone}_forecast"], label="Forecast issued 30 minutes earlier", color="#2563eb", lw=1)
        ax.axhline(window[f"{zone}_high_threshold"].iloc[0], color="#d97706", ls="--", label="Training high-demand threshold")
        ax.set(ylabel=zone.replace("_", " ").title())
    axes[0].set_title("Held-out replay: the first three test days (source units)", pad=35)
    axes[0].legend(fontsize=8, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    axes[-1].set_xlabel("Target timestamp")
    fig.savefig(ROOT / "reports/figures/test_replay.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
