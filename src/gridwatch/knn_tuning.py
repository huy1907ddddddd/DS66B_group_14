"""Select K on chronological training folds and explain a real forecast."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, ZONES, load_source
from .features import prepare_features
from .tuning import SCORER, chronological_folds, forecast_mae


K_VALUES = [1, 2, 3, 5, 7, 11, 15, 21, 31, 51, 101]


def make_knn(k=15):
    return make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=2),
    )


def select_k(X_train, y_train, folds):
    """Only training data enter parameter selection; score on future folds."""
    smallest_train = min(len(learn) for learn, _ in folds)
    tables, winners = [], {}

    def run_grid(values, phase):
        values = sorted({int(k) for k in values if 1 <= k <= smallest_train})
        print(f"KNN: {phase}, testing K={values}", flush=True)
        search = GridSearchCV(
            make_knn(), param_grid={"kneighborsregressor__n_neighbors": values},
            scoring=SCORER, cv=folds, refit=True, n_jobs=1, error_score="raise",
        )
        search.fit(X_train, y_train)
        results = search.cv_results_
        table = pd.DataFrame({
            "k": values, "phase": phase,
            "cv_mae": -results["mean_test_score"],
            "cv_std_mae": results["std_test_score"],
            "mean_fit_seconds": results["mean_fit_time"],
            "mean_score_seconds": results["mean_score_time"],
        })
        for index in range(len(folds)):
            table[f"fold_{index + 1}_mae"] = -results[f"split{index}_test_score"]
        tables.append(table)
        best_k = int(search.best_params_["kneighborsregressor__n_neighbors"])
        winners[best_k] = search.best_estimator_
        print(f"KNN: {phase} best K={best_k}, CV MAE={-search.best_score_:.2f}", flush=True)

    def scores():
        return pd.concat(tables, ignore_index=True).sort_values(["cv_mae", "k"])

    run_grid(K_VALUES, "coarse")
    for _ in range(2):
        table = scores()
        best, largest = int(table.iloc[0].k), int(table.k.max())
        if best == largest and largest < smallest_train:
            run_grid([min(2 * largest + 1, smallest_train)], "expand_upper")
        else:
            break
    table = scores()
    best = int(table.iloc[0].k)
    ordered = sorted(table.k.tolist())
    position = ordered.index(best)
    if 0 < position < len(ordered) - 1:
        nearby = range(ordered[position - 1] + 1, ordered[position + 1])
        extra = sorted(set(nearby) - set(ordered))
        if extra:
            run_grid(extra, "refine")
    table = scores()
    best = int(table.iloc[0].k)
    return winners[best], table.sort_values("k"), best


def explain_neighbors(model, X_train, y_train, timing, query, query_index):
    """Record the exact neighbors and check their weighted prediction."""
    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["kneighborsregressor"]
    distances, positions = estimator.kneighbors(scaler.transform(query))
    distances, positions = distances[0], positions[0]
    # Match sklearn's distance weighting, including exact matches.
    if (distances == 0).any():
        weights = (distances == 0).astype(float)
    else:
        weights = 1 / distances
    weights = weights / weights.sum()
    neighbors = X_train.iloc[positions]
    table = pd.DataFrame({
        "rank": np.arange(1, len(positions) + 1),
        "source_row_number": neighbors.index.to_numpy() + 2,
        "neighbor_origin": timing.loc[neighbors.index, "origin"].astype(str).to_numpy(),
        "neighbor_target_time": timing.loc[neighbors.index, "target_time"].astype(str).to_numpy(),
        "distance_after_scaling": distances, "normalized_weight": weights,
        "temperature": neighbors.temperature.to_numpy(),
        "humidity": neighbors.humidity.to_numpy(),
        "query_origin": str(timing.loc[query_index, "origin"]),
        "query_target_time": str(timing.loc[query_index, "target_time"]),
    })
    for zone in ZONES:
        answers = y_train.iloc[positions][zone].to_numpy()
        table[f"{zone}_current"] = neighbors[zone].to_numpy()
        table[f"{zone}_answer"] = answers
        table[f"{zone}_weighted_contribution"] = weights * answers
    manual = np.array([table[f"{zone}_weighted_contribution"].sum() for zone in ZONES])
    np.testing.assert_allclose(manual, model.predict(query)[0], rtol=1e-10, atol=1e-6)
    return table


def plot_search(scores, best_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(scores.k, scores.cv_mae, "o-", color="#2563eb", label="Mean MAE of three time folds")
    selected = scores.loc[scores.k == best_k].iloc[0]
    ax.scatter([best_k], [selected.cv_mae], color="#d97706", s=85, zorder=5,
               label=f"Selected K = {best_k}")
    ax.axvline(best_k, color="#d97706", ls=":", alpha=.7)
    ax.set(xscale="log", xlabel="K (logarithmic scale)", ylabel="MAE (source units)",
           title="KNN: choose K by chronological CV MAE, not by a visual elbow")
    ticks = sorted(set(K_VALUES + [best_k]) & set(scores.k))
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=8)
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(ROOT / "reports/figures/knn_k_search.png", dpi=150)
    plt.close(fig)


def show_results():
    comparisons = pd.read_csv(ROOT / "reports/knn_comparison.csv")
    print(comparisons.to_string(index=False))
    examples = pd.read_csv(ROOT / "reports/knn_real_examples.csv")
    first_origin = examples.origin.iloc[0]
    print("\nActual first validation example, zone 1:")
    print(examples.loc[(examples.origin == first_origin) & (examples.zone == "zone_1"),
                       ["model", "origin", "target_time", "actual", "prediction", "absolute_error"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-results", action="store_true")
    if parser.parse_args().show_results:
        show_results()
        return
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    X, y, timing, masks, _ = prepare_features(load_source(), horizon=config["forecast_horizon_minutes"] // 10)
    X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
    X_validation, y_validation = X.loc[masks["validation"]], y.loc[masks["validation"]]
    folds = chronological_folds(timing.loc[masks["train"]], horizon_steps=config["forecast_horizon_minutes"] // 10)
    model, scores, best_k = select_k(X_train, y_train, folds)
    scores.to_csv(ROOT / "reports/knn_k_search.csv", index=False)
    print("Comparing fixed candidates on outer validation; no test evaluation.", flush=True)
    initial = make_knn(15).fit(X_train, y_train)
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    comparisons, examples = [], []
    for label, candidate, k in [("Linear", linear, None), ("KNN initial", initial, 15), ("KNN selected", model, best_k)]:
        prediction = np.maximum(candidate.predict(X_validation), 0)
        comparisons.append({"model": label, "k": k, "validation_mae": forecast_mae(y_validation, prediction)})
        for position in range(2):
            idx = X_validation.index[position]
            origin_mae = float(np.abs(y_validation.iloc[position].to_numpy() - prediction[position]).mean())
            for zone_index, zone in enumerate(ZONES):
                examples.append({
                    "model": label, "k": k, "zone": zone,
                    "origin": str(timing.loc[idx, "origin"]), "target_time": str(timing.loc[idx, "target_time"]),
                    "temperature": float(X.loc[idx, "temperature"]), "humidity": float(X.loc[idx, "humidity"]),
                    "current": float(X.loc[idx, zone]), "actual": float(y.loc[idx, zone]),
                    "prediction": float(prediction[position, zone_index]),
                    "absolute_error": float(abs(y.loc[idx, zone] - prediction[position, zone_index])),
                    "one_origin_mean_absolute_error": origin_mae,
                })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(ROOT / "reports/knn_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(ROOT / "reports/knn_real_examples.csv", index=False)
    query = X_validation.iloc[[0]]
    for label, candidate in [("initial", initial), ("selected", model)]:
        neighbors = explain_neighbors(candidate, X_train, y_train, timing, query, query.index[0])
        neighbors.to_csv(ROOT / f"reports/knn_neighbors_{label}.csv", index=False)
    summary = {
        "selected_k": best_k, "weights": "distance", "distance": "Euclidean on train-standardized 48 features",
        "tested_k": sorted(scores.k.astype(int).tolist()), "folds": len(folds),
        "gap_rows": config["forecast_horizon_minutes"] // 10,
        "train_rows": len(X_train), "validation_rows": len(X_validation), "test_evaluated": False,
        "selected_cv_mae": float(scores.loc[scores.k == best_k, "cv_mae"].iloc[0]),
        "comparison": comparisons.where(pd.notna(comparisons), None).to_dict("records"),
        "manual_neighbor_prediction_verified": True,
        "limitations": ["Best explored K under fixed distance weights and features; not a universal optimum.",
                        "Curve is a validation-error curve, not an unsupervised inertia elbow.",
                        "Two real examples do not replace aggregate validation metrics."],
    }
    # Convert pandas' missing float to JSON null for Linear's absent K.
    for row in summary["comparison"]:
        if pd.isna(row["k"]):
            row["k"] = None
    (ROOT / "artifacts/knn_tuning_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_search(scores, best_k)
    show_results()


if __name__ == "__main__":
    main()
