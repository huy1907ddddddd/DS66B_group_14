"""Select a bounded random-forest structure with chronological folds."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, ZONES, load_source
from .features import prepare_features
from .tuning import SCORER, chronological_folds, forecast_mae


DEPTHS = [8, 14]
LEAF_SIZES = [5, 20, 50]
N_TREES = 60


def make_forest(depth=14, leaf_size=5, n_trees=N_TREES):
    return RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=depth,
        min_samples_leaf=leaf_size,
        max_features=.8,
        n_jobs=2,
        random_state=42,
    )


def select_forest(X_train, y_train, folds, n_trees=N_TREES):
    """Select only structure; validation and test remain outside this function."""
    search = GridSearchCV(
        make_forest(n_trees=n_trees),
        param_grid={"max_depth": DEPTHS, "min_samples_leaf": LEAF_SIZES},
        scoring=SCORER,
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    print(f"Random Forest: testing 6 structures with {n_trees} trees each", flush=True)
    search.fit(X_train, y_train)
    results = search.cv_results_
    table = pd.DataFrame({
        "max_depth": results["param_max_depth"].astype(int),
        "min_samples_leaf": results["param_min_samples_leaf"].astype(int),
        "n_trees": n_trees,
        "cv_mae": -results["mean_test_score"],
        "cv_std_mae": results["std_test_score"],
        "mean_fit_seconds": results["mean_fit_time"],
        "mean_score_seconds": results["mean_score_time"],
    })
    for index in range(len(folds)):
        table[f"fold_{index + 1}_mae"] = -results[f"split{index}_test_score"]
    table = table.sort_values(["cv_mae", "max_depth", "min_samples_leaf"]).reset_index(drop=True)
    selected = {
        "max_depth": int(search.best_params_["max_depth"]),
        "min_samples_leaf": int(search.best_params_["min_samples_leaf"]),
    }
    return search.best_estimator_, table, selected


def forest_sample_explanation(model, X, y, timing, query, query_index):
    """Show individual tree predictions and verify their average is the forest forecast."""
    per_tree = np.vstack([tree.predict(query.to_numpy())[0] for tree in model.estimators_])
    forest_prediction = model.predict(query)[0]
    np.testing.assert_allclose(per_tree.mean(axis=0), forest_prediction, rtol=1e-10, atol=1e-6)
    rows = pd.DataFrame(per_tree, columns=[f"{zone}_prediction" for zone in ZONES])
    rows.insert(0, "tree_number", np.arange(1, len(rows) + 1))
    rows["origin"] = str(timing.loc[query_index, "origin"])
    rows["target_time"] = str(timing.loc[query_index, "target_time"])
    summary = pd.DataFrame({
        "zone": ZONES,
        "forest_prediction": forest_prediction,
        "mean_of_tree_predictions": per_tree.mean(axis=0),
        "tree_prediction_min": per_tree.min(axis=0),
        "tree_prediction_max": per_tree.max(axis=0),
        "tree_prediction_std": per_tree.std(axis=0),
    })
    importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return rows, summary, importance


def plot_search(table, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for depth in DEPTHS:
        rows = table[table.max_depth == depth].sort_values("min_samples_leaf")
        ax.plot(rows.min_samples_leaf, rows.cv_mae, "o-", label=f"max depth = {depth}")
    selected_row = table[(table.max_depth == selected["max_depth"]) &
                         (table.min_samples_leaf == selected["min_samples_leaf"])].iloc[0]
    ax.scatter([selected["min_samples_leaf"]], [selected_row.cv_mae], color="#d97706", s=95,
               zorder=5, label="Selected structure")
    ax.set(xlabel="Minimum training samples per leaf", ylabel="Mean MAE of three time folds (source units)",
           title="Random Forest: choose structure by chronological CV MAE")
    ax.set_xticks(LEAF_SIZES)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / "reports/figures/forest_structure_search.png", dpi=150)
    plt.close(fig)


def show_results():
    comparison = pd.read_csv(ROOT / "reports/forest_comparison.csv")
    print(comparison.to_string(index=False))
    examples = pd.read_csv(ROOT / "reports/forest_real_examples.csv")
    first_origin = examples.origin.iloc[0]
    print("\nActual first validation example, zone 1:")
    print(examples.loc[(examples.origin == first_origin) & (examples.zone == "zone_1"),
                       ["model", "origin", "target_time", "actual", "prediction", "absolute_error"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-results", action="store_true")
    parser.add_argument("--reuse-search", action="store_true",
                        help="Reuse a completed structure-search table, then fit and report the winner.")
    arguments = parser.parse_args()
    if arguments.show_results:
        show_results()
        return

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    X, y, timing, masks, _ = prepare_features(load_source(), horizon=config["forecast_horizon_minutes"] // 10)
    X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
    X_validation, y_validation = X.loc[masks["validation"]], y.loc[masks["validation"]]
    train_timing = timing.loc[masks["train"]]
    folds = chronological_folds(train_timing, horizon_steps=config["forecast_horizon_minutes"] // 10)

    if arguments.reuse_search:
        search_table = pd.read_csv(ROOT / "reports/forest_structure_search.csv")
        winner = search_table.sort_values(["cv_mae", "max_depth", "min_samples_leaf"]).iloc[0]
        selected = {"max_depth": int(winner.max_depth), "min_samples_leaf": int(winner.min_samples_leaf)}
    else:
        _, search_table, selected = select_forest(X_train, y_train, folds)
        search_table.to_csv(ROOT / "reports/forest_structure_search.csv", index=False)
    selected_model = make_forest(depth=selected["max_depth"], leaf_size=selected["min_samples_leaf"]).fit(X_train, y_train)
    initial_model = make_forest().fit(X_train, y_train)
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    candidates = [("Linear", linear, None, None),
                  ("Forest initial", initial_model, 14, 5),
                  ("Forest selected", selected_model, selected["max_depth"], selected["min_samples_leaf"])]
    comparison, examples = [], []
    for name, model, depth, leaf_size in candidates:
        prediction = np.maximum(model.predict(X_validation), 0)
        comparison.append({
            "model": name, "max_depth": depth, "min_samples_leaf": leaf_size,
            "n_trees": N_TREES if name != "Linear" else None,
            "validation_mae": forecast_mae(y_validation, prediction),
        })
        for position in range(2):
            index = X_validation.index[position]
            origin_mae = float(np.abs(y_validation.iloc[position].to_numpy() - prediction[position]).mean())
            for zone_index, zone in enumerate(ZONES):
                examples.append({
                    "model": name, "max_depth": depth, "min_samples_leaf": leaf_size,
                    "n_trees": N_TREES if name != "Linear" else None,
                    "zone": zone, "origin": str(timing.loc[index, "origin"]),
                    "target_time": str(timing.loc[index, "target_time"]), "current": float(X.loc[index, zone]),
                    "actual": float(y.loc[index, zone]), "prediction": float(prediction[position, zone_index]),
                    "absolute_error": float(abs(y.loc[index, zone] - prediction[position, zone_index])),
                    "one_origin_mean_absolute_error": origin_mae,
                })
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(ROOT / "reports/forest_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(ROOT / "reports/forest_real_examples.csv", index=False)

    per_tree, explanation, importance = forest_sample_explanation(
        selected_model, X_train, y_train, timing, X_validation.iloc[[0]], X_validation.index[0])
    per_tree.to_csv(ROOT / "reports/forest_selected_tree_predictions.csv", index=False)
    explanation.to_csv(ROOT / "reports/forest_selected_prediction_summary.csv", index=False)
    importance.to_csv(ROOT / "reports/forest_feature_importance.csv", index=False)

    summary = {
        "selected": selected,
        "n_trees": N_TREES,
        "max_features": .8,
        "search_candidates": int(len(search_table)),
        "folds": len(folds), "gap_rows": config["forecast_horizon_minutes"] // 10,
        "train_rows": len(X_train), "validation_rows": len(X_validation), "test_evaluated": False,
        "selected_cv_mae": float(search_table.iloc[0].cv_mae),
        "tree_average_matches_forest_prediction": True,
        "comparison": comparison.where(pd.notna(comparison), None).to_dict("records"),
        "limitations": [
            "Best explored depth and leaf size under fixed tree count, max features and feature set.",
            "Feature importance measures impurity reduction, not causal importance.",
            "The outer validation and held-out test are not used to select structure.",
        ],
    }
    for row in summary["comparison"]:
        for key in ("max_depth", "min_samples_leaf"):
            if pd.isna(row[key]):
                row[key] = None
    (ROOT / "artifacts/forest_tuning_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_search(search_table, selected)
    show_results()


if __name__ == "__main__":
    main()
