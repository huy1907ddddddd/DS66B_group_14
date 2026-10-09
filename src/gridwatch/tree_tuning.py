"""Select bounded decision-tree complexity using chronological training folds."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data import ROOT, ZONES, load_source
from .features import prepare_features
from .tuning import SCORER, chronological_folds, forecast_mae


DEPTHS = [4, 8, 12, 16]
LEAF_SIZES = [10, 20, 50]


def make_tree(depth=12, leaf_size=20):
    return DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=leaf_size,
        random_state=42,
    )


def select_tree(X_train, y_train, folds):
    """Choose depth and leaf size on training-only chronological folds."""
    search = GridSearchCV(
        make_tree(),
        param_grid={"max_depth": DEPTHS, "min_samples_leaf": LEAF_SIZES},
        scoring=SCORER,
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    print("Decision Tree: testing 12 depth/leaf-size configurations", flush=True)
    search.fit(X_train, y_train)
    results = search.cv_results_
    table = pd.DataFrame({
        "max_depth": results["param_max_depth"].astype(int),
        "min_samples_leaf": results["param_min_samples_leaf"].astype(int),
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


def path_rows(model, X, y, timing, query, query_index):
    """Return the exact tests and leaf average for a displayed forecast."""
    leaf_id = int(model.apply(query)[0])
    rows = []
    for node in model.decision_path(query).indices:
        if node == leaf_id:
            continue
        feature = X.columns[model.tree_.feature[node]]
        value = float(query.iloc[0][feature])
        threshold = float(model.tree_.threshold[node])
        goes_left = np.float32(value) <= threshold
        rows.append({
            "node": int(node), "feature": feature, "input_value": value,
            "threshold": threshold, "rule": "<= threshold" if goes_left else "> threshold",
            "next_direction": "left" if goes_left else "right",
        })
    train_leaf_ids = model.apply(X)
    leaf_answers = y.iloc[train_leaf_ids == leaf_id]
    prediction = model.predict(query)[0]
    np.testing.assert_allclose(leaf_answers.mean().to_numpy(), prediction, rtol=1e-10, atol=1e-6)
    return pd.DataFrame(rows), leaf_id, leaf_answers, prediction


def plot_search(table, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = table.pivot(index="min_samples_leaf", columns="max_depth", values="cv_mae")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for depth in DEPTHS:
        rows = table[table.max_depth == depth].sort_values("min_samples_leaf")
        ax.plot(rows.min_samples_leaf, rows.cv_mae, "o-", label=f"max depth = {depth}")
    selected_row = table[(table.max_depth == selected["max_depth"]) &
                         (table.min_samples_leaf == selected["min_samples_leaf"])].iloc[0]
    ax.scatter([selected["min_samples_leaf"]], [selected_row.cv_mae], color="#d97706", s=95,
               zorder=5, label="Selected configuration")
    ax.set(xlabel="Minimum training samples per leaf", ylabel="Mean MAE of three time folds (source units)",
           title="Decision Tree: choose complexity by chronological CV MAE")
    ax.set_xticks(LEAF_SIZES)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / "reports/figures/tree_complexity_search.png", dpi=150)
    plt.close(fig)


def show_results():
    comparison = pd.read_csv(ROOT / "reports/tree_comparison.csv")
    print(comparison.to_string(index=False))
    examples = pd.read_csv(ROOT / "reports/tree_real_examples.csv")
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
    train_timing = timing.loc[masks["train"]]
    folds = chronological_folds(train_timing, horizon_steps=config["forecast_horizon_minutes"] // 10)

    selected_model, search_table, selected = select_tree(X_train, y_train, folds)
    search_table.to_csv(ROOT / "reports/tree_complexity_search.csv", index=False)

    initial_model = make_tree().fit(X_train, y_train)
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    candidates = [("Linear", linear, None, None),
                  ("Tree initial", initial_model, 12, 20),
                  ("Tree selected", selected_model, selected["max_depth"], selected["min_samples_leaf"])]
    comparison, examples = [], []
    for name, model, depth, leaf_size in candidates:
        prediction = np.maximum(model.predict(X_validation), 0)
        comparison.append({
            "model": name, "max_depth": depth, "min_samples_leaf": leaf_size,
            "validation_mae": forecast_mae(y_validation, prediction),
        })
        for position in range(2):
            index = X_validation.index[position]
            origin_mae = float(np.abs(y_validation.iloc[position].to_numpy() - prediction[position]).mean())
            for zone_index, zone in enumerate(ZONES):
                examples.append({
                    "model": name, "max_depth": depth, "min_samples_leaf": leaf_size, "zone": zone,
                    "origin": str(timing.loc[index, "origin"]), "target_time": str(timing.loc[index, "target_time"]),
                    "current": float(X.loc[index, zone]), "actual": float(y.loc[index, zone]),
                    "prediction": float(prediction[position, zone_index]),
                    "absolute_error": float(abs(y.loc[index, zone] - prediction[position, zone_index])),
                    "one_origin_mean_absolute_error": origin_mae,
                })
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(ROOT / "reports/tree_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(ROOT / "reports/tree_real_examples.csv", index=False)

    query = X_validation.iloc[[0]]
    paths, leaf_id, leaf_answers, prediction = path_rows(selected_model, X_train, y_train, train_timing, query, query.index[0])
    paths.to_csv(ROOT / "reports/tree_selected_path.csv", index=False)
    leaf_answers.assign(source_row_number=leaf_answers.index + 2,
                        origin=train_timing.loc[leaf_answers.index, "origin"].astype(str),
                        target_time=train_timing.loc[leaf_answers.index, "target_time"].astype(str)).to_csv(
                            ROOT / "reports/tree_selected_leaf_answers.csv", index=False)

    summary = {
        "selected": selected,
        "search_candidates": int(len(search_table)),
        "folds": len(folds),
        "gap_rows": config["forecast_horizon_minutes"] // 10,
        "train_rows": len(X_train), "validation_rows": len(X_validation), "test_evaluated": False,
        "selected_cv_mae": float(search_table.iloc[0].cv_mae),
        "selected_tree_depth": int(selected_model.get_depth()),
        "selected_tree_leaves": int(selected_model.get_n_leaves()),
        "displayed_leaf_id": leaf_id, "displayed_leaf_training_samples": int(len(leaf_answers)),
        "leaf_mean_matches_prediction": True,
        "comparison": comparison.where(pd.notna(comparison), None).to_dict("records"),
        "limitations": [
            "Best explored depth and leaf size under this fixed feature set and squared-error split criterion.",
            "A path for one forecast illustrates the tree; it does not explain all forecasts.",
            "The outer validation and the held-out test are not used to select complexity.",
        ],
    }
    for row in summary["comparison"]:
        for key in ("max_depth", "min_samples_leaf"):
            if pd.isna(row[key]):
                row[key] = None
    (ROOT / "artifacts/tree_tuning_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_search(search_table, selected)
    show_results()


if __name__ == "__main__":
    main()
