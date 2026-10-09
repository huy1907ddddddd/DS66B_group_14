"""Choose Ridge/Lasso alpha on train-only chronological folds, then validate."""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, ZONES, load_source
from .features import prepare_features


ALPHAS = {
    "Ridge": [0.01, 0.1, 1, 10, 100, 1000],
    "Lasso": [0.001, 0.01, 0.1, 1, 10, 100],
}
INITIAL_ALPHA = {"Ridge": 100, "Lasso": 10}


def forecast_mae(actual, predicted):
    """Use the same nonnegative forecasts as the initial benchmark."""
    return float(mean_absolute_error(actual, np.maximum(predicted, 0)))


SCORER = make_scorer(forecast_mae, greater_is_better=False)


def chronological_folds(timing, horizon_steps=3, n_splits=3):
    """The gap must keep training answers before each evaluation origin."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon_steps)
    folds = list(splitter.split(timing))
    for learn, evaluate in folds:
        if timing.iloc[learn].target_time.max() >= timing.iloc[evaluate].origin.min():
            raise ValueError("A fold's training answer crosses its evaluation boundary.")
    return folds


def make_model(name, alpha=None, initial=False):
    if name == "Linear":
        estimator = LinearRegression()
    elif name == "Ridge":
        estimator = Ridge(alpha=alpha)
    elif name == "Lasso":
        # A larger iteration limit permits convergence at weak penalties.
        estimator = Lasso(alpha=alpha, max_iter=3000 if initial else 200000, tol=.001,
                          selection="cyclic" if initial else "random", random_state=42,
                          precompute=not initial)
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(StandardScaler(), estimator)


def search_alpha(name, X_train, y_train, folds, alphas=None, expand=True, refine=True):
    """No outer validation/test measurements are accepted by this function."""
    tables, fitted_winners = [], {}
    initial_grid = ALPHAS[name] if alphas is None else alphas
    if not initial_grid or any(alpha <= 0 for alpha in initial_grid):
        raise ValueError("The alpha grid must contain positive values.")

    def run_grid(values, phase):
        values = sorted(set(float(value) for value in values))
        print(f"{name}: {phase}, testing alpha={values}", flush=True)
        search = GridSearchCV(
            make_model(name),
            param_grid={f"{name.lower()}__alpha": values},
            scoring=SCORER, cv=folds, refit=True, return_train_score=True,
            n_jobs=1, error_score="raise",
        )
        # A nonconverged candidate must not silently participate in selection.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(X_train, y_train)
        results = search.cv_results_
        table = pd.DataFrame({
            "model": name, "phase": phase, "alpha": values,
            "cv_mae": -results["mean_test_score"],
            "cv_std_mae": results["std_test_score"],
            "cv_train_mae": -results["mean_train_score"],
            "mean_fit_seconds": results["mean_fit_time"],
        })
        for fold in range(len(folds)):
            table[f"fold_{fold + 1}_mae"] = -results[f"split{fold}_test_score"]
        tables.append(table)
        chosen = float(search.best_params_[f"{name.lower()}__alpha"])
        fitted_winners[chosen] = search.best_estimator_
        print(f"{name}: {phase} best alpha={chosen:g}, CV MAE={-search.best_score_:.3f}", flush=True)

    def current_results():
        return pd.concat(tables, ignore_index=True).sort_values(["cv_mae", "alpha"])

    run_grid(initial_grid, "coarse")
    if expand:
        for _ in range(2):
            scores = current_results()
            best, low, high = scores.iloc[0].alpha, scores.alpha.min(), scores.alpha.max()
            if best == low:
                run_grid([low / 10], "expand_lower")
            elif best == high:
                run_grid([high * 10], "expand_upper")
            else:
                break
    scores = current_results()
    best = float(scores.iloc[0].alpha)
    ordered = sorted(scores.alpha.unique())
    position = ordered.index(best)
    if refine and 0 < position < len(ordered) - 1:
        neighborhood = np.geomspace(ordered[position - 1], ordered[position + 1], 5)
        extra = [float(a) for a in neighborhood if not np.isclose(a, ordered, rtol=1e-10, atol=0).any()]
        if extra:
            run_grid(extra, "refine")
    scores = current_results()
    best = float(scores.iloc[0].alpha)
    boundary = bool(best in (scores.alpha.min(), scores.alpha.max()))
    return fitted_winners[best], scores.sort_values("alpha"), {"alpha": best, "boundary_winner": boundary}


def comparison_row(label, model, X_train, y_train, X_validation, y_validation, cv_mae, alpha=None):
    prediction = np.maximum(model.predict(X_validation), 0)
    per_zone_rmse = np.sqrt(np.mean((y_validation.to_numpy() - prediction) ** 2, axis=0))
    coefficients = model.steps[-1][1].coef_
    row = {
        "model": label, "alpha": alpha, "cv_mae": cv_mae,
        "train_mae": forecast_mae(y_train, model.predict(X_train)),
        "validation_mae": forecast_mae(y_validation, prediction),
        "validation_rmse": float(per_zone_rmse.mean()),
    }
    for index, zone in enumerate(ZONES):
        row[f"{zone}_zero_coefficients"] = int((coefficients[index] == 0).sum())
    return row, prediction


def plot_search(scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, name in zip(axes, ALPHAS):
        rows = scores[scores.model == name].sort_values("alpha")
        winner = rows.loc[rows.cv_mae.idxmin()]
        ax.plot(rows.alpha, rows.cv_mae, "o-", color="#2563eb", label="Mean of three time folds")
        ax.plot(rows.alpha, rows.cv_train_mae, "--", color="#64748b", label="Mean training MAE")
        ax.scatter([winner.alpha], [winner.cv_mae], color="#d97706", s=80, zorder=5, label="Selected alpha")
        ax.axvline(winner.alpha, color="#d97706", ls=":", alpha=.7)
        ax.set(xscale="log", title=f"{name}: selected alpha = {winner.alpha:g}",
               xlabel="Alpha (logarithmic scale)", ylabel="MAE (source units)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Alpha selection uses chronological folds inside train only")
    fig.savefig(ROOT / "reports/figures/alpha_search.png", dpi=150)
    plt.close(fig)


def write_report(summary, comparison):
    lines = [
        "# Ridge and Lasso: alpha selection\n",
        "## Method\n",
        "Three expanding chronological folds are created inside the original training split. "
        "A three-observation gap prevents thirty-minute-ahead training labels from crossing a fold boundary. "
        "The scaler is fitted inside each fold's pipeline. The scorer uses nonnegative predictions and average-zone MAE.",
        "Alpha is selected by minimum mean fold MAE, with two bounded endpoint expansions and a local logarithmic refinement. "
        "This identifies the best tested alpha under this protocol, not a universal optimum. "
        "Refitted candidates are then compared on the original validation split. The held-out test is not evaluated by this stage.",
        "Lasso keeps tolerance 0.001; its search iteration limit is increased from 3,000 to 200,000 and "
        "uses randomized coordinate updates with seed 42 to permit convergence at weak penalties. "
        "The small feature Gram matrix is precomputed for efficiency; this does not change the objective. "
        "Convergence warnings stop the search rather than silently ranking incomplete fits.\n",
        "## Actual results\n",
        "| Model | Alpha | Train MAE | Time-fold MAE | Validation MAE | Validation RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.to_dict("records"):
        alpha = "—" if pd.isna(row["alpha"]) else f"{row['alpha']:g}"
        lines.append(f"| {row['model']} | {alpha} | {row['train_mae']:.2f} | {row['cv_mae']:.2f} | "
                     f"{row['validation_mae']:.2f} | {row['validation_rmse']:.2f} |")
    lines.extend(["\n## Interpretation\n", f"The lowest validation MAE among these candidates is obtained by **{summary['validation_winner']}**."])
    for name, details in summary["selected"].items():
        lines.append(f"- {name}: selected alpha {details['alpha']:g}; "
                     f"{'winner remains at the explored boundary; smaller/larger penalties remain untested.' if details['boundary_winner'] else 'winner lies inside the explored range.'}")
    lines.extend([
        "- A favorable individual prediction is not evidence of lower average error. Inspect the complete validation comparison.",
        "- Train/fold differences can reflect seasonal change as well as model fit; they do not alone prove overfitting.",
        "- Zero Lasso coefficients do not establish that a weather variable has no real-world effect.",
        "- The reported fold score is used for parameter selection, not an unbiased final test result.",
        "\n![Alpha search](figures/alpha_search.png)",
        "\n## Reproduce and present\n",
        "Run `./run_tuning.ps1` to repeat the search. Run `./run_tuning.ps1 -ShowResults` to display saved results immediately.",
        "Detailed evidence: `alpha_search.csv`, `alpha_folds.csv`, `alpha_comparison.csv`, "
        "`alpha_real_examples.csv`, and `../artifacts/alpha_tuning_summary.json`.",
        "\nReferences: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), "
        "[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).",
    ])
    (ROOT / "reports/Alpha_tuning.md").write_text("\n".join(lines), encoding="utf-8")


def show_results():
    table = pd.read_csv(ROOT / "reports/alpha_comparison.csv")
    print(table[["model", "alpha", "cv_mae", "validation_mae", "validation_rmse"]].to_string(index=False))
    print("\nEvidence: reports/Alpha_tuning.md and reports/figures/alpha_search.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-results", action="store_true", help="Show saved results without fitting models.")
    if parser.parse_args().show_results:
        show_results()
        return
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    frame = load_source()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    X, y, timing, masks, _ = prepare_features(frame, horizon=config["forecast_horizon_minutes"] // 10)
    X_train, y_train, train_timing = X.loc[masks["train"]], y.loc[masks["train"]], timing.loc[masks["train"]]
    X_validation, y_validation = X.loc[masks["validation"]], y.loc[masks["validation"]]
    folds = chronological_folds(train_timing, horizon_steps=config["forecast_horizon_minutes"] // 10)
    fold_rows = []
    for index, (learn, evaluate) in enumerate(folds, 1):
        fold_rows.append({"fold": index, "train_rows": len(learn), "evaluation_rows": len(evaluate),
                          "last_train_target": str(train_timing.iloc[learn].target_time.max()),
                          "first_evaluation_origin": str(train_timing.iloc[evaluate].origin.min())})
    pd.DataFrame(fold_rows).to_csv(ROOT / "reports/alpha_folds.csv", index=False)
    scores, selected, models = [], {}, {}
    for name in ALPHAS:
        models[name], table, selected[name] = search_alpha(name, X_train, y_train, folds)
        scores.append(table)
        pd.concat(scores).to_csv(ROOT / "reports/alpha_search.csv", index=False)
    scores = pd.concat(scores, ignore_index=True)
    linear = make_model("Linear")
    linear_cv = cross_validate(linear, X_train, y_train, cv=folds, scoring=SCORER, error_score="raise")
    linear.fit(X_train, y_train)
    candidates = [("Linear", linear, None, float(-linear_cv["test_score"].mean()))]
    for name in ALPHAS:
        initial = make_model(name, INITIAL_ALPHA[name], initial=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            initial.fit(X_train, y_train)
        initial_cv = float(scores.loc[(scores.model == name) & (scores.alpha == INITIAL_ALPHA[name]), "cv_mae"].iloc[0])
        candidates.append((f"{name} initial", initial, INITIAL_ALPHA[name], initial_cv))
        candidates.append((f"{name} tuned", models[name], selected[name]["alpha"], float(scores.loc[scores.model == name, "cv_mae"].min())))
    comparison, examples = [], []
    for label, model, alpha, cv_mae in candidates:
        row, prediction = comparison_row(label, model, X_train, y_train, X_validation, y_validation, cv_mae, alpha)
        comparison.append(row)
        for position in range(2):
            idx = X_validation.index[position]
            for zone_index, zone in enumerate(ZONES):
                examples.append({"model": label, "zone": zone, "origin": str(timing.loc[idx, "origin"]),
                                 "target_time": str(timing.loc[idx, "target_time"]), "current": float(X.loc[idx, zone]),
                                 "actual": float(y.loc[idx, zone]), "prediction": float(prediction[position, zone_index]),
                                 "absolute_error": float(abs(y.loc[idx, zone] - prediction[position, zone_index]))})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(ROOT / "reports/alpha_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(ROOT / "reports/alpha_real_examples.csv", index=False)
    summary = {"method": "train_only_expanding_time_cv", "folds": fold_rows, "gap_rows": 3,
               "train_rows": len(X_train), "validation_rows": len(X_validation), "test_evaluated": False,
               "selected": selected, "validation_winner": comparison.loc[comparison.validation_mae.idxmin(), "model"],
               "scoring": "mean absolute error across three zones, after clipping negative predictions to zero",
               "search_candidates": {name: int((scores.model == name).sum()) for name in ALPHAS}}
    (ROOT / "artifacts/alpha_tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_search(scores)
    write_report(summary, comparison)
    show_results()


if __name__ == "__main__":
    main()
