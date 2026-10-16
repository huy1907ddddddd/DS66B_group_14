# Tetouan GridWatch

**Statistical Learning — DS66B, Group 14**

Trần Quang Huy · Bùi Ngọc Thuấn · Nguyễn Hoàng Thành

## Start with the notebook

[Open in Google Colab](https://colab.research.google.com/github/huy1907ddddddd/DS66B_group_14/blob/main/notebooks/GridWatch_Colab.ipynb)

Open `notebooks/GridWatch_Colab.ipynb` and choose **Runtime → Run all**. The notebook reads one public CSV and uses ordinary pandas and scikit-learn code. It needs no Windows paths, Drive mount, project-module imports or saved-result files. Each result is calculated by the visible cells.

## What it does

1. Read and check the Tetouan data; inspect its training-period statistics.
2. Create 48 current, historical and calendar inputs; predict the three zones 30 minutes ahead.
3. Split by time, keeping future answers before the next period's boundary.
4. Compare two simple baselines and six regression models using validation MAE.
5. Show one real prediction and search small Alpha/K grids using three chronological folds inside train.
6. Check feature groups, demonstrate a Zone 1 high-demand alert, compare hypothetical alert penalties and show a checked past-reading fallback for a synthetic bad input.

The normal run includes Alpha/K selection. Set `RUN_TREE_SEARCH = True` to also search tree/forest structure. Set `RUN_TEST_REPLAY = True` only to replay the previously inspected test period. These switches are visible in the relevant cells.

## Scope and interpretation

- The source samples every 10 minutes; the project's chosen forecast horizon is 30 minutes. A future demand reading is not total energy consumed over 30 minutes. Measurement units are retained as source units.
- The chronological split remains provisional until agreed with peer groups. The historical test has already been inspected and is not a new untouched final test.
- The small grids select the best tried settings on their folds, not a universal optimum. The initial-model comparison and tuned-model comparison are separate tables.
- High demand is defined by the training 90th percentile, not physical grid capacity. Error penalties are assumed points, not proven savings. The faulty packet is a classroom simulation.
- Chapters 11–15 and the final English report/slides remain unfinished. Record each member's actual contribution.

The other `src/`, `app/`, `reports/` and older notebook files retain the earlier, larger implementation and its historical outputs. They are not dependencies of this simplified notebook, and their guides/scores do not automatically describe the new small-grid searches.

## Sources

- [UCI Tetouan dataset](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), Salam & El Hibaoui, DOI 10.24432/C5B034, CC BY 4.0.
- [Assignment topics and requirements](https://docs.google.com/spreadsheets/d/17AhxqqU-uAUvH9RTIqbgO1w13_obfjTln45ukoH7SmM/edit).
- [Alsalem (2025), Scientific Reports](https://www.nature.com/articles/s41598-025-91123-8), related Tetouan forecasting work. Scores require the same horizon, data split and metrics before comparison.
