# Tetouan GridWatch

**DS66B — Group 14:** Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

**Progress snapshot:** data preparation; forecasting baselines; chronological tuning of Ridge/Lasso, KNN, Decision Tree and Random Forest; detailed Logistic alert analysis; decision-aware threshold development; basic data-quality checks and fallbacks; local historical demo. The original LaTeX/slides are draft milestone artifacts and do not yet incorporate every later experiment. Chapter 11–15 models remain planned.

**Week 10 milestone:** forecast the demand of three distribution zones 30 minutes ahead and evaluate high-demand alerts. A decision laboratory now selects thresholds using explicit hypothetical error penalties, and a basic input-quality gate demonstrates recent-measurement fallbacks. Regime adaptation and advanced anomaly detection remain planned.

## Start here

Open PowerShell in this project folder. On the current computer, the dependencies and experiment outputs are already prepared:

```powershell
.\run_demo.ps1
```

Open http://localhost:8501. Select a zone, day and forecast origin. Future observations are hidden until explicitly revealed. This is a **historical test replay**, not a live connection to the electricity grid.

The **Decision lab** and **Data quality** tabs use separate development-assessment windows within validation. Run `.\run_operations.ps1` to regenerate their evidence; `.\run_operations.ps1 -ShowResults` prints saved policy results. Earlier calibration has 3,926 rows, later assessment 3,930 rows, and three boundary rows are purged. This validation data was previously inspected, so the later window is development evidence, not a new pristine test set.

The decision objective is `miss_penalty * FN + false_alarm_penalty * FP`, in hypothetical points per 10-minute sample. All distinct thresholds are considered on calibration only; later assessment is used for reporting. Penalties do not represent actual monetary savings. Zone 3 lacks high-demand calibration examples, so no operational threshold is offered.

The quality gate checks finite required inputs and training-only sensor envelopes (minimum/maximum extended by a fixed 25% of their range, with basic physical bounds). On a blocked packet it uses the current acceptable zone measurement, otherwise the acceptable measurement 10 minutes earlier; if both fail it returns no forecast. Report both MAE on available predictions and coverage. Faults are synthetic edits to prepared feature packets; they are not a raw-sensor outage simulation or labeled real faults. See `reports/Operations_Guide_vi.md` for real examples and presentation coaching.

On a new Windows computer with Python 3.12 installed:

```powershell
.\setup.ps1
.\run_experiment.ps1
.\run_demo.ps1
```

The local ASCII-path Python junction avoids a Windows numerical-library path issue. It points to this project's own virtual environment; it does not contain a second copy of the code. The setup uses pinned dependency versions. An Internet connection is needed to install dependencies and download the public UCI dataset.

Portable commands, from the project root:

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements.txt
# Set PYTHONPATH to the project's src directory.
python -m unittest discover -s tests -v
python -m gridwatch.data
python -m gridwatch.experiment
python -m streamlit run app/dashboard.py --server.address 127.0.0.1
```

## What is actually evaluated

- UCI Tetouan CSV: 52,416 observations, every 10 minutes, 1 January–30 December 2017. Five weather fields and three zone-demand targets.
- Inputs: measurements available at the forecast origin, lagged demand, trailing statistics, and the known target-time calendar. No future observed weather is used.
- Forecast target: the zone measurement exactly 30 minutes later. Original measurement units are retained as **source units** because the UCI variable table does not confirm the units. A point forecast is not energy integrated over 30 minutes.
- Training, validation and test follow time order. Target timestamps cannot cross development boundaries. The split remains provisional until coordinated with peer groups.
- Forecast baselines: last available measurement and the previous day's same target slot. Classroom candidates: linear regression, Ridge, Lasso, KNN, decision tree and random forest.
- Alert label: demand at or above each zone's training 90th percentile. This research definition is not physical capacity, an outage label or a sensor-fault label.
- Classifiers: always-normal baseline, logistic regression, perceptron, Gaussian Naive Bayes, KNN, decision tree, random forest and linear SVM. Each score threshold targets at most 5% false positives on validation negatives. Select by validation recall, then precision, then fit time. Later test false-positive rates can change.
- No positive validation events means alert sensitivity cannot be ranked. Such a zone retains an explicitly unvalidated always-normal placeholder. An undefined recall is reported as unavailable, not as perfect performance.

All selected models and thresholds are frozen before the held-out test is evaluated. Validation ablations remove weather or all demand history, using exactly the same samples. This is a bounded initial benchmark, not exhaustive tuning or evidence that a contribution is novel.

## Project layout

```text
src/gridwatch/          Data audit, temporal features and experiment pipeline
  decision_support.py  Threshold counts and cost-sensitive calibration
  quality.py           Basic input checks, fallback and synthetic fault assessment
  operations.py        Reproducible development experiments for both new panels
tests/                 Temporal-information and alert-metric contracts
app/dashboard.py       Historical forecast replay and evidence tables
app/operations_panel.py Decision and quality demonstrations
notebooks/             English, executed week-10 research walkthrough
data/raw/              Original public CSV (downloaded, excluded from Git)
data/processed/        Generated predictions (excluded from Git)
artifacts/             Split manifest, verified summary, fitted models and replay
reports/               Tables, figures, report source and future-module contracts
config.json            Forecast horizon and provisional split configuration
requirements.txt       Exact installed dependency versions
```

`run_experiment.ps1` regenerates audited results and figures. Results depend on the pinned software stack and fixed random seed; minor timing differences are expected. A batch timing is not a single-request latency claim.

## Research direction and course coverage

Chapters 1–5 support problem definition, Python, dataset inspection, descriptive statistics and the evaluation process. Chapter 6 supplies regression, regularization and classification; chapter 7 KNN; chapter 8 Gaussian Naive Bayes; chapter 9 decision trees and random forests; chapter 10 classification SVM. Methods are applied where they fit the target, rather than forcing a classification-only method into demand regression.

After chapter 10: small neural benchmarks (11), regime clustering (12), the missing chapter 13 after confirmation, feature selection/PCA (14), and advanced anomaly detection (15). The implemented basic gate is elementary validation logic, not a later-chapter trained anomaly detector. See `reports/Future_modules.md`.

Two hypotheses guide the final project: direct alerts might reduce misses at comparable false alarms; adaptive regimes and a quality gate might improve forecasting under changing patterns or explicitly synthetic input faults. They must be tested against suitable comparators. Successful implementation alone does not establish novelty.

## Sources and submission notes

- [UCI dataset and license](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) — public data; keep its attribution and check the license when redistributing.
- [Alsalem, 2025, Scientific Reports](https://www.nature.com/articles/s41598-025-91123-8) — existing clustering/ML work on this dataset; clustering alone is not an original contribution. Published scores are not directly comparable without matching splits and horizons.

Official notebook, slides and report are in English. Vietnamese files are internal planning and presentation coaching. Group identifier, member contribution table, final shared split and GitHub sharing details require actual group information. Do not invent names, IDs, contributions or final-course results.
