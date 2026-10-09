# Planned extensions after chapter 10

These interfaces are design contracts, not trained models. No result is claimed.

A basic rule-based quality gate and recent-measurement fallback are now implemented in `src/gridwatch/quality.py`, with synthetic feature-packet tests in the Data quality demo. The chapter-15 extension below concerns an advanced anomaly detector, not this elementary input-validation logic.

| Module | Input available at forecast origin | Output | Evidence required before integration |
|---|---|---|---|
| Neural forecasting, chapter 11 | Past-only multi-zone sequences and available weather | Three demand forecasts for a fixed horizon | Same split, same baseline, accuracy and latency comparison |
| Regime adaptation, chapter 12 | Past-window summaries and known calendar | Regime label or weights over specialists | Compare a global model to an adaptive model; do not use a full future day's profile |
| Chapter 13 | To be defined after course material is available | To be defined | Course topic must be confirmed |
| Dimensionality reduction, chapter 14 | Training-only feature matrix and an unchanged prediction sample | Selected features or PCA projection | Error, speed, dimensionality and interpretability comparison |
| Anomaly gate and fallback, chapter 15 | Current available inputs and already-observed forecast residuals | Flag, reason and documented fallback | Clean-input false alarms; explicitly synthetic faults; no claim of real sensor-fault labels |

Every extension uses the existing temporal contract and split manifest. Fitted transformations belong in the training pipeline. Test targets and future measured weather cannot enter inputs or operating-point selection. A new experiment version must state what changed before evaluating new test evidence.
