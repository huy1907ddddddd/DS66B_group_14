# Decision-aware alerts and basic input-quality resilience

This extension develops two operational questions: when does a high-demand alert justify preparation, and how should the forecast service respond to a faulty input packet? It adds a decision laboratory and a basic quality gate to the existing 30-minute point-forecast benchmark. The implementation is a transparent application of established techniques, not a claim of algorithmic novelty or demonstrated monetary savings.

## Experimental protocol

Keep the original train/validation/test contract. Reuse the train-only Linear regressor and fit zone-specific balanced Logistic classifiers on training labels defined by the training 90th percentile. Split the previously inspected validation period into 3,926 earlier calibration origins and 3,930 later assessment origins. Purge three boundary origins so the last calibration target (10 October 2017, 01:50) precedes the first assessment origin (02:00). The original test is not re-evaluated. The assessment window supports development diagnostics; prior inspection of validation prevents treating it as a new pristine test set.

Enumerate every distinct alert decision threshold on calibration, handling tied scores together. Select the threshold minimizing `c_miss * FN + c_false_alarm * FP`. Tie-breaking favors fewer false alarms, then fewer misses. Report its fixed-threshold performance on the later assessment window alongside fixed-threshold, always-normal and always-alert comparators. Penalties are hypothetical points per 10-minute sample, not money; adjacent positive samples may belong to the same peak episode. Logistic scores have not been verified as calibrated probabilities. Zone 3 has no positive calibration examples, so no useful policy is presented as validated.

## Decision results

For Zone 1 Logistic, equal penalties select a threshold around 0.949622, producing 15 misses and seven false alarms on later assessment. A 20:1 miss-to-false-alarm penalty selects about 0.623382, producing one miss and 44 false alarms, for 64 points. Under the same 20:1 penalty, the fixed 0.5 comparator produces one miss and 62 false alarms, for 82 points. These observations support the proposed decision interface within this development period, not universal improvement. Zone 2 at 200:1 demonstrates a counterexample to unconditional classifier superiority: Linear-score gives 344 later penalty points, versus 517 for Logistic.

## Basic quality protocol and results

Check finite required features and sensor envelopes fitted on training only. Envelopes extend training min/max by a fixed 25% of their range and respect elementary constraints such as nonnegative demand and 0–100 humidity. This margin is an engineering assumption, not a tuned parameter or a forecast confidence interval. Only acceptable packets enter the Linear model. A blocked packet falls back by zone to an acceptable current measurement, then an acceptable 10-minute-old measurement. Without either, the service explicitly returns no forecast. The fault demonstration withholds direct classifier alerts from faulty packets.

Apply four separate feature-packet scenarios to the same 3,930 later assessment origins: original observations, missing temperature, a tenfold Zone 1 current measurement, and missing current/recent backup measurements. Faults are synthetic edits after feature construction, not complete raw-sensor outages with recomputed history. Report forecast coverage and MAE on available outputs together.

For Zone 1, clean-packet MAE changes from 418.98 without the gate to 419.49 with it; two original wind-speed observations breach the envelope and trigger fallback. These breaches are not confirmed sensor faults. Missing-temperature packets produce no unguarded forecasts; fallback restores 100% coverage with MAE 1,069.96. The tenfold-current scenario changes MAE from 278,785.15 to 1,394.05 with 100% coverage. Without current or recent backups, both approaches return no forecasts; MAE is undefined, not zero.

## Practical scope and course coverage

An operator could use lead time to review agreed flexible loads or available storage, provided the deployment has those resources. The dataset does not supply physical grid capacity, outage labels, tariffs, storage characteristics or intervention outcomes. This prototype does not optimize dispatch or demonstrate avoided outages or realized savings. Implemented methods use regression/classification, descriptive statistics, confusion counts and ordinary input-validation logic. Advanced anomaly detection remains reserved for chapter 15; neural, clustering and dimensionality-reduction extensions are still planned.

Code: `src/gridwatch/decision_support.py`, `quality.py`, `operations.py`; interface: `app/operations_panel.py`. Reproduce with `run_operations.ps1`; evidence: `reports/policy_assessment.csv`, `quality_assessment.csv`, and row-level replays in `artifacts/`. The Vietnamese presentation guide is `reports/Operations_Guide_vi.md`. The original LaTeX and slides require these results to be integrated before final submission.
