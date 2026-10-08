# DS66B_group_14

Statistical Learning Midterm Project — Topic 18: Electricity Consumption Prediction.

## Team

- Class: DS66B
- Group: 14
- Members: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành

## Reviewed progress: Part 2 — Data preparation

Tetouan GridWatch prepares a 30-minute-ahead forecasting task for three electricity distribution zones. This milestone loads the source CSV, builds 48 input features, aligns future targets, and creates chronological train/validation/test sets. It does not train a forecasting model or claim forecast accuracy or a proven novel contribution.

The complete CSV contains 52,416 observations. After constructing sufficient history, matching the forecast horizon and excluding targets crossing split boundaries:

| Set | Samples | First forecast origin | Last forecast origin |
| --- | ---: | --- | --- |
| Train | 35,680 | 2017-01-08 00:00 | 2017-09-12 18:30 |
| Validation | 7,859 | 2017-09-12 19:10 | 2017-11-06 08:50 |
| Test | 7,860 | 2017-11-06 09:30 | 2017-12-30 23:20 |

## Run this milestone

Tested with Python 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python -m unittest discover -s tests -p "test_temporal_contract.py" -v
```

The preparation command writes `artifacts/split_manifest.json` and `artifacts/real_example.csv`. The tests check exact 30-minute target alignment, split boundaries, and that changing future measurements cannot change earlier input features.

## Files and their purpose

| Location | Purpose |
| --- | --- |
| `data/raw/` | Unmodified source CSV and attribution |
| `src/gridwatch/data.py` | Read the CSV and convert column names and data types |
| `src/gridwatch/features.py` | Construct current/past inputs, future targets and chronological splits |
| `config.json` | Forecast horizon and split fractions |
| `prepare_data.py` | Run this stage and print actual examples |
| `artifacts/` | Reproducible split details and a real input/target example |
| `tests/` | Checks of the temporal forecasting contract |
| `docs/Part_02.md` | Plain-English explanation with exact code and source values |

## Dataset and assumptions

[UCI Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), donated by Abdulwahed Salami and Abdellatif Daoudi. Retain the original source values and timestamps. Demand is reported in source units; measurement units and a timezone offset are not established by this milestone.

This setup assumes current demand and weather readings are available at each forecast origin. Targets 30 minutes later are labels used for training and scoring, never forecast inputs. Evaluation will advance through the test period, so already observed test readings can become history for subsequent predictions.

## Planned next stages

After team review: classroom forecasting baselines through chapter 10, high-demand alert evaluation, and an English presentation/report. Later course extensions may investigate regime adaptation and a data-quality gate with fallback. Those extensions remain planned and require evidence before any claim of improvement.
