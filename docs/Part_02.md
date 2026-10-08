# Part 2: Preparing the forecasting task

## 1. A real forecast question

The following values are copied from the source CSV; only three columns are shown. The date format is month/day/year.

| DateTime | Temperature | Zone 1 Power Consumption |
| --- | ---: | ---: |
| 1/7/2017 23:50 | 13.89 | 29401.51899 |
| 1/8/2017 0:00 | 14.07 | 28332.1519 |
| 1/8/2017 0:30 | 14.34 | 26296.70886 |

At midnight on January 8, the current temperature is 14.07 and current zone-1 demand is 28332.1519. The answer we later score against is zone-1 demand at 00:30: 26296.70886. The future temperature of 14.34 is unavailable at midnight and is excluded from the inputs.

## 2. Align the answer with the question

Exact code in `src/gridwatch/features.py`:

```python
y = frame[ZONES].shift(-horizon)
```

`ZONES` names the three demand columns. `horizon` defaults to 3, corresponding to three ten-minute steps. The negative shift pairs each current row with the demand three rows later. `y` is the answer table, not an input table.

## 3. Add observations from the past

Exact code:

```python
for zone in ZONES:
    for lag in (1, 3, 6, 18, 144, 1008):
        X[f"{zone}_lag_{lag}"] = frame[zone].shift(lag)
```

The six lags represent 10 minutes, 30 minutes, 1 hour, 3 hours, 1 day and 1 week. For the midnight example, `zone_1_lag_1` is 29401.51899, from the previous 23:50 row. Positive shifts only use earlier measurements.

The first 1,008 rows cannot be forecast examples with a full week's history. They remain in the raw file and supply history for later rows. The final three rows lack observed 30-minute-ahead targets. Three additional origins at each split boundary are excluded to keep their targets in the appropriate split. In total, 51,399 prepared examples remain.

## 4. What the 48 inputs mean

- Eight current measurements: five weather variables and demand in three zones.
- Eighteen past-demand values: six lags for each zone.
- Twelve recent summaries: mean and sample standard deviation over six and eighteen readings, for each zone. These include the current reading and earlier readings only.
- Three previous-day references: the demand at the same clock time as the target, on the previous day. With a 30-minute horizon, this is 141 readings before the origin.
- Seven known calendar features: sine/cosine pairs for target time of day, weekday and day of year, plus a weekend indicator. A future calendar time is known in advance; a future measured temperature is not.

Sine/cosine coding keeps nearby times around midnight close to each other. For example, 23:50 and 00:00 are ten minutes apart even though their raw hour numbers lie at opposite ends of a day. The calendar coding does not introduce future measurements.

## 5. Separate learning, selection and final evaluation

The initial source-row boundaries use 70% train, 15% validation and 15% test. The final sample ratios differ slightly because of required history and boundary exclusions. Exact masks:

```python
masks = {
    "train": timing.target_time.lt(validation_start),
    "validation": timing.origin.ge(validation_start) & timing.target_time.lt(test_start),
    "test": timing.origin.ge(test_start),
}
```

Training examples must have their answers before validation starts. Validation examples begin at its boundary and must have answers before test starts. Test examples begin at the test boundary. Validation is used to choose settings; test is held for final evaluation.

Exact guard:

```python
assert timing.loc[masks["train"], "target_time"].max() < validation_start
```

If a training answer crosses into validation, this assertion stops the program. The two tests also check that target times are exactly thirty minutes later and that changing future measurements cannot change earlier inputs.

## 6. Why this design and what it does not establish

Historical lags and chronological evaluation are standard time-series practices. They are understandable and reproducible; this milestone does not claim that the chosen lags or split fractions are optimal. A week of history sacrifices the first week of forecast examples. Using current readings assumes those measurements are available at forecast time. Later model comparisons must establish any accuracy benefit.

## Oral progress statement

We have prepared a thirty-minute-ahead prediction task for three Tetouan electricity zones. Inputs contain current readings, historical consumption and known calendar information. We use chronological training, validation and test sets and check that future labels do not cross their boundaries. The preparation stage runs successfully; model training and accuracy comparisons are the next stage.
