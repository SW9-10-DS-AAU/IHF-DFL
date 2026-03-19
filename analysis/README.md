# Analysis Pipeline

Replaces the old CSV-only output with a structured **pickle-based logging and analysis pipeline**.
Every experiment now saves a `.pkl` file alongside its `.csv`. The `.pkl` file contains clean pandas DataFrames that are ready to load, transform, aggregate, and plot — no string-parsing required.

---

## Pipeline Overview

```
Experiment run
  └─ ExperimentLogger.log_*()        ← called inside fl_challenge.py
       └─ logger.save(path.pkl)      ← called in experiments.py
            └─ loader.load_runs()
                 └─ transform.normalize_runs() + merge_runs()
                      └─ aggregations.agg_*()
                           └─ plots.plot_*()
```

---

## Output File Location

Each experiment produces **two files** in the same timestamped folder:

```
experiment/data/experimentData/26-02-26--10_57_12/
  mnist-accuracy_only-3-1.0-3-1.0-True-False.csv   ← existing CSV (unchanged)
  mnist-accuracy_only-3-1.0-3-1.0-True-False.pkl   ← new pickle
```

---

## Pickle Structure

```python
{
  "experiment_id": "mnist-accuracy_only-3-1.0-3-1.0-True-False",
  "metadata": {
      "dataset": "mnist",
      "contribution_score_strategy": "accuracy_only",
      "minimum_rounds": 5,
      "number_of_good_contributors": 4,
      # ... all ExperimentConfiguration fields + timestamp
  },
  "tables": {
      "global":   DataFrame,   # one row per round
      "users":    DataFrame,   # one row per (round, user_id)
      "votes":    DataFrame,   # one row per (round, giver_id, receiver_id)
      "receipts": DataFrame,   # one row per blockchain transaction
  }
}
```

### `global` table columns

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | str | run identifier |
| `round` | int | round number (0 = baseline before training) |
| `round_time` | float | wall-clock seconds for this round |
| `global_accuracy` | float | global model accuracy (raw ratio, e.g. 0.92) |
| `global_loss` | float | global model loss |
| `reward_pool` | int | remaining reward balance in wei |
| `punishment_pool` | int | total punishments this round in wei |
| `merged` | bool | whether any participants were merged |

### `users` table columns

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | str | run identifier |
| `round` | int | round number |
| `user_id` | int | participant index |
| `state` | str | `"active"` or `"disqualified"` |
| `behavior` | str | current attitude: `good`, `bad`, `freerider`, `inactive` |
| `role` | str | intended final role (`futureAttitude`) |
| `accuracy` | float | this user's local model accuracy |
| `loss` | float | this user's local model loss |
| `grs` | int | global reputation score in wei |
| `prev_global_accuracy` | float | global accuracy before this round |
| `prev_global_loss` | float | global loss before this round |
| `contribution_score` | int | on-chain contribution score in wei-scaled units |
| `is_negative_contrib` | bool | whether contribution score was negative |
| `round_reputation_assigned` | int | round reputation after voting |
| `reward_delta` | int | reward received this round in wei |

### `votes` table columns

| Column | Type | Description |
|--------|------|-------------|
| `round` | int | round number |
| `giver_id` / `receiver_id` | int | participant indices |
| `vote_score` | int | vote value: `1` positive, `0` neutral, `-1` negative |
| `giver_address` / `receiver_address` | str | on-chain addresses |

### `receipts` table columns

| Column | Type | Description |
|--------|------|-------------|
| `round` | int | round when the tx was submitted |
| `tx_type` | str | `register`, `slot`, `weights`, `feedback`, `contrib`, `close`, `exit` |
| `tx_hash` | str | transaction hash |
| `gas_used` | int | gas consumed |

---

## Usage

### Minimal example (notebook or script)

```python
import sys
sys.path.insert(0, ".")   # run from repo root

from pathlib import Path
from analysis import load_runs_recursive, normalize_runs, merge_runs, aggregations as agg, plots

# 1. Load
runs   = load_runs_recursive(Path("experiment/data/experimentData/"))

# 2. Normalize units  (wei → ETH, ratio → %)
runs   = normalize_runs(runs)

# 3. Merge into flat DataFrames tagged with metadata
merged = merge_runs(runs)

# 4. Aggregate
agg_global   = agg.agg_global_accuracy_by_round(merged["global"])
agg_strategy = agg.agg_accuracy_by_strategy(merged["global"])
agg_final    = agg.agg_final_round_accuracy_by_strategy(merged["global"])
agg_grs      = agg.agg_grs_by_behavior(merged["users"])

# 5. Plot
fig1 = plots.plot_accuracy_loss_over_rounds(agg_global)
fig2 = plots.plot_strategy_comparison_lines(agg_strategy)
fig3 = plots.plot_strategy_comparison_boxplot(agg_final)
fig4 = plots.plot_grs_by_behavior(agg_grs)

# 6. Save or show
plots.save_figure(fig2, Path("figures/strategy_comparison.png"))
import matplotlib.pyplot as plt
plt.show()
```

### Load a single run

```python
from analysis import load_run, normalize_run
from pathlib import Path

run = load_run(Path("experiment/data/experimentData/26-02-26--10_57_12/mnist-accuracy_only-3-1.0-3-1.0-True-False.pkl"))
run = normalize_run(run)

print(run.experiment_id)
print(run.metadata)
print(run.rounds_global.head())
print(run.rounds_users.head())
```

### Inspect raw DataFrames without normalizing

```python
from analysis import load_run
run = load_run(path)

# GRS is in wei here — divide by 1e18 manually, or call normalize_run() first
print(run.rounds_users[["round", "user_id", "behavior", "grs"]].head(10))
```

### Filter merged data by strategy

```python
merged = merge_runs(normalize_runs(load_runs_recursive(data_dir)))

dotproduct_global = merged["global"][
    merged["global"]["contribution_score_strategy"] == "dotproduct"
]
```

---

## Module Reference

### `analysis.loader`

| Function | Description |
|----------|-------------|
| `load_run(path)` | Load one `.pkl` file → `RunData` |
| `load_runs(directory)` | Load all `.pkl` files in a flat directory |
| `load_runs_recursive(root)` | Walk all subdirectories and load every `.pkl` |

`RunData` is a dataclass with fields: `experiment_id`, `metadata`, `rounds_global`, `rounds_users`, `votes`, `receipts`.

---

### `analysis.transform`

| Function | Description |
|----------|-------------|
| `normalize_run(run)` | Convert units in one `RunData`; returns a new `RunData` |
| `normalize_runs(runs)` | Apply `normalize_run` to a list |
| `merge_runs(runs)` | Concatenate all runs; tag rows with config metadata; returns `{"global", "users", "votes", "receipts"}` |

**Unit conversions applied by `normalize_run`:**

| Column | Conversion |
|--------|-----------|
| `reward_pool`, `punishment_pool` | ÷ 1e18 (wei → ETH) |
| `grs`, `reward_delta`, `contribution_score` | ÷ 1e18 |
| `global_accuracy` | × 100 (ratio → %) |
| `accuracy` (users table) | × 100 (ratio → %) |
| `is_baseline` | added: `True` where `round == 0` |

---

### `analysis.aggregations`

| Function | Input | Output |
|----------|-------|--------|
| `agg_global_accuracy_by_round(merged_global)` | `merged["global"]` | `round, accuracy_mean, accuracy_std, loss_mean, loss_std` |
| `agg_accuracy_by_strategy(merged_global)` | `merged["global"]` | `contribution_score_strategy, round, accuracy_mean, accuracy_std` |
| `agg_final_round_accuracy_by_strategy(merged_global)` | `merged["global"]` | `contribution_score_strategy, experiment_id, final_accuracy` |
| `agg_grs_by_behavior(merged_users)` | `merged["users"]` | `behavior, round, grs_mean, grs_std` (excludes round 0) |
| `agg_contribution_score_by_behavior(merged_users)` | `merged["users"]` | `behavior, round, score_mean, score_std` |

---

### `analysis.plots`

All functions return a `matplotlib.figure.Figure`. Call `plt.show()` or `plots.save_figure(fig, path)` to display/save.

| Function | Chart type | Uses |
|----------|-----------|------|
| `plot_accuracy_loss_over_rounds(agg_global)` | Dual-axis line (accuracy left, loss right) with ±1σ shading | `agg_global_accuracy_by_round` output |
| `plot_strategy_comparison_lines(agg_by_strategy)` | One line per strategy with ±1σ bands | `agg_accuracy_by_strategy` output |
| `plot_strategy_comparison_boxplot(agg_final)` | Box plot, one box per strategy | `agg_final_round_accuracy_by_strategy` output |
| `plot_grs_by_behavior(agg_grs)` | One line per behavior with ±1σ bands | `agg_grs_by_behavior` output |
| `save_figure(fig, path, dpi=150)` | Saves to PNG/PDF/SVG (inferred from extension) | — |

---

## What Changed in the Existing Code

### `experiment/experiments.py`
- Imports `ExperimentLogger` from the `analysis` package.
- Before each experiment: instantiates `ExperimentLogger(experiment_id, metadata)` where `metadata` contains all config fields plus `dataset` and `timestamp`.
- After each experiment: calls `logger.save(path.with_suffix(".pkl"))` to write the pickle file next to the CSV.

### `experiment/experiment_runner.py`
- `run_experiment()` accepts a new optional `logger=None` parameter and forwards it to `FLChallenge`.

### `src/openfl/contracts/fl_challenge.py`
- `FLChallenge.__init__()` accepts `logger=None` and stores it as `self.logger`.
- `log_receipt()` now also calls `self.logger.log_receipt(...)` to capture every blockchain transaction.
- `simulate()` calls:
  - `logger.log_global_round(round=0, ...)` once for the baseline (before training starts).
  - Per round after settlement: `logger.log_vote(...)` for every feedback matrix entry, `logger.log_user_round(...)` for every active and disqualified participant, and `logger.log_global_round(...)` for the round summary.
- **The CSV / `AsyncWriter` path is completely unchanged.** Logger runs in parallel and adds no breaking changes.

### `src/openfl/utils/async_writer.py`
- `NullWriter` was missing `writeResult`, `writeComment`, and `finish` methods — calls to these raised `AttributeError` when no writer was provided. All three are now no-ops.

---

## Notes

- **Normalization is not applied automatically on load.** Call `normalize_run()` / `normalize_runs()` explicitly before aggregating or plotting, otherwise GRS and contribution scores will be in raw wei (1e18 scale).
- **Round 0 is the baseline** (registration only, no training). It is logged with `global_accuracy=0` and `merged=False`. Use `is_baseline` column or filter `round > 0` to exclude it from training-curve plots.
- **`contribution_score` may be `None`** for disqualified users or inactive users who were not in the merge set.
- **`votes` table is empty for round 0** — feedback exchange only happens in rounds 1+.
