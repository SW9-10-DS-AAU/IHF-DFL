"""Tests for analysis/transform.py — normalize_run, normalize_runs, merge_runs."""
import pandas as pd
import pytest

from analysis.loader import RunData
from analysis.transform import normalize_run, normalize_runs, merge_runs, MERGE_META_KEYS
from analysis_helpers import make_users, make_global


# ---------------------------------------------------------------------------
# Helpers — build RunData with specific column values
# ---------------------------------------------------------------------------

def _run_with_global(global_rows, experiment_id="exp-A"):
    return RunData(
        experiment_id=experiment_id,
        metadata={},
        setup={},
        rounds_global=pd.DataFrame(global_rows),
        rounds_users=pd.DataFrame(),
        votes=pd.DataFrame(),
        receipts=pd.DataFrame(),
        contributions=pd.DataFrame(),
        warnings=pd.DataFrame(),
    )


def _run_with_users(user_rows, experiment_id="exp-A"):
    return RunData(
        experiment_id=experiment_id,
        metadata={},
        setup={},
        rounds_global=pd.DataFrame(),
        rounds_users=pd.DataFrame(user_rows),
        votes=pd.DataFrame(),
        receipts=pd.DataFrame(),
        contributions=pd.DataFrame(),
        warnings=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# normalize_run — global table unit conversions
# ---------------------------------------------------------------------------

def test_normalize_reward_pool_wei_to_eth():
    run = _run_with_global([{"round": 1, "reward_pool": int(2e18), "punishment_pool": 0}])
    out = normalize_run(run)
    assert out.rounds_global["reward_pool"].iloc[0] == pytest.approx(2.0)


def test_normalize_punishment_pool_wei_to_eth():
    run = _run_with_global([{"round": 1, "reward_pool": 0, "punishment_pool": int(5e18)}])
    out = normalize_run(run)
    assert out.rounds_global["punishment_pool"].iloc[0] == pytest.approx(5.0)


def test_normalize_objective_global_accuracy_column():
    """'objective_global_accuracy' is a ratio 0..1 and should be scaled to % (×100)."""
    run = _run_with_global([
        {"round": 1, "objective_global_accuracy": 0.92, "reward_pool": 0, "punishment_pool": 0},
    ])
    out = normalize_run(run)
    assert out.rounds_global["objective_global_accuracy"].iloc[0] == pytest.approx(92.0)


# ---------------------------------------------------------------------------
# normalize_run — users table unit conversions
# ---------------------------------------------------------------------------

def test_normalize_grs_wei_to_eth():
    run = _run_with_users([{"round": 1, "user_id": 0, "grs": int(1e18)}])
    out = normalize_run(run)
    assert out.rounds_users["grs"].iloc[0] == pytest.approx(1.0)


def test_normalize_reward_delta_wei_to_eth():
    run = _run_with_users([{"round": 1, "user_id": 0, "reward_delta": int(5e17)}])
    out = normalize_run(run)
    assert out.rounds_users["reward_delta"].iloc[0] == pytest.approx(0.5)


def test_normalize_subjective_personal_accuracy_column():
    """'subjective_personal_accuracy' is a ratio 0..1 and should be scaled to % (×100)."""
    run = _run_with_users([{"round": 1, "user_id": 0, "subjective_personal_accuracy": 0.85}])
    out = normalize_run(run)
    assert out.rounds_users["subjective_personal_accuracy"].iloc[0] == pytest.approx(85.0)


def test_normalize_subjective_global_accuracy_column():
    """'subjective_global_accuracy' is a ratio 0..1 and should be scaled to % (×100)."""
    run = _run_with_users([{"round": 1, "user_id": 0, "subjective_global_accuracy": 0.77}])
    out = normalize_run(run)
    assert out.rounds_users["subjective_global_accuracy"].iloc[0] == pytest.approx(77.0)


# ---------------------------------------------------------------------------
# normalize_run — immutability
# ---------------------------------------------------------------------------

def test_normalize_does_not_mutate_input(minimal_run):
    original_grs = minimal_run.rounds_users["grs"].copy()
    normalize_run(minimal_run)
    pd.testing.assert_series_equal(minimal_run.rounds_users["grs"], original_grs)


def test_normalize_preserves_experiment_id(minimal_run):
    out = normalize_run(minimal_run)
    assert out.experiment_id == minimal_run.experiment_id


# ---------------------------------------------------------------------------
# normalize_runs
# ---------------------------------------------------------------------------

def test_normalize_runs_returns_list_same_length(minimal_run):
    result = normalize_runs([minimal_run, minimal_run])
    assert len(result) == 2


def test_normalize_runs_all_items_are_rundata(minimal_run):
    result = normalize_runs([minimal_run])
    assert all(isinstance(r, RunData) for r in result)


# ---------------------------------------------------------------------------
# merge_runs
# ---------------------------------------------------------------------------

def test_merge_runs_metadata_has_one_row_per_run(minimal_run):
    run_b = RunData(
        experiment_id="exp-B", metadata={}, setup={},
        rounds_global=pd.DataFrame(), rounds_users=pd.DataFrame(),
        votes=pd.DataFrame(), receipts=pd.DataFrame(),
        contributions=pd.DataFrame(), warnings=pd.DataFrame(),
    )
    result = merge_runs([minimal_run, run_b])
    assert len(result["metadata"]) == 2


def test_merge_runs_metadata_contains_experiment_ids(minimal_run):
    run_b = RunData(
        experiment_id="exp-B", metadata={}, setup={},
        rounds_global=pd.DataFrame(), rounds_users=pd.DataFrame(),
        votes=pd.DataFrame(), receipts=pd.DataFrame(),
        contributions=pd.DataFrame(), warnings=pd.DataFrame(),
    )
    result = merge_runs([minimal_run, run_b])
    ids = set(result["metadata"]["experiment_id"])
    assert "test-exp" in ids
    assert "exp-B" in ids


def test_merge_runs_users_contains_both_experiments(two_exp_users, two_exp_global):
    run_a = RunData(
        experiment_id="exp-A", metadata={}, setup={},
        rounds_global=make_global(["exp-A"]),
        rounds_users=make_users(["exp-A"]),
        votes=pd.DataFrame(), receipts=pd.DataFrame(),
        contributions=pd.DataFrame(), warnings=pd.DataFrame(),
    )
    run_b = RunData(
        experiment_id="exp-B", metadata={}, setup={},
        rounds_global=make_global(["exp-B"]),
        rounds_users=make_users(["exp-B"]),
        votes=pd.DataFrame(), receipts=pd.DataFrame(),
        contributions=pd.DataFrame(), warnings=pd.DataFrame(),
    )
    result = merge_runs([run_a, run_b])
    ids = set(result["users"]["experiment_id"])
    assert ids == {"exp-A", "exp-B"}


def test_merge_runs_metadata_keys_preserved(minimal_run):
    result = merge_runs([minimal_run])
    meta = result["metadata"]
    assert "dataset" in meta.columns
    assert "aggregation_rule" in meta.columns
    assert "contribution_score_strategy" in meta.columns


def test_merge_runs_missing_metadata_key_is_none(minimal_run):
    # minimal_run.metadata doesn't have freerider_noise_scale
    result = merge_runs([minimal_run])
    assert pd.isna(result["metadata"]["freerider_noise_scale"].iloc[0])


def test_merge_runs_empty_tables_produce_empty_dataframes():
    run = RunData(
        experiment_id="empty", metadata={}, setup={},
        rounds_global=pd.DataFrame(), rounds_users=pd.DataFrame(),
        votes=pd.DataFrame(), receipts=pd.DataFrame(),
        contributions=pd.DataFrame(), warnings=pd.DataFrame(),
    )
    result = merge_runs([run])
    assert result["global"].empty
    assert result["users"].empty
