"""Tests for analysis/aggregations.py.

Focus areas:
  1. Correct output schemas (column names).
  2. Numerical correctness (hand-verified expected values).
  3. Two-stage aggregation — per-experiment isolation before cross-experiment stats.
  4. Key regressions: round-0 exclusion in agg_merge_stats_by_behavior,
     multi-experiment deduplication bug.
"""
import pandas as pd
import pytest

import analysis.aggregations as agg
from analysis_helpers import make_users, make_metadata


# ---------------------------------------------------------------------------
# agg_global_accuracy_loss_by_round
# ---------------------------------------------------------------------------

class TestAggGlobalAccuracyLossByRound:

    def test_output_columns(self, two_exp_global):
        result = agg.agg_global_accuracy_loss_by_round(two_exp_global)
        assert set(result.columns) == {"round", "accuracy_mean", "accuracy_std", "loss_mean", "loss_std"}

    def test_round_count_matches_unique_rounds(self, two_exp_global):
        result = agg.agg_global_accuracy_loss_by_round(two_exp_global)
        assert len(result) == two_exp_global["round"].nunique()

    def test_accuracy_mean_within_bounds(self, two_exp_global):
        result = agg.agg_global_accuracy_loss_by_round(two_exp_global)
        lo = two_exp_global["objective_global_accuracy"].min()
        hi = two_exp_global["objective_global_accuracy"].max()
        assert result["accuracy_mean"].between(lo, hi).all()

    def test_two_identical_experiments_zero_std(self):
        """When two experiments have identical values, std should be 0."""
        global_df = pd.DataFrame([
            {"experiment_id": "A", "round": 1, "objective_global_accuracy": 0.8, "objective_global_loss": 1.0},
            {"experiment_id": "B", "round": 1, "objective_global_accuracy": 0.8, "objective_global_loss": 1.0},
        ])
        result = agg.agg_global_accuracy_loss_by_round(global_df)
        assert result["accuracy_std"].iloc[0] == pytest.approx(0.0)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError):
            agg.agg_global_accuracy_loss_by_round(pd.DataFrame())



# ---------------------------------------------------------------------------
# agg_grs_by_role
# ---------------------------------------------------------------------------

class TestAggGrsByRole:

    def test_output_columns(self, two_exp_users, two_exp_metadata):
        result = agg.agg_grs_by_role(two_exp_users, two_exp_metadata)
        assert set(result.columns) == {"role", "round", "grs_mean", "grs_std"}

    def test_all_roles_present(self, two_exp_users, two_exp_metadata):
        result = agg.agg_grs_by_role(two_exp_users, two_exp_metadata)
        assert set(result["role"]) >= {"good", "bad", "freerider", "inactive"}

    def test_single_experiment_std_is_nan(self):
        """With only one experiment, cross-experiment std must be NaN."""
        pass  # imported at top
        users = make_users(["exp-A"])
        meta  = make_metadata(["exp-A"])
        result = agg.agg_grs_by_role(users, meta)
        assert result["grs_std"].isna().all()

    def test_two_identical_experiments_zero_std(self):
        """Two experiments with identical GRS values → std == 0."""
        pass  # imported at top
        users = make_users(["exp-A", "exp-B"])
        meta  = make_metadata(["exp-A", "exp-B"])
        result = agg.agg_grs_by_role(users, meta)
        assert result["grs_std"].dropna().eq(0).all()

    def test_inconsistent_activation_raises(self):
        pass  # imported at top
        users = make_users(["exp-A", "exp-B"])
        meta = make_metadata(["exp-A", "exp-B"])
        meta.loc[meta["experiment_id"] == "exp-B", "malicious_start_round"] = 5
        with pytest.raises(ValueError, match="different"):
            agg.agg_grs_by_role(users, meta)

    def test_empty_raises(self, two_exp_metadata):
        with pytest.raises(ValueError):
            agg.agg_grs_by_role(pd.DataFrame(), two_exp_metadata)


# ---------------------------------------------------------------------------
# grs_by_user
# ---------------------------------------------------------------------------

class TestGrsByUser:

    def test_output_columns(self, two_exp_users):
        pass  # imported at top
        single = make_users(["exp-A"])
        result = agg.grs_by_user(single)
        assert set(result.columns) >= {"grs", "user_id", "role", "round"}

    def test_multi_experiment_raises(self, two_exp_users):
        with pytest.raises(ValueError, match="single-experiment"):
            agg.grs_by_user(two_exp_users)

    def test_single_experiment_ok(self):
        pass  # imported at top
        result = agg.grs_by_user(make_users(["exp-A"]))
        assert not result.empty

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            agg.grs_by_user(pd.DataFrame())


# ---------------------------------------------------------------------------
# agg_gas_used_by_tx_type
# ---------------------------------------------------------------------------

class TestAggGasUsedByTxType:

    def _receipts(self):
        return pd.DataFrame([
            {"experiment_id": exp, "round": r, "tx_type": tx, "gas_used": 50000}
            for exp in ["exp-A", "exp-B"]
            for r in [1, 2]
            for tx in ["register", "slot", "weights"]
        ])

    def test_output_columns(self, two_exp_metadata):
        result = agg.agg_gas_used_by_tx_type(self._receipts(), two_exp_metadata)
        assert set(result.columns) == {"tx_type", "contribution_score_strategy", "gas_mean", "gas_std"}

    def test_tx_types_present(self, two_exp_metadata):
        result = agg.agg_gas_used_by_tx_type(self._receipts(), two_exp_metadata)
        assert set(result["tx_type"]) >= {"register", "slot", "weights"}

    def test_empty_raises(self, two_exp_metadata):
        with pytest.raises(ValueError):
            agg.agg_gas_used_by_tx_type(pd.DataFrame(), two_exp_metadata)


# ---------------------------------------------------------------------------
# agg_round_kicked_by_strategy
# ---------------------------------------------------------------------------

class TestAggRoundKickedByStrategy:

    def _disqualified_users(self):
        pass  # imported at top
        users = make_users(["exp-A"])
        meta  = make_metadata(["exp-A"])
        # Mark user 1 (role=bad) as disqualified from round 4 onward
        mask = (users["user_id"] == 1) & (users["round"] >= 4)
        users.loc[mask, "state"] = "disqualified"
        return users, meta

    def test_empty_when_no_disqualifications(self, two_exp_metadata):
        pass  # imported at top
        users = make_users(["exp-A", "exp-B"])  # all users stay "active"
        result = agg.agg_round_kicked_by_strategy(users, two_exp_metadata)
        assert result.empty

    def test_output_columns(self):
        users, meta = self._disqualified_users()
        result = agg.agg_round_kicked_by_strategy(users, meta)
        assert set(result.columns) == {
            "contribution_score_strategy", "role", "mean_round_kicked", "low_err", "high_err"
        }

    def test_mean_round_kicked_correct(self):
        users, meta = self._disqualified_users()
        result = agg.agg_round_kicked_by_strategy(users, meta)
        bad_row = result[result["role"] == "bad"].iloc[0]
        # First disqualified round for user 1 is 4
        assert bad_row["mean_round_kicked"] == pytest.approx(4.0)

    def test_error_bars_non_negative(self):
        users, meta = self._disqualified_users()
        result = agg.agg_round_kicked_by_strategy(users, meta)
        assert (result["low_err"] >= 0).all()
        assert (result["high_err"] >= 0).all()
