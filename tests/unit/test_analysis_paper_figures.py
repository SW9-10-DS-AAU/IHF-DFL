"""Tests for analysis/paper_figures.py helpers (small synthetic frames).

The acceptance numbers on the 480 stored runs live in
tests/integration/test_paper_figures_acceptance.py (slow).
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from analysis import paper_figures as pf


def _users(rows):
    return pd.DataFrame(rows, columns=["experiment_id", "round", "user_id", "role", "state"])


def test_removal_round_is_first_disqualified_round():
    users = _users([
        ("e", 0, 0, "good", "active"), ("e", 1, 0, "good", "active"),
        ("e", 2, 0, "good", "disqualified"), ("e", 3, 0, "good", "disqualified"),
    ])
    out = pf.removal_rounds(users)
    assert out.loc[0, "removed_round"] == 2


def test_removal_round_is_nan_for_never_disqualified():
    users = _users([("e", 0, 0, "good", "active"), ("e", 1, 0, "good", "active")])
    out = pf.removal_rounds(users)
    assert pd.isna(out.loc[0, "removed_round"])
    assert not out.loc[0, "removed"]


def test_removal_round_keeps_role_and_separates_users_and_runs():
    users = _users([
        ("e1", 1, 0, "good", "disqualified"),
        ("e1", 1, 1, "freerider", "active"),
        ("e2", 4, 0, "freerider", "disqualified"),
    ])
    out = pf.removal_rounds(users).set_index(["experiment_id", "user_id"])
    assert out.loc[("e1", 0), "removed_round"] == 1
    assert out.loc[("e1", 1), "role"] == "freerider"
    assert pd.isna(out.loc[("e1", 1), "removed_round"])
    assert out.loc[("e2", 0), "removed_round"] == 4


def test_post_activation_delay_is_removal_round_minus_activation_round():
    assert pf.post_activation_delay(removed_round=7, activation_round=3) == 4


def test_post_activation_delay_is_nan_when_never_removed():
    assert np.isnan(pf.post_activation_delay(removed_round=np.nan, activation_round=3))


def test_wilson_interval_known_value():
    # Textbook example: 1 success in 10 trials, 95% -> (0.0179, 0.4041).
    lo, hi = pf.wilson_interval(1, 10)
    assert lo == pytest.approx(0.0179, abs=1e-4)
    assert hi == pytest.approx(0.4041, abs=1e-4)


def test_wilson_interval_zero_successes_has_zero_lower_bound_and_positive_upper():
    lo, hi = pf.wilson_interval(0, 60)
    assert lo == 0.0
    assert 0.0 < hi < 0.07


def test_wilson_interval_with_no_trials_is_nan():
    lo, hi = pf.wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_is_deterministic_for_same_seed():
    x = np.arange(20.0)
    assert pf.bootstrap_ci(x) == pf.bootstrap_ci(x)


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(0).normal(10, 2, size=50)
    lo, hi = pf.bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_bootstrap_ci_of_constant_values_is_that_constant():
    assert pf.bootstrap_ci(np.full(10, 3.0)) == (3.0, 3.0)


def test_within_round_variance_is_population_variance_averaged_over_rounds():
    # round 1: [0, 2] -> var 1 ; round 2: [0, 4] -> var 4 ; mean = 2.5
    c = pd.DataFrame({
        "experiment_id": ["e"] * 4, "round": [1, 1, 2, 2], "user_id": [0, 1, 0, 1],
        "contribution_score": [0.0, 2.0, 0.0, 4.0],
    })
    assert pf.mean_within_round_variance(c)["variance"].iloc[0] == pytest.approx(2.5)


def test_round_one_gas_sums_round_one_receipts_per_run_only():
    r = pd.DataFrame({
        "experiment_id": ["e", "e", "e", "f"], "round": [1, 1, 2, 1],
        "tx_type": ["slot", "weights", "slot", "slot"], "gas_used": [10, 20, 999, 5],
    })
    out = pf.round_one_gas_per_run(r).set_index("experiment_id")["gas_used"]
    assert out.to_dict() == {"e": 30, "f": 5}
