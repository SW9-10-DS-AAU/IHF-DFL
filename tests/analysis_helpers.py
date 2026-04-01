"""Shared factory functions for analysis test fixtures.

Kept in a separate importable module so test files can import directly
(pytest conftest.py is not reliably importable as a module).
"""
import pandas as pd

_ROLES_CONFIG = [
    ("good",      None),
    ("bad",       2),
    ("freerider", 2),
    ("inactive",  None),
]


def make_users(experiment_ids, roles_config=None, n_rounds=6):
    """Build a users DataFrame.

    merged logic:
      round 0                   → None  (no merging in init round)
      round >= 4, role != good  → False (excluded from merge)
      otherwise                 → True
    """
    if roles_config is None:
        roles_config = _ROLES_CONFIG
    rows = []
    for exp in experiment_ids:
        for uid, (role, switch_round) in enumerate(roles_config):
            for r in range(n_rounds):
                if role in ("good", "inactive") or switch_round is None:
                    behavior = role
                else:
                    behavior = "good" if r < switch_round else role
                if r == 0:
                    merged = None
                    merge_weight = None
                elif role != "good" and r >= 4:
                    merged = False
                    merge_weight = None
                else:
                    merged = True
                    merge_weight = 0.25
                rows.append({
                    "experiment_id":                exp,
                    "round":                        r,
                    "user_id":                      uid,
                    "role":                         role,
                    "behavior":                     behavior,
                    "state":                        "active",
                    "grs":                          (r + 1) * int(1e18),
                    "reward_delta":                 int(1e18),
                    "is_reward":                    True,
                    "merged":                       merged,
                    "merge_weight":                 merge_weight,
                    "subjective_personal_accuracy": 0.80 + r * 0.01,
                    "subjective_personal_loss":     100 - r,
                    "round_reputation_assigned":    100,
                })
    return pd.DataFrame(rows)


def make_global(experiment_ids, n_rounds=6):
    rows = []
    for exp in experiment_ids:
        for r in range(n_rounds):
            rows.append({
                "experiment_id":             exp,
                "round":                     r,
                "objective_global_accuracy": 0.70 + r * 0.02,
                "objective_global_loss":     200 - r * 10,
                "reward_pool":               int(10e18),
                "punishment_pool":           0,
                "round_time":                1.5,
            })
    return pd.DataFrame(rows)


def make_metadata(experiment_ids):
    return pd.DataFrame([
        {
            "experiment_id":                    eid,
            "dataset":                          "mnist",
            "aggregation_rule":                 "FedAVG",
            "contribution_score_strategy":      "accuracy_loss",
            "malicious_start_round":            2,
            "freerider_start_round":            2,
            "freerider_noise_scale":            0.1,
            "malicious_noise_scale":            0.5,
            "use_outlier_detection":            True,
            "force_merge_all":                  False,
            "number_of_good_contributors":      1,
            "number_of_bad_contributors":       1,
            "number_of_freerider_contributors": 1,
            "number_of_inactive_contributors":  1,
            "minimum_rounds":                   5,
        }
        for eid in experiment_ids
    ])
