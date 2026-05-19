import pickle
import pandas as pd
import numpy as np
from pathlib import Path


class ExperimentLogger:

    def __init__(self, experiment_id, metadata: dict):
        self.experiment_id = experiment_id
        self.metadata = metadata

        # Accumulators (fast)
        self._global_rows = []
        self._user_rows = []
        self._vote_rows = []
        self._receipt_rows = []
        self._contribution_rows = []
        self._contribution_mad_rows = []
        self._warning_rows = []
        self._punishment_rows = []
        # self._eval_reward_rows = []
        # self._eval_vote_rows = []

    # -------- GLOBAL ROUND --------

    def global_round(self, round=None, round_time=None,
                     obj_global_acc=None, obj_global_loss=None,
                     reward_pool=None, punishment_pool=None,
                     agg_func_1=None, agg_weight_1=None,
                     agg_func_2=None, agg_weight_2=None):

        self._global_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "round_time": round_time,
            "objective_global_accuracy": obj_global_acc,
            "objective_global_loss": obj_global_loss,
            "reward_pool": reward_pool,
            "punishment_pool": punishment_pool,
            "agg_func_1": agg_func_1,
            "agg_weight_1": agg_weight_1,
            "agg_func_2": agg_func_2,
            "agg_weight_2": agg_weight_2
            # Add
        })

    # -------- USER ROUND --------

    def user_round(self, round=None, user_id=None, state=None, behavior=None, role=None,
                   grs=None,
                   sub_personal_acc=None, sub_personal_loss=None,
                   sub_global_acc=None, sub_global_loss=None,
                   round_reputation_assigned=None,
                   reward_delta=None,
                   merged=None,
                   merge_weight=None,
                   attack_type=None):

        self._user_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "user_id": user_id,
            "state": state,
            "behavior": behavior,
            "role": role,
            "attack_type": attack_type,
            "grs": grs,
            "subjective_personal_accuracy": sub_personal_acc,
            "subjective_personal_loss": sub_personal_loss,
            "subjective_global_accuracy": sub_global_acc,
            "subjective_global_loss": sub_global_loss,
            "round_reputation_assigned": round_reputation_assigned,
            "reward_delta": reward_delta,
            "merged": merged,
            "merge_weight": merge_weight,
        })

    # -------- VOTE --------

    def vote(self, round=None, giver_id=None, receiver_id=None, giver_address=None, receiver_address=None, vote_feedback_score=None, vote_prev_accuracy=None, vote_prev_loss=None, vote_accuracy=None, vote_loss=None):
        self._vote_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "giver_id": giver_id,
            "receiver_id": receiver_id,
            "giver_address": giver_address,
            "receiver_address": receiver_address,
            "vote_feedback_score": vote_feedback_score,
            "vote_prev_accuracy": vote_prev_accuracy,
            "vote_prev_loss": vote_prev_loss,
            "vote_accuracy": vote_accuracy,
            "vote_loss": vote_loss,
        })

    # -------- CONTRIBUTION SCORES (parent: one row per round x user) --------

    def contribution_scores(self, round=None, user_ids=None, user_addresses=None, scores=None):
        for user_id, address, score in zip(user_ids, user_addresses, scores):
            self._contribution_rows.append({
                "experiment_id":      self.experiment_id,
                "round":              round,
                "user_id":            user_id,
                "user_address":       address,
                "contribution_score": score,
            })

    # -------- CONTRIBUTION SCORE MAD (child: 0..N rows per parent, keyed by metric) --------

    def contribution_score_mad(self, round=None, user_ids=None, user_addresses=None,
                               metric=None, raw_values=None, outlier_info=None, previous_avg=None):
        # metric is the measured signal, not the scoring strategy:
        # "accuracy" | "loss" | "dotproduct". This distinguishes child rows
        # when a strategy emits more than one per user (accuracy_loss writes both).
        n = len(user_ids)
        if raw_values is None:
            raw_values = [None] * n
        if outlier_info is None:
            outlier_info = [{} for _ in range(n)]
        for user_id, address, raw_val, info in zip(user_ids, user_addresses, raw_values, outlier_info):
            row = {
                "experiment_id":    self.experiment_id,
                "round":            round,
                "user_id":          user_id,
                "user_address":     address,
                "metric":           metric,
                "user_mad_avg":               raw_val,
                "current_excluded_values":    info.get("current_removed", []),
                "current_accepted_values":    info.get("current_accepted", []),
                "current_mad_median":         info.get("current_median"),
                "current_mad_value":          info.get("current_mad"),
                "current_mad_max_deviation":  info.get("current_boundary"),
                "prev_avg_round_val_after_mad": previous_avg,
                "previous_excluded_values":   info.get("previous_removed", []),
                "previous_accepted_values":   info.get("previous_accepted", []),
                "previous_mad_median":        info.get("previous_median"),
                "previous_mad_value":         info.get("previous_mad"),
                "previous_mad_max_deviation": info.get("previous_boundary"),
            }
            if "dotproduct_outlier_weight_count" in info:
                row["dotproduct_outlier_weight_count"]    = info["dotproduct_outlier_weight_count"]
                row["dotproduct_outlier_weight_fraction"] = info["dotproduct_outlier_weight_fraction"]
            self._contribution_mad_rows.append(row)

    # -------- PUNISHMENT --------

    def punishment(self, round=None, user_id=None, user_address=None,
                   punishment_type=None, loss=None, round_score=None, new_reputation=None):
        self._punishment_rows.append({
            "experiment_id":  self.experiment_id,
            "round":          round,
            "user_id":        user_id,
            "user_address":   user_address,
            "punishment_type": punishment_type,  # "punishment" | "contribution" | "passive"
            "loss":           loss,
            "round_score":    round_score,
            "new_reputation": new_reputation,
        })


    # -------- RECEIPT --------

    def receipt(self, round=None, tx_type=None, tx_hash=None, gas_used=None):
        self._receipt_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "tx_type": tx_type,
            "tx_hash": tx_hash,
            "gas_used": gas_used,
        })

    # -------- RUNTIME WARNINGS --------

    def warning(self, round=None, message=None, user_id=None, user_address=None):
        self._warning_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "message": message,
            "user_id": user_id,
            "user_address": user_address,
        })

    # -------- SETUP --------

    def setup(self, total_experiment_time=None, hardware=None, config=None):
        """Capture a one-time snapshot of experiment context."""
        self._setup = {
            "total_experiment_time": total_experiment_time,
            "hardware": hardware,
            "config": config,
        }

    # -------- FINALIZE --------

    def finalize(self):
        return {
            "global":              pd.DataFrame(self._global_rows),
            "users":               pd.DataFrame(self._user_rows),
            "votes":               pd.DataFrame(self._vote_rows),
            "receipts":            pd.DataFrame(self._receipt_rows),
            "contributions":       pd.DataFrame(self._contribution_rows),
            "contributions_mad":   pd.DataFrame(self._contribution_mad_rows),
            "warnings":            pd.DataFrame(self._warning_rows),
            "punishments":         pd.DataFrame(self._punishment_rows),
            # "evaluation_rewards":  pd.DataFrame(self._eval_reward_rows),
            # "evaluation_votes":    pd.DataFrame(self._eval_vote_rows),
        }

    # -------- SAVE --------

    def save(self, path):
        path = Path(path)
        payload = {
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "setup": getattr(self, "_setup", {}),
            "tables": self.finalize(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class NullExperimentLogger:
    """No-op logger used when no ExperimentLogger is provided."""

    def global_round(self, round=None, round_time=None, obj_global_acc=None, obj_global_loss=None, reward_pool=None, punishment_pool=None, agg_func_1=None, agg_weight_1=None, agg_func_2=None, agg_weight_2=None): pass
    def user_round(self, round=None, user_id=None, state=None, behavior=None, role=None, grs=None, sub_personal_acc=None, sub_personal_loss=None, sub_global_acc=None, sub_global_loss=None, round_reputation_assigned=None, reward_delta=None, merged=None, merge_weight=None, attack_type=None): pass
    def vote(self, round=None, giver_id=None, receiver_id=None, giver_address=None, receiver_address=None, vote_feedback_score=None, vote_prev_accuracy=None, vote_prev_loss=None, vote_accuracy=None, vote_loss=None): pass
    def contribution_scores(self, round=None, user_ids=None, user_addresses=None, scores=None): pass
    def contribution_score_mad(self, round=None, user_ids=None, user_addresses=None, metric=None, raw_values=None, outlier_info=None, previous_avg=None): pass
    def punishment(self, round=None, user_id=None, user_address=None, punishment_type=None, loss=None, round_score=None, new_reputation=None): pass
    # def evaluation_voting_reward(self, round=None, user_id=None, user_address=None, staked=None, rewarded=None, new_reputation=None): pass
    # def evaluation_vote(self, round=None, evaluated_user_id=None, evaluated_user_address=None, voter_user_id=None, voter_user_address=None, loss_vote=None, avg_loss_true_value=None, softmax_reward=None): pass
    def receipt(self, round=None, tx_type=None, tx_hash=None, gas_used=None): pass
    def warning(self, round=None, message=None): pass
    def setup(self, total_experiment_time=None, hardware=None, config=None): pass
    def finalize(self): pass
    def save(self, path=None): pass
