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
        self._warning_rows = []

    # -------- GLOBAL ROUND --------

    def log_global_round(self, round, round_time,
                         obj_global_acc, obj_global_loss,
                         reward_pool, punishment_pool
                         ):

        self._global_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "round_time": round_time,
            "objective_global_accuracy": obj_global_acc,
            "objective_global_loss": obj_global_loss,
            "reward_pool": reward_pool,
            "punishment_pool": punishment_pool

            # Add
        })

    # -------- USER ROUND --------

    def log_user_round(self, round, user_id, state, behavior, role,
                       grs,
                       sub_personal_acc, sub_personal_loss,
                       sub_global_acc, sub_global_loss,
                       contribution_score,
                       round_reputation_assigned,
                       reward_delta,
                       is_reward,
                       merged):

        self._user_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "user_id": user_id,
            "state": state,
            "behavior": behavior,
            "role": role,
            "grs": grs,
            "subjective_personal_accuracy": sub_personal_acc,
            "subjective_personal_loss": sub_personal_loss,
            "subjective_global_accuracy": sub_global_acc,
            "subjective_global_loss": sub_global_loss,
            "round_reputation_assigned": round_reputation_assigned,
            "reward_delta": reward_delta,
            "is_reward": is_reward,
            "merged": merged
        })

    # -------- VOTE --------

    def log_vote(self, round, giver_id, receiver_id, giver_address, receiver_address, votes_feedback_score, votes_prev_accuracy, votes_prev_loss, votes_accuracy, votes_loss):
        self._vote_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "giver_id": giver_id,
            "receiver_id": receiver_id,
            "giver_address": giver_address,
            "receiver_address": receiver_address,
            "votes_feedback_score": votes_feedback_score,
            "votes_prev_accuracy": votes_prev_accuracy,
            "votes_prev_loss": votes_prev_loss,
            "votes_accuracy": votes_accuracy,
            "votes_loss": votes_loss,
        })

    # -------- CONTRIBUTION SCORES --------

    def log_contribution_scores(self, round, user_ids, user_addresses, scores, raw_values, outlier_info, previous_avg):
        for user_id, address, raw_val, info, score in zip(user_ids, user_addresses, raw_values, outlier_info, scores):
            self._contribution_rows.append({
                "experiment_id":    self.experiment_id,
                "round":            round,
                "user_id":          user_id,
                "user_address":     address,
                "contribution_score": score,
                "user_mad_avg":               raw_val,
                # current (per-user) MAD stats
                "current_excluded_values":    info.get("current_removed", []),
                "current_accepted_values":    info.get("current_accepted", []),
                "current_mad_median":         info.get("current_median"),
                "current_mad_value":          info.get("current_mad"),
                "current_mad_max_deviation":  info.get("current_boundary"),
                # previous (global baseline) MAD stats — same value for all users in a round
                "prev_avg_round_val_after_mad": previous_avg,
                "previous_excluded_values":   info.get("previous_removed", []),
                "previous_accepted_values":   info.get("previous_accepted", []),
                "previous_mad_median":        info.get("previous_median"),
                "previous_mad_value":         info.get("previous_mad"),
                "previous_mad_max_deviation": info.get("previous_boundary"),
            })

    # -------- RECEIPT --------

    def log_receipt(self, round, tx_type, tx_hash, gas_used):
        self._receipt_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "tx_type": tx_type,
            "tx_hash": tx_hash,
            "gas_used": gas_used,
            # TODO: Maybe an address or user_id?
        })

    # -------- RUNTIME WARNINGS --------

    def log_warning(self, round, message: str):
        self._warning_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "message": message,
            # TODO: Maybe an address or user_id?
        })

    # -------- SETUP --------

    def log_setup(self, total_experiment_time, hardware, config, users_roster):
        """Capture a one-time snapshot of experiment context."""
        self._setup = {
            "total_experiment_time": total_experiment_time,
            "hardware": hardware,
            "config": config,
        }

    # -------- FINALIZE --------

    def finalize(self):
        return {
            "global":        pd.DataFrame(self._global_rows),
            "users":         pd.DataFrame(self._user_rows),
            "votes":         pd.DataFrame(self._vote_rows),
            "receipts":      pd.DataFrame(self._receipt_rows),
            "contributions": pd.DataFrame(self._contribution_rows),
            "warnings":      pd.DataFrame(self._warning_rows),
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
