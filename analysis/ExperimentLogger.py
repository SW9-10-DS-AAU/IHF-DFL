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

    # -------- GLOBAL ROUND --------

    def log_global_round(self, round, round_time,
                         global_accuracy, global_loss,
                         reward_pool, punishment_pool
                         ):

        self._global_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "round_time": round_time,
            "global_accuracy": global_accuracy,
            "global_loss": global_loss,
            "reward_pool": reward_pool,
            "punishment_pool": punishment_pool
        })

    # -------- USER ROUND --------

    def log_user_round(self, round, user_id, state, behavior, role,
                       accuracy, loss, grs,
                       prev_global_acc, prev_global_loss,
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
            "accuracy": accuracy,
            "loss": loss,
            "grs": grs,
            "prev_global_accuracy": prev_global_acc,
            "prev_global_loss": prev_global_loss,
            "contribution_score": contribution_score,
            # "is_negative_contrib": is_negative_contrib,
            "round_reputation_assigned": round_reputation_assigned,
            "reward_delta": reward_delta,
            "is_reward": is_reward,
            "merged": merged
        })

    # -------- VOTE --------

    def log_vote(self, round, giver_id, receiver_id, vote_score,
                 giver_address, receiver_address):
        self._vote_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "giver_id": giver_id,
            "receiver_id": receiver_id,
            "vote_score": vote_score,
            "giver_address": giver_address,
            "receiver_address": receiver_address,
        })

    # -------- CONTRIBUTION SCORES --------

    def log_contribution_scores(self, round, user_ids, user_addresses, raw_values, outlier_info, scores, avg_prev):
        for user_id, address, raw_val, info, score in zip(user_ids, user_addresses, raw_values, outlier_info, scores):
            self._contribution_rows.append({
                "experiment_id":    self.experiment_id,
                "round":            round,
                "user_id":          user_id,
                "user_address":     address,
                "raw_avg_value":    raw_val,
                "outliers_removed": info.get("removed", []),
                "mad_median":       info.get("median"),
                "mad_value":        info.get("mad"),
                "mad_boundary":     info.get("boundary"),
                "contribution_score": score,
                "avg_prev":         avg_prev,
            })

    # -------- RECEIPT --------

    def log_receipt(self, round, tx_type, tx_hash, gas_used):
        self._receipt_rows.append({
            "experiment_id": self.experiment_id,
            "round": round,
            "tx_type": tx_type,
            "tx_hash": tx_hash,
            "gas_used": gas_used,
        })

    # -------- SETUP --------

    def log_setup(self, total_experiment_time, hardware, config, users_roster):
        """Capture a one-time snapshot of experiment context."""
        self._setup = {
            "total_experiment_time": total_experiment_time,
            "hardware": hardware,
            "config": config,
            "users": pd.DataFrame(users_roster),
        }

    # -------- FINALIZE --------

    def finalize(self):
        return {
            "global":        pd.DataFrame(self._global_rows),
            "users":         pd.DataFrame(self._user_rows),
            "votes":         pd.DataFrame(self._vote_rows),
            "receipts":      pd.DataFrame(self._receipt_rows),
            "contributions": pd.DataFrame(self._contribution_rows),
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
