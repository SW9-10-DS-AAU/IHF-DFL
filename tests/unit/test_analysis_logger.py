"""Tests for analysis/ExperimentLogger.py — logging, finalize, save/load roundtrip."""
import pandas as pd
import pytest

from analysis.ExperimentLogger import ExperimentLogger, NullExperimentLogger
from analysis.loader import load_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populated_logger(experiment_id="test-exp"):
    logger = ExperimentLogger(experiment_id, metadata={"dataset": "mnist"})
    logger.log_global_round(
        round=1, round_time=1.5,
        obj_global_acc=0.85, obj_global_loss=120,
        reward_pool=int(10e18), punishment_pool=0,
    )
    logger.log_user_round(
        round=1, user_id=0, state="active", behavior="good", role="good",
        grs=int(1e18), sub_personal_acc=0.82, sub_personal_loss=110,
        sub_global_acc=0.85, sub_global_loss=120,
        round_reputation_assigned=100,
        reward_delta=int(5e17), is_reward=True,
        merged=True, merge_weight=0.25,
    )
    logger.log_vote(
        round=1, giver_id=0, receiver_id=1,
        giver_address="0xAAA", receiver_address="0xBBB",
        vote_feedback_score=1,
        vote_prev_accuracy=0.80, vote_prev_loss=115,
        vote_accuracy=0.82, vote_loss=110,
    )
    logger.log_receipt(round=1, tx_type="weights", tx_hash="0xDEAD", gas_used=21000)
    logger.log_warning(round=1, message="test warning")
    return logger


# ---------------------------------------------------------------------------
# ExperimentLogger — finalize
# ---------------------------------------------------------------------------

def test_finalize_returns_all_tables():
    logger = ExperimentLogger("exp", {})
    tables = logger.finalize()
    assert set(tables.keys()) == {"global", "users", "votes", "receipts", "contributions", "warnings"}


def test_global_table_columns():
    logger = _populated_logger()
    tables = logger.finalize()
    df = tables["global"]
    for col in ("experiment_id", "round", "objective_global_accuracy", "objective_global_loss",
                "reward_pool", "punishment_pool", "round_time"):
        assert col in df.columns, f"Missing column: {col}"


def test_users_table_columns():
    logger = _populated_logger()
    tables = logger.finalize()
    df = tables["users"]
    for col in ("experiment_id", "round", "user_id", "state", "behavior", "role",
                "grs", "merged", "merge_weight", "reward_delta"):
        assert col in df.columns, f"Missing column: {col}"


def test_votes_table_columns():
    logger = _populated_logger()
    tables = logger.finalize()
    df = tables["votes"]
    for col in ("experiment_id", "round", "giver_id", "receiver_id", "vote_feedback_score"):
        assert col in df.columns


def test_receipts_table_columns():
    logger = _populated_logger()
    tables = logger.finalize()
    df = tables["receipts"]
    for col in ("experiment_id", "round", "tx_type", "gas_used"):
        assert col in df.columns


def test_warnings_table_columns():
    logger = _populated_logger()
    tables = logger.finalize()
    df = tables["warnings"]
    for col in ("experiment_id", "round", "message"):
        assert col in df.columns


def test_global_table_row_values():
    logger = ExperimentLogger("my-exp", {})
    logger.log_global_round(round=3, round_time=2.0, obj_global_acc=0.9,
                            obj_global_loss=50, reward_pool=int(1e18), punishment_pool=0)
    df = logger.finalize()["global"]
    assert df["round"].iloc[0] == 3
    assert df["objective_global_accuracy"].iloc[0] == pytest.approx(0.9)
    assert df["experiment_id"].iloc[0] == "my-exp"


def test_empty_logger_produces_empty_dataframes():
    logger = ExperimentLogger("empty", {})
    tables = logger.finalize()
    assert tables["global"].empty
    assert tables["users"].empty
    assert tables["votes"].empty


def test_multiple_log_calls_produce_multiple_rows():
    logger = ExperimentLogger("exp", {})
    for r in range(1, 4):
        logger.log_global_round(round=r, round_time=1.0,
                                obj_global_acc=0.8, obj_global_loss=100,
                                reward_pool=0, punishment_pool=0)
    df = logger.finalize()["global"]
    assert len(df) == 3


# ---------------------------------------------------------------------------
# ExperimentLogger — save → load_run roundtrip
# ---------------------------------------------------------------------------

def test_save_creates_pkl_file(tmp_path):
    logger = _populated_logger("roundtrip-exp")
    path = tmp_path / "roundtrip-exp.pkl"
    logger.save(path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_load_run_after_save_returns_rundata(tmp_path):
    from analysis.loader import RunData
    logger = _populated_logger("saved-exp")
    path = tmp_path / "saved-exp.pkl"
    logger.save(path)
    run = load_run(path)
    assert isinstance(run, RunData)


def test_roundtrip_experiment_id_preserved(tmp_path):
    logger = _populated_logger("my-unique-id")
    path = tmp_path / "my-unique-id.pkl"
    logger.save(path)
    run = load_run(path)
    assert run.experiment_id == "my-unique-id"


def test_roundtrip_metadata_preserved(tmp_path):
    meta = {"dataset": "cifar", "aggregation_rule": "binary_switch"}
    logger = ExperimentLogger("exp", metadata=meta)
    path = tmp_path / "exp.pkl"
    logger.save(path)
    run = load_run(path)
    assert run.metadata["dataset"] == "cifar"
    assert run.metadata["aggregation_rule"] == "binary_switch"


def test_roundtrip_global_table_preserved(tmp_path):
    logger = _populated_logger("exp")
    path = tmp_path / "exp.pkl"
    logger.save(path)
    run = load_run(path)
    assert not run.rounds_global.empty
    assert "objective_global_accuracy" in run.rounds_global.columns


def test_roundtrip_users_table_preserved(tmp_path):
    logger = _populated_logger("exp")
    path = tmp_path / "exp.pkl"
    logger.save(path)
    run = load_run(path)
    assert not run.rounds_users.empty
    assert "grs" in run.rounds_users.columns


def test_roundtrip_setup_preserved(tmp_path):
    logger = ExperimentLogger("exp", {})
    logger.log_setup(total_experiment_time=42.0, hardware="gpu", config={"lr": 0.01})
    path = tmp_path / "exp.pkl"
    logger.save(path)
    run = load_run(path)
    assert run.setup["total_experiment_time"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# NullExperimentLogger — no-op, never raises
# ---------------------------------------------------------------------------

def test_null_logger_all_methods_do_not_raise():
    null = NullExperimentLogger()
    null.log_global_round(round=1, round_time=1.0, obj_global_acc=0.9,
                          obj_global_loss=50, reward_pool=0, punishment_pool=0,
                          agg_func_1="positives_only", agg_weight_1=0.7,
                          agg_func_2="plus_one_normalize", agg_weight_2=0.3)
    null.log_user_round(round=1, user_id=0, state="active", behavior="good", role="good",
                        grs=0, sub_personal_acc=0, sub_personal_loss=0,
                        sub_global_acc=0, sub_global_loss=0,
                        round_reputation_assigned=0,
                        reward_delta=0, is_reward=True, merged=True, merge_weight=0.25)
    null.log_vote(round=1, giver_id=0, receiver_id=1,
                  giver_address="0x0", receiver_address="0x1",
                  vote_feedback_score=1, vote_prev_accuracy=0,
                  vote_prev_loss=0, vote_accuracy=0, vote_loss=0)
    null.log_contribution_scores(round=1, user_ids=[0], user_addresses=["0x0"],
                                 scores=[100], raw_values=None, outlier_info=None, previous_avg=0)
    null.log_receipt(round=1, tx_type="weights", tx_hash="0x0", gas_used=21000)
    null.log_warning(round=1, message="warn")
    null.log_setup(total_experiment_time=1.0, hardware="cpu", config={})
    null.finalize()
    null.save(path=None)
