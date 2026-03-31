import os
import sys
import types

# Provide a lightweight yaml stub to avoid external dependency during tests
if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(_stream):
        return {
            "printing": {"ONLY_PRINT_ROUND_SUMMARY": False},
            "contracts": {
                "WAIT_DELAY": 172800,
                "FEEDBACK_ROUND_TIMEOUT": 30,
                "CONTRIBUTION_ROUND_TIMEOUT": 30,
            },
        }

    yaml_stub.safe_load = _safe_load
    sys.modules["yaml"] = yaml_stub

# Add src and tests folders to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Stub artifacts package used during contract imports
if "artifacts" not in sys.modules:
    artifacts_mod = types.ModuleType("artifacts")
    bytecode_mod = types.ModuleType("artifacts.bytecode")
    abi_model_mod = types.ModuleType("artifacts.bytecode.abi_model")
    abi_model_mod.OPEN_FL_MODEL_ABI = []

    sys.modules["artifacts"] = artifacts_mod
    sys.modules["artifacts.bytecode"] = bytecode_mod
    sys.modules["artifacts.bytecode.abi_model"] = abi_model_mod


# ---------------------------------------------------------------------------
# Shared analysis fixtures
# ---------------------------------------------------------------------------

import pandas as pd
import pytest

from analysis.loader import RunData
from analysis_helpers import make_users, make_global, make_metadata


@pytest.fixture
def two_exp_users():
    return make_users(["exp-A", "exp-B"])


@pytest.fixture
def two_exp_global():
    return make_global(["exp-A", "exp-B"])


@pytest.fixture
def two_exp_metadata():
    return make_metadata(["exp-A", "exp-B"])


@pytest.fixture
def minimal_run():
    """Single-experiment RunData; suitable for load/save roundtrips."""
    return RunData(
        experiment_id="test-exp",
        metadata={
            "dataset": "mnist",
            "contribution_score_strategy": "accuracy_loss",
            "aggregation_rule": "FedAVG",
            "malicious_start_round": 2,
            "freerider_start_round": 2,
            "use_outlier_detection": True,
            "number_of_good_contributors": 1,
            "number_of_bad_contributors": 1,
            "number_of_freerider_contributors": 1,
            "number_of_inactive_contributors": 1,
            "minimum_rounds": 5,
        },
        setup={},
        rounds_global=make_global(["test-exp"]),
        rounds_users=make_users(["test-exp"]),
        votes=pd.DataFrame(),
        receipts=pd.DataFrame(),
        contributions=pd.DataFrame(),
        warnings=pd.DataFrame(),
    )