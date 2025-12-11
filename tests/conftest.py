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

# Add src folder to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Stub artifacts package used during contract imports
if "artifacts" not in sys.modules:
    artifacts_mod = types.ModuleType("artifacts")
    bytecode_mod = types.ModuleType("artifacts.bytecode")
    abi_model_mod = types.ModuleType("artifacts.bytecode.abi_model")
    abi_model_mod.OPEN_FL_MODEL_ABI = []

    sys.modules["artifacts"] = artifacts_mod
    sys.modules["artifacts.bytecode"] = bytecode_mod
    sys.modules["artifacts.bytecode.abi_model"] = abi_model_mod