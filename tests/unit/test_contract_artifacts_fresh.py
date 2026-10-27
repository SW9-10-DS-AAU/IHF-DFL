"""
The experiments deploy the bytecode in artifacts/bytecode/, which is gitignored and only
rebuilt by scripts/compile_contracts.py. On 2026-10-27 a run crashed in settle() because the
local artifacts predated the underflow fix in contracts/OpenFLModel.sol (8f39f19).
This test recompiles the sources with the same settings and requires the artifacts to match.
"""
import json
from pathlib import Path

import pytest

solcx = pytest.importorskip("solcx")

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
BUILD = ROOT / "artifacts" / "bytecode"
SOLC_VERSION = "0.8.9"

# source file -> (contract name, bytecode artifact, abi artifact); mirrors scripts/compile_contracts.py
ARTIFACTS = {
    "OpenFLManager.sol": ("OpenFLManager", "bytecode_mgr.txt", "abi_mgr.txt"),
    "OpenFLManager_NobodyIsKicked.sol": ("OpenFLManager_NobodyIsKicked", "bytecode_mgr_nobody.txt", "abi_mgr_nobody.txt"),
    "OpenFLModel.sol": ("OpenFLModel", "bytecode_model.txt", "abi_model.txt"),
    "OpenFLModel_NobodyIsKicked.sol": ("OpenFLModel_NobodyIsKicked", "bytecode_model_nobody.txt", "abi_model_nobody.txt"),
}
SOURCE_DIRS = {
    "OpenFLManager.sol": CONTRACTS,
    "OpenFLManager_NobodyIsKicked.sol": CONTRACTS / "harnesses",
    "OpenFLModel.sol": CONTRACTS,
    "OpenFLModel_NobodyIsKicked.sol": CONTRACTS / "harnesses",
}


@pytest.fixture(scope="module")
def compiled():
    if not BUILD.exists():
        pytest.skip("artifacts/bytecode not built - run scripts/compile_contracts.py")
    if SOLC_VERSION not in {str(v) for v in solcx.get_installed_solc_versions()}:
        pytest.skip(f"solc {SOLC_VERSION} not installed")
    sources = {name: {"content": (SOURCE_DIRS[name] / name).read_text(encoding="utf-8")} for name in ARTIFACTS}
    return solcx.compile_standard({
        "language": "Solidity",
        "sources": sources,
        "settings": {
            "optimizer": {"enabled": True, "runs": 200},
            "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}},
        },
    }, solc_version=SOLC_VERSION)


@pytest.mark.parametrize("source", sorted(ARTIFACTS))
def test_deployed_bytecode_matches_contract_source(compiled, source):
    contract, bytecode_file, abi_file = ARTIFACTS[source]
    out = compiled["contracts"][source][contract]
    stale = "artifacts are stale - run scripts/compile_contracts.py"
    assert (BUILD / bytecode_file).read_text(encoding="utf-8").strip() == out["evm"]["bytecode"]["object"], stale
    assert json.loads((BUILD / abi_file).read_text(encoding="utf-8")) == out["abi"], stale
