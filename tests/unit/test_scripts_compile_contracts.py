import importlib.util
import sys
from pathlib import Path


def test_compile_contracts_runs_with_stubs(tmp_path, monkeypatch):
    # Prepare fake solcx module
    class FakeSolcx:
        def __init__(self):
            self.installed = []
            self.version = None
            self.compiled = None

        def install_solc(self, version):
            self.installed.append(version)

        def set_solc_version(self, version):
            self.version = version

        def compile_standard(self, config):
            self.compiled = config
            return {
                "contracts": {
                    "OpenFLModel.sol": {"OpenFLModel": {"evm": {"bytecode": {"object": "aa"}}, "abi": [1]}},
                    "OpenFLManager.sol": {"OpenFLManager": {"evm": {"bytecode": {"object": "bb"}}, "abi": [2]}},
                }
            }

    fake = FakeSolcx()
    monkeypatch.setitem(sys.modules, "solcx", fake)

    # Create temporary module copy with __file__ pointing to tmp_path
    src_path = Path("scripts/compile_contracts.py")
    code = src_path.read_text()
    module_path = tmp_path / "pkg" / "compile_contracts.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(code)

    # Create fake contract sources
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    (contracts_dir / "OpenFLManager.sol").write_text("pragma solidity ^0.8.9;", encoding="utf-8")
    (contracts_dir / "OpenFLModel.sol").write_text("pragma solidity ^0.8.9;", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tmp_compile_contracts", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tmp_compile_contracts"] = module

    # Ensure module sees correct __file__ and Path operations
    module.__file__ = str(module_path)
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    build_dir = tmp_path / "artifacts" / "bytecode"
    assert build_dir.exists()
    assert (build_dir / "abi.txt").exists()
    assert (build_dir / "bytecode.txt").exists()
    assert fake.installed == ["0.8.9"]
    assert fake.version == "0.8.9"
