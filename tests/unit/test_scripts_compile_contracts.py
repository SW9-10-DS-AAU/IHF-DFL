import importlib.util
import types
import sys
from pathlib import Path

# Vi må lige overveje om denne test er relevant, maybe delete
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
                    "OpenFLModel_NobodyIsKicked.sol": {"OpenFLModel_NobodyIsKicked": {"evm": {"bytecode": {"object": "cc"}}, "abi": [3]}},
                    "OpenFLManager_NobodyIsKicked.sol": {"OpenFLManager_NobodyIsKicked": {"evm": {"bytecode": {"object": "dd"}}, "abi": [4]}},
                }
            }

    fake = FakeSolcx()
    solcx_stub = types.ModuleType("solcx")
    solcx_stub.install_solc = fake.install_solc
    solcx_stub.set_solc_version = fake.set_solc_version
    solcx_stub.compile_standard = fake.compile_standard
    solcx_stub.get_installed_solc_versions = lambda: list(fake.installed)

    monkeypatch.setitem(sys.modules, "solcx", solcx_stub)

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "scripts" / "compile_contracts.py"
    code = src_path.read_text(encoding="utf-8")


    module_path = tmp_path / "pkg" / "compile_contracts.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(code)

    # Create fake contract sources
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    (contracts_dir / "OpenFLManager.sol").write_text("pragma solidity ^0.8.9;", encoding="utf-8")
    (contracts_dir / "OpenFLModel.sol").write_text("pragma solidity ^0.8.9;", encoding="utf-8")

    harnesses_dir = contracts_dir / "harnesses"
    harnesses_dir.mkdir()
    (harnesses_dir / "OpenFLManager_NobodyIsKicked.sol").write_text("pragma solidity ^0.8.9;", encoding="utf-8")
    (harnesses_dir / "OpenFLModel_NobodyIsKicked.sol").write_text("pragma solidity ^0.8.9;", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("tmp_compile_contracts", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tmp_compile_contracts"] = module

    # Ensure module sees correct __file__ and Path operations
    module.__file__ = str(module_path)
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    build_dir = tmp_path / "artifacts" / "bytecode"
    assert build_dir.exists()
    for name in (
        "abi_mgr.txt", "bytecode_mgr.txt",
        "abi_mgr_nobody.txt", "bytecode_mgr_nobody.txt",
        "abi_model.txt", "bytecode_model.txt",
        "abi_model_nobody.txt", "bytecode_model_nobody.txt",
    ):
        assert (build_dir / name).exists(), f"missing {name}"
    assert fake.installed == ["0.8.9"]
    assert fake.version == "0.8.9"
