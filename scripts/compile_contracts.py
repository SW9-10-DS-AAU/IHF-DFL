import os
import platform
from solcx import install_solc, set_solc_version, compile_standard, get_installed_solc_versions
from pathlib import Path
import json
from utils.paths import repo_root

print(get_installed_solc_versions())

SOLC_VERSION = "0.8.9"
IS_ARM = platform.machine() in ("aarch64", "arm64")

compile_kwargs = {}
if IS_ARM:
    # solcx has no prebuilt ARM binary; use a system-installed solc.
    # Override via SOLC_BINARY env var if installed outside ~/.local/bin.
    compile_kwargs["solc_binary"] = os.environ.get(
        "SOLC_BINARY", str(Path.home() / ".local/bin/solc")
    )
else:
    install_solc(SOLC_VERSION)
    set_solc_version(SOLC_VERSION)

# 2) Load sources
root = repo_root(Path(__file__))
contracts_dir = root / "contracts"
harnesses_contracts_dir = root / "contracts" / "harnesses"
sources = {
    "OpenFLManager.sol": {"content": (contracts_dir / "OpenFLManager.sol").read_text(encoding="utf-8")},
    "OpenFLManager_NobodyIsKicked.sol": {"content": (harnesses_contracts_dir / "OpenFLManager_NobodyIsKicked.sol").read_text(encoding="utf-8")},
    "OpenFLModel.sol":   {"content": (contracts_dir / "OpenFLModel.sol").read_text(encoding="utf-8")},
    "OpenFLModel_NobodyIsKicked.sol": {"content": (harnesses_contracts_dir / "OpenFLModel_NobodyIsKicked.sol").read_text(encoding="utf-8")},
}

# 3) Compile
compiled = compile_standard({
    "language": "Solidity",
    "sources": sources,
    "settings": {
        "optimizer": {"enabled": True, "runs": 200},
        "outputSelection": {"*": {"*": ["abi","evm.bytecode.object"]}}
    }
}, **compile_kwargs)

bytecode = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLModel: {size_bytes} bytes ({size_kb:.2f} KB)")


bytecode = compiled["contracts"]["OpenFLModel_NobodyIsKicked.sol"]["OpenFLModel_NobodyIsKicked"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLModel_NobodyIsKicked: {size_bytes} bytes ({size_kb:.2f} KB)")


bytecode = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLManager: {size_bytes} bytes ({size_kb:.2f} KB)")


bytecode = compiled["contracts"]["OpenFLManager_NobodyIsKicked.sol"]["OpenFLManager_NobodyIsKicked"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLManager_NobodyIsKicked: {size_bytes} bytes ({size_kb:.2f} KB)")


# 4) Extract artifacts
mgr = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]
mgr_nobody = compiled["contracts"]["OpenFLManager_NobodyIsKicked.sol"]["OpenFLManager_NobodyIsKicked"]
mdl = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]
mdl_nobody = compiled["contracts"]["OpenFLModel_NobodyIsKicked.sol"]["OpenFLModel_NobodyIsKicked"]

build = root / "artifacts" / "bytecode"
build.mkdir(parents=True, exist_ok=True)

# IMPORTANT: abi.txt should be JSON, because Python should json.load it later
(Path(build / "abi_mgr.txt")).write_text(json.dumps(mgr["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_mgr.txt")).write_text(mgr["evm"]["bytecode"]["object"], encoding="utf-8")
(Path(build / "abi_mgr_nobody.txt")).write_text(json.dumps(mgr_nobody["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_mgr_nobody.txt")).write_text(mgr_nobody["evm"]["bytecode"]["object"], encoding="utf-8")

(Path(build / "abi_model.txt")).write_text(json.dumps(mdl["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "abi_model_nobody.txt")).write_text(json.dumps(mdl_nobody["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_model.txt")).write_text(mdl["evm"]["bytecode"]["object"], encoding="utf-8")
(Path(build / "bytecode_model_nobody.txt")).write_text(mdl_nobody["evm"]["bytecode"]["object"], encoding="utf-8")


print("Artifacts written to build/: abi.txt, OPEN_FL_MODEL_ABI.py, bytecode.txt, abi_model.txt, bytecode_model.txt and similar for nobody_is_kicked")
