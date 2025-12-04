from solcx import install_solc, set_solc_version, compile_standard
from pathlib import Path
import json

# 1) Ensure exact compiler
install_solc("0.8.9")
set_solc_version("0.8.9")

# 2) Load sources
root = Path(__file__).parents[1]
contracts_dir = root / "contracts"
sources = {
    "OpenFLManager.sol": {"content": (contracts_dir / "OpenFLManager.sol").read_text(encoding="utf-8")},
    "OpenFLModel.sol":   {"content": (contracts_dir / "OpenFLModel.sol").read_text(encoding="utf-8")},
}

# 3) Compile
compiled = compile_standard({
    "language": "Solidity",
    "sources": sources,
    "settings": {
        "optimizer": {"enabled": True, "runs": 200},
        "outputSelection": {"*": {"*": ["abi","evm.bytecode.object"]}}
    }
})

bytecode = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size: {size_bytes} bytes ({size_kb:.2f} KB)")

# 4) Extract artifacts
mgr = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]
mdl = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]

build = root / "artifacts" / "bytecode"
build.mkdir(parents=True, exist_ok=True)

# IMPORTANT: abi.txt should be JSON, because Python should json.load it later
(Path(build / "abi.txt")).write_text(json.dumps(mgr["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode.txt")).write_text(mgr["evm"]["bytecode"]["object"], encoding="utf-8")

(Path(build / "abi_model.txt")).write_text(json.dumps(mdl["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_model.txt")).write_text(mdl["evm"]["bytecode"]["object"], encoding="utf-8")
# Write ABI as a Python variable file
abi_py_file = build / "abi_model.py"

with open(abi_py_file, "w", encoding="utf-8") as f:
    f.write("# Auto-generated OpenFLModel ABI\n")
    # Dump ABI as a JSON string (triple quotes)
    f.write("import json\n\n")
    f.write("OPEN_FL_MODEL_ABI = json.loads('''\n")
    f.write(json.dumps(mdl["abi"], indent=2))  # JSON string inside triple quotes
    f.write("\n''')\n")



print("Artifacts written to build/: abi.txt, OPEN_FL_MODEL_ABI.py, bytecode.txt, abi_model.txt, bytecode_model.txt")
