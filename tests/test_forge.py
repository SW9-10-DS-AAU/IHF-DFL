# tests/test_foundry.py
import sys
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOUNDRY_DIR = ROOT / "foundry"
DEPS_DIR = FOUNDRY_DIR / "dependencies"


# - Windows → skipped (forge in WSL)
# - Linux without forge → skipped with a helpful install message
# - Linux with forge, deps missing → auto-runs forge soldeer install then tests
# - Linux with forge, deps present → runs tests directly

def _forge_available() -> bool:
    return shutil.which("forge") is not None


def _deps_installed() -> bool:
    return any(DEPS_DIR.glob("forge-std-*/src/Test.sol"))


@pytest.mark.skipif(
    sys.platform == "win32" or not _forge_available(),
    reason="forge not available — install Foundry (https://getfoundry.sh) or run in WSL on Windows",
)
def test_forge_tests_pass():
    if not _deps_installed():
        print("\nforge dependencies missing — running forge soldeer install...")
        subprocess.run(["forge", "soldeer", "install"], cwd=FOUNDRY_DIR, check=True)

    result = subprocess.run(
        ["forge", "test"],
        cwd=FOUNDRY_DIR,
        text=True,
        capture_output=True,
    )

    # Print just the final summary line (e.g. "29 tests passed, 0 failed, 0 skipped")
    for line in result.stdout.splitlines():
        if "tests passed" in line or "test suites" in line:
            print(f"\n[forge] {line.strip()}")

    assert result.returncode == 0, (
        "forge test failed\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )