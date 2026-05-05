# tests/test_foundry.py
import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOUNDRY_DIR = ROOT / "foundry"


@pytest.mark.skipif(sys.platform == "win32", reason="forge runs inside WSL on Windows")
def test_forge_tests_pass():
    result = subprocess.run(
        ["forge", "test", "-vvv"],
        cwd=FOUNDRY_DIR,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        "forge test failed\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )