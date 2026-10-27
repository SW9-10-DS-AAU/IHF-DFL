"""Repro tests against the `main` worktree. Each test asserts the behavior the paper describes; a failure confirms a finding.

Run: python -m pytest audit/test_audit_repro.py -v -p no:cacheprovider
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MAIN = Path(__file__).resolve().parents[2] / "IHF-DFL-main-audit"
if not MAIN.exists():
    # The tests read the audited commit from a sibling worktree; without it there is nothing to check.
    pytest.skip(f"audit worktree not found at {MAIN}", allow_module_level=True)
sys.path.insert(0, str(MAIN / "src"))
sys.path.insert(0, str(MAIN))

from experiment.experiment_configuration import ExperimentConfiguration  # noqa: E402
from experiment.experiment_presets import PRESETS  # noqa: E402
from contracts.contribution import normalize_contribution_scores_new  # noqa: E402
import ml.attacks as attacks  # noqa: E402

PAPER_PRESETS = ["mnist_openfl_low_noise", "mnist_openfl_high_noise", "cifar_openfl_low_noise", "cifar_openfl_high_noise"]


@pytest.mark.parametrize("preset", PAPER_PRESETS)
def test_paper_presets_configure_a_malicious_start_round(preset: str) -> None:
    # experiments.py iterates PRESETS[preset].malicious_start_round directly ([None] when unset)
    # and assigns it to config.malicious_start_round, bypassing the fallback in ExperimentConfiguration.
    assert PRESETS[preset].malicious_start_round is not None


def test_malicious_participant_switches_to_bad_by_round_1() -> None:
    # PytorchModel.add_participant("bad") passes _attitudeSwitch = self.malicious_start_round (None here).
    user = SimpleNamespace(attitude="good", futureAttitude="bad", attitudeSwitch=None, address="0xabc", color=None)
    pm = SimpleNamespace(participants=[user], round=1)
    for r in range(1, 26):
        pm.round = r
        attacks.update_users_attitude(pm)
    assert user.attitude == "bad", "malicious participant never activates: attitudeSwitch=None never equals pm.round"


def test_fallback_exists_but_is_overridden() -> None:
    cfg = ExperimentConfiguration("cifar_openfl_low_noise")
    assert cfg.malicious_start_round == 1  # fallback to freerider_start_round works here...
    cfg.malicious_start_round = None  # ...but experiments.py:180 overwrites it with the per-run None
    assert cfg.malicious_start_round is not None


def test_single_metric_normalization_is_signed_sum_eq7() -> None:
    # Paper Eq. 7: n_i = d_i / sum_k d_k. Two improvers, two worseners, net negative sum.
    acc_prev = 100.0
    accs = [101.0, 102.0, 90.0, 95.0]  # d = [1, 2, -10, -5], sum = -12
    eq7 = [d / -12 for d in (1.0, 2.0, -10.0, -5.0)]
    got = normalize_contribution_scores_new(accs, acc_prev, "accuracy")
    assert got == pytest.approx(eq7), f"code returns {got}, Eq. 7 gives {eq7}"
