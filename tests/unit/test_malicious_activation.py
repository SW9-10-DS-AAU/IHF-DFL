"""
#83: the malicious participant never activated in the paper presets (all 480 FAIRFL runs).
The presets leave malicious_* as None, which falls back to the run's free-rider round and noise;
missing values fail fast.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "experiment"))

import experiment.experiments as experiments
import experiment.experiment_runner as runner
from experiment.experiment_presets import PRESETS, ExperimentPreset
from ml.attacks import update_users_attitude

PAPER_PRESETS = {
    "mnist_openfl_low_noise": "mnist",
    "mnist_openfl_high_noise": "mnist",
    "cifar_openfl_low_noise": "cifar.10",
    "cifar_openfl_high_noise": "cifar.10",
}
MALICIOUS_PRESETS = sorted(
    name for name, p in PRESETS.items()
    if isinstance(p, ExperimentPreset) and p.contribution_score_strategy and p.number_of_bad_contributors > 0
)


# ---- per-run configuration -------------------------------------------------

@pytest.mark.parametrize("name,dataset", PAPER_PRESETS.items())
def test_paper_presets_malicious_follows_the_runs_freerider_round_and_noise(name, dataset):
    configs = experiments.build_run_configs(name, [dataset])
    for config, _ in configs:
        assert config.malicious_start_round == config.freerider_start_round
        assert config.malicious_noise_scale == config.freerider_noise_scale
    # per run, not fixed to the preset's first free-rider value
    assert {c.malicious_start_round for c, _ in configs} == set(PRESETS[name].freerider_start_round)


@pytest.mark.parametrize("name", MALICIOUS_PRESETS)
def test_no_run_with_a_malicious_participant_has_unresolved_attack_params(name):
    for config, _ in experiments.build_run_configs(name, ["mnist"]):
        assert config.malicious_start_round is not None
        assert config.malicious_noise_scale is not None


def test_explicit_preset_values_are_kept():
    preset = PRESETS["test"]
    for config, _ in experiments.build_run_configs("test", ["mnist"]):
        assert config.malicious_start_round in preset.malicious_start_round
        assert config.malicious_noise_scale in preset.malicious_noise_scale


# ---- attitude switch -------------------------------------------------------

@pytest.mark.parametrize("start_round", PRESETS["mnist_openfl_low_noise"].freerider_start_round)
def test_bad_participant_turns_bad_in_its_start_round(start_round):
    config = next(c for c, _ in experiments.build_run_configs("mnist_openfl_low_noise", ["mnist"])
                  if c.freerider_start_round == start_round)
    user = SimpleNamespace(address="0xbad0000000000000000", attitude="good", futureAttitude="bad",
                           attitudeSwitch=config.malicious_start_round, color=None)
    pm = SimpleNamespace(round=start_round - 1, participants=[user])

    update_users_attitude(pm)
    assert user.attitude == "good"  # not before its round

    pm.round = start_round
    update_users_attitude(pm)
    assert user.attitude == "bad"


# ---- fail fast -------------------------------------------------------------

def _config(**overrides):
    values = dict(number_of_bad_contributors=1, malicious_start_round=1, malicious_noise_scale=0.1)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("missing", ["malicious_start_round", "malicious_noise_scale"])
def test_runner_refuses_malicious_participant_without_attack_params(missing):
    with pytest.raises(ValueError, match="never activate"):
        runner.run_experiment("mnist", _config(**{missing: None}), 1)


def test_runner_check_ignores_presets_without_malicious_participant():
    runner.check_attack_params(_config(number_of_bad_contributors=0, malicious_start_round=None,
                                       malicious_noise_scale=None))


# ---- --skipFolder must not reuse results from the broken runs --------------

OLD_BROKEN = "mnist_openfl_low_noise-mnist-loss_only-True-1-0.01-None-None-1-{0b8f7f02-1111-2222-3333-444455556666}.pkl"
NEW_FIXED = "mnist_openfl_low_noise-mnist-loss_only-True-1-0.01-1-0.01-1-{0b8f7f02-1111-2222-3333-444455556666}.pkl"


def _first_run_skip():
    config, run_id = experiments.build_run_configs("mnist_openfl_low_noise", ["mnist"])[0]
    return experiments.skip_for("mnist_openfl_low_noise", config, run_id)


def test_old_none_result_file_does_not_count_as_a_finished_fixed_run(monkeypatch):
    monkeypatch.setattr(experiments, "skips", [experiments.skip_from_filename(OLD_BROKEN)])
    assert not experiments.shouldSkip(_first_run_skip())


def test_fixed_result_file_still_counts_as_finished(monkeypatch):
    monkeypatch.setattr(experiments, "skips", [experiments.skip_from_filename(NEW_FIXED)])
    assert experiments.shouldSkip(_first_run_skip())


def test_none_of_the_120_old_paper_files_match_a_fixed_run(monkeypatch):
    runs = experiments.build_run_configs("mnist_openfl_low_noise", ["mnist"])
    old_files = [
        "-".join(map(str, ["mnist_openfl_low_noise", "mnist", c.contribution_score_strategy, c.use_outlier_detection,
                           c.freerider_start_round, c.freerider_noise_scale, None, None, r]))
        + "-{0b8f7f02-1111-2222-3333-444455556666}.pkl"
        for c, r in runs
    ]
    monkeypatch.setattr(experiments, "skips", [experiments.skip_from_filename(f) for f in old_files])
    assert not any(experiments.shouldSkip(experiments.skip_for("mnist_openfl_low_noise", c, r)) for c, r in runs)
