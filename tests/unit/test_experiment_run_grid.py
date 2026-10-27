import uuid
from pathlib import Path

import pytest

from experiment.experiment_configuration import ExperimentConfiguration
from experiment.helper import getPath
from experiment.run_grid import RunSpec, expand_preset

STRATEGIES = ["loss_only", "accuracy_only", "accuracy_loss", "naive"]

# Preset name, its dataset, and its free-rider noise (the only value that differs between presets).
PAPER_PRESETS = [
    ("mnist_openfl_low_noise", "mnist", 0.01),
    ("mnist_openfl_high_noise", "mnist", 0.1),
    ("cifar_openfl_low_noise", "cifar.10", 0.01),
    ("cifar_openfl_high_noise", "cifar.10", 0.1),
]


@pytest.mark.parametrize("preset,dataset,noise", PAPER_PRESETS)
def test_paper_preset_expands_to_its_full_grid(preset, dataset, noise):
    specs = expand_preset(preset, dataset)

    assert len(specs) == 120
    assert len(set(specs)) == 120
    assert {s.strategy for s in specs} == set(STRATEGIES)
    assert {s.outlier_detection for s in specs} == {True}
    assert {s.freerider_round for s in specs} == {1, 3, 5}
    assert {s.freerider_noise for s in specs} == {noise}
    assert {s.run_id for s in specs} == set(range(1, 11))
    assert {s.dataset for s in specs} == {dataset}


@pytest.mark.parametrize("preset,dataset,noise", PAPER_PRESETS)
def test_paper_preset_keeps_malicious_values_none(preset, dataset, noise):
    specs = expand_preset(preset, dataset)

    assert {s.malicious_round for s in specs} == {None}
    assert {s.malicious_noise for s in specs} == {None}


def test_four_paper_presets_give_480_runs():
    total = sum(len(expand_preset(p, d)) for p, d, _ in PAPER_PRESETS)

    assert total == 480


def test_expansion_order_is_strategy_then_round_then_run_id():
    specs = expand_preset("mnist_openfl_low_noise", "mnist")

    assert specs[0] == RunSpec("loss_only", True, 1, 0.01, None, None, "mnist", 1)
    assert specs[1] == RunSpec("loss_only", True, 1, 0.01, None, None, "mnist", 2)
    assert specs[10] == RunSpec("loss_only", True, 3, 0.01, None, None, "mnist", 1)
    assert specs[-1] == RunSpec("naive", True, 5, 0.01, None, None, "mnist", 10)


def test_test_preset_expands_to_three_runs_with_malicious_values():
    specs = expand_preset("test", "mnist")

    assert specs == [
        RunSpec("accuracy_loss", True, 2, 0, 2, 0.1, "mnist", run_id)
        for run_id in (1, 2, 3)
    ]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        expand_preset("no_such_preset", "mnist")


def test_run_name_is_unchanged_for_a_paper_run(monkeypatch):
    monkeypatch.setattr(uuid, "uuid4", lambda: "FIXED")
    spec = expand_preset("mnist_openfl_low_noise", "mnist")[11]
    config = ExperimentConfiguration(preset="mnist_openfl_low_noise")
    config.contribution_score_strategy = spec.strategy
    config.use_outlier_detection = spec.outlier_detection
    config.freerider_start_round = spec.freerider_round
    config.freerider_noise_scale = spec.freerider_noise
    config.malicious_start_round = spec.malicious_round
    config.malicious_noise_scale = spec.malicious_noise

    path = getPath(config, "t", spec.dataset, "mnist_openfl_low_noise", Path("out"), run_id=spec.run_id)

    assert path.name == "mnist_openfl_low_noise-mnist-loss_only-True-3-0.01-None-None-2-{FIXED}.csv"
