import argparse
import dataclasses
import subprocess
import uuid
from pathlib import Path

import pytest

from experiment.experiment_configuration import ExperimentConfiguration
from experiment.experiment_presets import PRESETS
from experiment.helper import getPath
import experiment.run_grid as run_grid
from experiment.run_grid import GitState, RunSpec, build_arg_parser, expand_preset, plan_runs, read_git_state

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
def test_paper_preset_malicious_participant_starts_in_round_one_with_freerider_noise(preset, dataset, noise):
    specs = expand_preset(preset, dataset)

    assert {s.malicious_round for s in specs} == {1}
    assert {s.malicious_noise for s in specs} == {noise}


def test_four_paper_presets_give_480_runs():
    total = sum(len(expand_preset(p, d)) for p, d, _ in PAPER_PRESETS)

    assert total == 480


def test_expansion_order_is_strategy_then_round_then_run_id():
    specs = expand_preset("mnist_openfl_low_noise", "mnist")

    assert specs[0] == RunSpec("loss_only", True, 1, 0.01, 1, 0.01, "mnist", 1)
    assert specs[1] == RunSpec("loss_only", True, 1, 0.01, 1, 0.01, "mnist", 2)
    assert specs[10] == RunSpec("loss_only", True, 3, 0.01, 1, 0.01, "mnist", 1)
    assert specs[-1] == RunSpec("naive", True, 5, 0.01, 1, 0.01, "mnist", 10)


def test_test_preset_expands_to_three_runs_with_malicious_values():
    specs = expand_preset("test", "mnist")

    assert specs == [
        RunSpec("accuracy_loss", True, 2, 0, 2, 0.1, "mnist", run_id)
        for run_id in (1, 2, 3)
    ]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        expand_preset("no_such_preset", "mnist")


def test_run_name_of_a_paper_run_carries_the_malicious_values(monkeypatch):
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

    assert path.name == "mnist_openfl_low_noise-mnist-loss_only-True-3-0.01-1-0.01-2-{FIXED}.csv"


CLEAN = GitState(commit="abc123", dirty=False)


def _flags(*argv):
    return build_arg_parser().parse_args(list(argv))


def test_planned_run_metadata_records_the_commit():
    runs = plan_runs(_flags(), CLEAN, start_time="t")

    assert [(r.metadata["git_commit"], r.metadata["git_dirty"]) for r in runs] == [("abc123", False)] * 3


def test_dirty_tree_without_allow_dirty_refuses_to_plan_any_run():
    with pytest.raises(SystemExit, match="--allow-dirty"):
        plan_runs(_flags(), GitState(commit="abc123", dirty=True), start_time="t")


def test_allow_dirty_plans_the_runs_and_records_the_dirty_tree():
    runs = plan_runs(_flags("--allow-dirty"), GitState(commit="abc123", dirty=True), start_time="t")

    assert [r.metadata["git_dirty"] for r in runs] == [True] * 3


def _fake_git(diff_returncode):
    def run(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(args, 0, stdout="abc123\n")
        return subprocess.CompletedProcess(args, diff_returncode)
    return run


@pytest.mark.parametrize("diff_returncode,dirty", [(0, False), (1, True)])
def test_read_git_state_reports_commit_and_tracked_changes(monkeypatch, diff_returncode, dirty):
    monkeypatch.setattr(run_grid.subprocess, "run", _fake_git(diff_returncode))

    assert read_git_state(Path(".")) == GitState(commit="abc123", dirty=dirty)


def test_read_git_state_without_git_is_unknown_and_dirty(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(run_grid.subprocess, "run", no_git)

    assert read_git_state(Path(".")) == GitState(commit="unknown", dirty=True)


def test_flags_select_the_preset_and_dataset():
    runs = plan_runs(_flags("--preset", "cifar_openfl_low_noise", "--dataset", "cifar.10"), CLEAN, start_time="t")

    assert len(runs) == 120
    assert {r.config.preset_name for r in runs} == {"cifar_openfl_low_noise"}
    assert {r.metadata["dataset"] for r in runs} == {"cifar.10"}
    assert {r.config.minimum_rounds for r in runs} == {25}


def test_without_flags_the_test_preset_runs_on_mnist():
    runs = plan_runs(_flags(), CLEAN, start_time="t")

    assert [r.spec for r in runs] == expand_preset("test", "mnist")
    assert {r.config.preset_name for r in runs} == {"test"}


def test_planned_config_carries_the_cell_values():
    run = plan_runs(_flags("--preset", "mnist_openfl_high_noise"), CLEAN, start_time="t")[11]

    assert (run.config.contribution_score_strategy, run.config.freerider_start_round,
            run.config.freerider_noise_scale, run.config.malicious_start_round,
            run.config.malicious_noise_scale) == ("loss_only", 3, 0.1, 1, 0.1)
    assert run.metadata["timestamp"] == "t"


def test_read_git_state_when_git_diff_fails_is_unknown_and_dirty(monkeypatch):
    monkeypatch.setattr(run_grid.subprocess, "run", _fake_git(diff_returncode=128))

    assert read_git_state(Path(".")) == GitState(commit="unknown", dirty=True)


def test_plan_refuses_a_preset_whose_malicious_participant_would_never_activate(monkeypatch):
    # Checked while planning, so a batch stops up front instead of failing run by run.
    broken = dataclasses.replace(PRESETS["test"], freerider_start_round=None, freerider_noise_scale=None,
                                 malicious_start_round=None, malicious_noise_scale=None)
    monkeypatch.setitem(PRESETS, "broken", broken)

    with pytest.raises(ValueError, match="malicious_start_round, malicious_noise_scale"):
        plan_runs(argparse.Namespace(preset="broken", dataset="mnist", allow_dirty=False), CLEAN, start_time="t")
