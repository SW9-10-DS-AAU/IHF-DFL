"""
Characterization tests for the batch driver's per-run configurations (experiment/experiments.py).

They pin what the driver produced before build_run_configs() was extracted (#97): the product
order of the grid, the per-run values (malicious values resolved per #83 since that fix),
the run names used for the result files, and the Skip entries used by --skipFolder.
"""
import sys
from itertools import product
from pathlib import Path

import pytest

# experiments.py imports `selector` as a top-level module (experiment/ is on sys.path when run as a script)
sys.path.append(str(Path(__file__).resolve().parents[2] / "experiment"))

import experiment.experiments as experiments
from experiment.experiment_presets import PRESETS
from experiment.helper import getPath

PAPER_PRESETS = {
    "mnist_openfl_low_noise": "mnist",
    "mnist_openfl_high_noise": "mnist",
    "cifar_openfl_low_noise": "cifar.10",
    "cifar_openfl_high_noise": "cifar.10",
}


def _expected_runs(name, dataset):
    """The driver's grid as it was written inline in main() before the extraction."""
    datasets = dataset if isinstance(dataset, list) else [dataset]
    p = PRESETS[name]
    as_list = lambda v: v if v is not None else [None]
    runs = list(range(1, (p.number_of_runs or 1) + 1)) or [1]
    grid = product(
        p.contribution_score_strategy,
        p.use_outlier_detection,
        as_list(p.freerider_start_round),
        as_list(p.freerider_noise_scale),
        as_list(p.malicious_start_round),
        as_list(p.malicious_noise_scale),
        datasets,
        runs,
    )
    # #83: a None malicious value takes this run's free-rider value
    return [(s, od, fr, fn, fr if mr is None else mr, fn if mn is None else mn, ds, r)
            for s, od, fr, fn, mr, mn, ds, r in grid]


def _values(config, run_id):
    return (config.contribution_score_strategy, config.use_outlier_detection,
            config.freerider_start_round, config.freerider_noise_scale,
            config.malicious_start_round, config.malicious_noise_scale,
            config.dataset, run_id)


def _run_name(name, config, run_id):
    stem = getPath(config, "t", config.dataset, name, Path("."), run_id=run_id).stem
    return stem.rsplit("-{", 1)[0]  # drop the random uuid


@pytest.mark.parametrize("name,dataset", PAPER_PRESETS.items())
def test_paper_preset_yields_120_runs_in_driver_order(name, dataset):
    runs = experiments.build_run_configs(name, [dataset])
    assert len(runs) == 120
    assert [_values(c, r) for c, r in runs] == _expected_runs(name, dataset)


def test_paper_presets_yield_480_runs_in_total():
    assert sum(len(experiments.build_run_configs(n, [d])) for n, d in PAPER_PRESETS.items()) == 480


@pytest.mark.parametrize("name,dataset", PAPER_PRESETS.items())
def test_run_configs_carry_preset_name_and_defaults(name, dataset):
    for config, _ in experiments.build_run_configs(name, [dataset]):
        assert config.preset_name == name
        assert config.dataset == dataset
        assert config.number_of_bad_contributors == PRESETS[name].number_of_bad_contributors
        assert config.punish_factor == PRESETS["default"].punish_factor


@pytest.mark.parametrize("name,dataset", PAPER_PRESETS.items())
def test_run_names_are_unchanged(name, dataset):
    names = [_run_name(name, c, r) for c, r in experiments.build_run_configs(name, [dataset])]
    expected = ["-".join(map(str, [name, ds, s, od, fr, fn, mr, mn, r]))
                for s, od, fr, fn, mr, mn, ds, r in _expected_runs(name, dataset)]
    assert names == expected


def test_paper_preset_run_name_example():
    config, run_id = experiments.build_run_configs("mnist_openfl_low_noise", ["mnist"])[0]
    assert _run_name("mnist_openfl_low_noise", config, run_id) == \
        "mnist_openfl_low_noise-mnist-loss_only-True-1-0.01-1-0.01-1"


@pytest.mark.parametrize("name,dataset", PAPER_PRESETS.items())
def test_skip_entries_are_unchanged(name, dataset):
    skips = [experiments.skip_for(name, c, r) for c, r in experiments.build_run_configs(name, [dataset])]
    expected = [experiments.Skip(name, ds, s, od, fr, fn, mr, mn, r)
                for s, od, fr, fn, mr, mn, ds, r in _expected_runs(name, dataset)]
    assert skips == expected


def test_multiple_datasets_stay_inside_the_product():
    runs = experiments.build_run_configs("test", ["mnist", "cifar.10"])
    assert [_values(c, r) for c, r in runs] == _expected_runs("test", ["mnist", "cifar.10"])


def test_main_runs_every_config_in_order(monkeypatch, tmp_path):
    calls = []

    class FakeWriter:
        def __init__(self, path, *a):
            self.path = path

        def finish(self):
            pass

    class FakeLogger:
        def __init__(self, experiment_id, metadata):
            self.experiment_id = experiment_id

        def save(self, path):
            calls[-1]["saved"] = path.name

    def fake_run(dataset, config, run_id, writer, logger):
        calls.append({"values": _values(config, run_id), "dataset": dataset, "name": writer.path.stem.rsplit("-{", 1)[0]})

    monkeypatch.setattr(experiments, "preset", "mnist_openfl_low_noise")
    monkeypatch.setattr(experiments, "datasets", ["mnist"])
    monkeypatch.setattr(experiments, "RESULTDATAFOLDER", tmp_path)
    monkeypatch.setattr(experiments, "AsyncWriter", FakeWriter)
    monkeypatch.setattr(experiments, "ExperimentLogger", FakeLogger)
    monkeypatch.setattr(experiments.ExperimentRunner, "run_experiment", fake_run)
    monkeypatch.setattr(experiments, "progress_bar", lambda *a: None)
    monkeypatch.setattr(experiments.args, "skipFolder", None)

    experiments.main("tester")

    expected = _expected_runs("mnist_openfl_low_noise", "mnist")
    assert [c["values"] for c in calls] == expected
    assert all(c["dataset"] == "mnist" and c["saved"].endswith(".pkl") for c in calls)
    assert calls[0]["name"] == "mnist_openfl_low_noise-mnist-loss_only-True-1-0.01-1-0.01-1"
