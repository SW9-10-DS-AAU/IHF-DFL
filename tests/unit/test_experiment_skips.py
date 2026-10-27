import importlib
import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "experiment"


@pytest.fixture
def experiments(monkeypatch):
    # The driver parses the command line on import, and runs with experiment/ on sys.path.
    monkeypatch.syspath_prepend(str(EXPERIMENT_DIR))
    monkeypatch.setattr(sys, "argv", ["experiments.py"])
    module = importlib.import_module("experiment.experiments")
    monkeypatch.setattr(module, "skips", [])
    return module


def _recorded_skip(module, **overrides):
    values = dict(preset="mnist_openfl_low_noise", dataset="mnist", strategy="loss_only", outlier_detection=True,
                  freerider_activation_round=1, freerider_noise=0.01,
                  malicious_activation_round=1, malicious_noise=0.01, run_id=3)
    return module.Skip(**{**values, **overrides})


def _scan(module, monkeypatch, tmp_path, filename):
    folder = tmp_path / "28-10-26--10_00_00"
    folder.mkdir()
    (folder / filename).touch()
    monkeypatch.setattr(module, "RESULTDATAFOLDER", tmp_path)
    monkeypatch.setattr(module, "choose_from_list", lambda options, *args: list(options))
    module.parseSkips()


def test_run_recorded_with_malicious_values_is_recognised_as_done(experiments, monkeypatch, tmp_path):
    name = "mnist_openfl_low_noise-mnist-loss_only-True-1-0.01-1-0.01-3-{1b4e28ba-2fa1-11d2-883f-0016d3cca427}.pkl"

    _scan(experiments, monkeypatch, tmp_path, name)

    assert experiments.shouldSkip(_recorded_skip(experiments))


def test_run_recorded_with_other_malicious_noise_is_not_recognised_as_done(experiments, monkeypatch, tmp_path):
    name = "mnist_openfl_low_noise-mnist-loss_only-True-1-0.01-1-0.01-3-{1b4e28ba-2fa1-11d2-883f-0016d3cca427}.pkl"

    _scan(experiments, monkeypatch, tmp_path, name)

    assert not experiments.shouldSkip(_recorded_skip(experiments, malicious_noise=0.1))
