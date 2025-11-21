import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))

import experiment.experiments as experiments


def test_main_invokes_runner(monkeypatch):
    calls = {}

    from types import SimpleNamespace

    def fake_run(dataset, cfg):
        calls["dataset"] = dataset
        calls["config"] = cfg
        return SimpleNamespace(model=SimpleNamespace(visualize_simulation=lambda *_: None))

    monkeypatch.setattr(experiments.ExperimentRunner, "run_experiment", fake_run)
    monkeypatch.setattr(experiments.ExperimentRunner, "print_transactions", lambda exp: calls.setdefault("printed", True))

    experiments.main()

    assert calls["dataset"] == experiments.DATASET
    assert "printed" in calls
