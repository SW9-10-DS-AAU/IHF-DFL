"""Tests for analysis/loader.py — load_run, load_runs, load_runs_recursive."""
import pickle

import pandas as pd
import pytest

from analysis.loader import RunData, load_run, load_runs, load_runs_recursive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pkl(path, experiment_id, metadata=None):
    """Write a minimal valid .pkl that load_run can parse."""
    payload = {
        "experiment_id": experiment_id,
        "metadata": metadata or {},
        "setup": {},
        "tables": {
            "global":        pd.DataFrame(),
            "users":         pd.DataFrame(),
            "votes":         pd.DataFrame(),
            "receipts":      pd.DataFrame(),
            "contributions": pd.DataFrame(),
            "warnings":      pd.DataFrame(),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _nested_tree(tmp_path):
    """
    tmp_path/
      26-03-15/
        mnist-accuracy_loss-FedAVG-run1.pkl
        cifar-dotproduct-binary_switch-run2.pkl
      26-04-01/
        mnist-dotproduct-FedAVG-run3.pkl
    """
    d1 = tmp_path / "26-03-15"
    d1.mkdir()
    _write_pkl(d1 / "mnist-accuracy_loss-FedAVG-run1.pkl", "mnist-accuracy_loss-FedAVG-run1")
    _write_pkl(d1 / "cifar-dotproduct-binary_switch-run2.pkl", "cifar-dotproduct-binary_switch-run2")

    d2 = tmp_path / "26-04-01"
    d2.mkdir()
    _write_pkl(d2 / "mnist-dotproduct-FedAVG-run3.pkl", "mnist-dotproduct-FedAVG-run3")


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------

def test_load_run_returns_rundata(tmp_path):
    pkl = tmp_path / "run.pkl"
    _write_pkl(pkl, "run")
    assert isinstance(load_run(pkl), RunData)


def test_load_run_experiment_id_preserved(tmp_path):
    stem = "mnist-accuracy_loss-FedAVG-run1"
    pkl = tmp_path / f"{stem}.pkl"
    _write_pkl(pkl, stem)
    assert load_run(pkl).experiment_id == stem


def test_load_run_tables_are_dataframes(tmp_path):
    pkl = tmp_path / "run.pkl"
    _write_pkl(pkl, "run")
    run = load_run(pkl)
    assert isinstance(run.rounds_global, pd.DataFrame)
    assert isinstance(run.rounds_users, pd.DataFrame)
    assert isinstance(run.votes, pd.DataFrame)
    assert isinstance(run.receipts, pd.DataFrame)
    assert isinstance(run.contributions, pd.DataFrame)
    assert isinstance(run.warnings, pd.DataFrame)


def test_load_run_metadata_preserved(tmp_path):
    meta = {"dataset": "mnist", "aggregation_rule": "FedAVG"}
    pkl = tmp_path / "run.pkl"
    _write_pkl(pkl, "run", metadata=meta)
    run = load_run(pkl)
    assert run.metadata["dataset"] == "mnist"
    assert run.metadata["aggregation_rule"] == "FedAVG"


def test_load_run_accepts_path_string(tmp_path):
    pkl = tmp_path / "run.pkl"
    _write_pkl(pkl, "run")
    run = load_run(str(pkl))
    assert isinstance(run, RunData)


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------

def test_load_runs_finds_all_pkls(tmp_path):
    for i in range(3):
        _write_pkl(tmp_path / f"run-{i}.pkl", f"run-{i}")
    runs = load_runs(tmp_path)
    assert len(runs) == 3


def test_load_runs_ignores_non_pkl(tmp_path):
    _write_pkl(tmp_path / "run.pkl", "run")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "data.csv").write_text("also ignored")
    runs = load_runs(tmp_path)
    assert len(runs) == 1


def test_load_runs_empty_directory(tmp_path):
    assert load_runs(tmp_path) == []


def test_load_runs_skips_corrupt_files(tmp_path, capsys):
    _write_pkl(tmp_path / "good.pkl", "good")
    (tmp_path / "corrupt.pkl").write_bytes(b"not valid pickle data!!!")
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert "Warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# load_runs_recursive — filter logic
# ---------------------------------------------------------------------------

def test_load_runs_recursive_no_filter_loads_all(tmp_path):
    _nested_tree(tmp_path)
    assert len(load_runs_recursive(tmp_path)) == 3


def test_load_runs_recursive_prefix_filter(tmp_path):
    _nested_tree(tmp_path)
    runs = load_runs_recursive(tmp_path, prefix="26-03-15")
    assert len(runs) == 2
    assert all("26-03-15" in str(r.experiment_id) or True for r in runs)


def test_load_runs_recursive_dataset_filter(tmp_path):
    _nested_tree(tmp_path)
    runs = load_runs_recursive(tmp_path, dataset="mnist")
    assert len(runs) == 2


def test_load_runs_recursive_aggregation_rule_filter(tmp_path):
    _nested_tree(tmp_path)
    runs = load_runs_recursive(tmp_path, aggregation_rule="FedAVG")
    assert len(runs) == 2


def test_load_runs_recursive_contribution_score_filter(tmp_path):
    _nested_tree(tmp_path)
    runs = load_runs_recursive(tmp_path, contribution_score="dotproduct")
    assert len(runs) == 2


def test_load_runs_recursive_combined_filters_narrow(tmp_path):
    _nested_tree(tmp_path)
    # Only cifar-dotproduct-binary_switch matches cifar + binary_switch
    runs = load_runs_recursive(tmp_path, dataset="cifar", aggregation_rule="binary_switch")
    assert len(runs) == 1


def test_load_runs_recursive_no_match_returns_empty(tmp_path):
    _nested_tree(tmp_path)
    assert load_runs_recursive(tmp_path, dataset="imagenet") == []


def test_load_runs_recursive_prefix_excludes_other_dirs(tmp_path):
    _nested_tree(tmp_path)
    runs = load_runs_recursive(tmp_path, prefix="26-04-01")
    assert len(runs) == 1
