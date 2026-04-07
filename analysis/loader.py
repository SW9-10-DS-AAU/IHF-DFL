import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RunData:
    experiment_id: str
    metadata:      dict
    setup:         dict
    rounds_global: pd.DataFrame
    rounds_users:  pd.DataFrame
    votes:         pd.DataFrame
    receipts:      pd.DataFrame
    contributions: pd.DataFrame
    warnings:      pd.DataFrame


def load_run(path: Path) -> RunData:
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    tables = payload["tables"]
    return RunData(
        experiment_id=payload["experiment_id"],
        metadata=     payload["metadata"],
        setup=        payload.get("setup", {}),
        rounds_global=tables.get("global",        pd.DataFrame()),
        rounds_users= tables.get("users",         pd.DataFrame()),
        votes=        tables.get("votes",         pd.DataFrame()),
        receipts=     tables.get("receipts",      pd.DataFrame()),
        contributions=tables.get("contributions", pd.DataFrame()),
        warnings=     tables.get("warnings", pd.DataFrame())
    )


def load_runs(
    root: Path,
    recursive: bool = True,
    prefix: str | None = None,
    experiment_ids: list[str] | None = None,
    aggregation_rule: str | None = None,
    contribution_score: str | None = None,
    dataset: str | None = None,
    data_distribution: str | None = None,
    dirichlet_alpha: str | None = None,
) -> list[RunData]:
    """Recursively walk subdirectories and load all .pkl files.

    Args:
        root:               Root directory to search under.
        prefix:             Optional timestamp prefix (e.g. "26-02-26"). Only .pkl files
                            whose immediate parent folder name starts with this string are
                            loaded. Pass None (default) to load everything.
        experiment_ids:     Optional list of GUIDs to load (e.g. "af165b73-71d2-45b7-b1a4-afcfc07b3af4").
                            Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        aggregation_rule:   Optional aggregation strategy to filter by (e.g. "FedAVG", "binary_switch").
                            Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        contribution_score: Optional contribution score strategy to filter by (e.g. "dotproduct", "accuracy_loss").
                            Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        dataset:            Optional dataset to filter by (e.g. "mnist", "cifar.10").
                            Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        data_distribution:  Optional data distribution to filter by (e.g. "random_split", "dirichlet_split").
                               Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        dirichlet_alpha:       Optional dirichlet alpha value to filter by (e.g. "0.5").
                               Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
    """
    root = Path(root)
    runs = []
    glob_fn = root.rglob if recursive else root.glob
    for pkl_file in sorted(glob_fn("*.pkl")):
        if prefix is not None and not pkl_file.parent.name.startswith(prefix):
            continue
        if experiment_ids is not None and not any(guid in pkl_file.stem for guid in experiment_ids):
            continue
        if aggregation_rule is not None and aggregation_rule not in pkl_file.stem:
            continue
        if contribution_score is not None and contribution_score not in pkl_file.stem:
            continue
        if dataset is not None and dataset not in pkl_file.stem:
            continue
        if data_distribution is not None and data_distribution not in pkl_file.stem:
            continue
        if dirichlet_alpha is not None and dirichlet_alpha not in pkl_file.stem:
            continue
        try:
            runs.append(load_run(pkl_file))
        except Exception as e:
            print(f"Warning: could not load {pkl_file}: {e}")
    return runs
