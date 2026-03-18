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


def load_runs(directory: Path) -> list[RunData]:
    """Load all .pkl files in a flat directory."""
    directory = Path(directory)
    runs = []
    for pkl_file in sorted(directory.glob("*.pkl")):
        try:
            runs.append(load_run(pkl_file))
        except Exception as e:
            print(f"Warning: could not load {pkl_file}: {e}")
    return runs


def load_runs_recursive(root: Path, prefix: str | None = None) -> list[RunData]:
    """Recursively walk subdirectories and load all .pkl files.

    Args:
        root:   Root directory to search under.
        prefix: Optional timestamp prefix (e.g. "26-02-26"). Only .pkl files
                whose immediate parent folder name starts with this string are
                loaded. Pass None (default) to load everything.
    """
    root = Path(root)
    runs = []
    for pkl_file in sorted(root.rglob("*.pkl")):
        if prefix is not None and not pkl_file.parent.name.startswith(prefix):
            continue
        try:
            runs.append(load_run(pkl_file))
        except Exception as e:
            print(f"Warning: could not load {pkl_file}: {e}")
    return runs
