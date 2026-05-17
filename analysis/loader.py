import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class RunData:
    experiment_id:      str
    metadata:           dict
    setup:              dict
    rounds_global:      pd.DataFrame
    rounds_users:       pd.DataFrame
    votes:              pd.DataFrame
    receipts:           pd.DataFrame
    contributions:      pd.DataFrame
    warnings:           pd.DataFrame
    punishments:        pd.DataFrame
    # New table, but optional for existing RunData(...) call sites.
    # Use default_factory because DataFrame is mutable; a direct pd.DataFrame()
    # default would be shared across instances.
    contributions_mad:  pd.DataFrame = field(default_factory=pd.DataFrame)
    # evaluation_rewards: pd.DataFrame
    # evaluation_votes:   pd.DataFrame


_PARENT_CONTRIB_COLS = ["experiment_id", "round", "user_id", "user_address", "contribution_score"]

_LEGACY_MAD_COLS = [
    "user_mad_avg",
    "current_excluded_values", "current_accepted_values",
    "current_mad_median", "current_mad_value", "current_mad_max_deviation",
    "prev_avg_round_val_after_mad",
    "previous_excluded_values", "previous_accepted_values",
    "previous_mad_median", "previous_mad_value", "previous_mad_max_deviation",
    "dotproduct_outlier_weight_count", "dotproduct_outlier_weight_fraction",
]

_LEGACY_STRATEGY_TO_METRIC = {
    "accuracy_only": "accuracy",
    "loss_only":     "loss",
    "dotproduct":    "dotproduct",
}


def _split_legacy_contributions(contrib: pd.DataFrame, strategy: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Old pickles stored MAD stats inline on the contributions table.
    Split into parent (just the score) + child (MAD detail tagged by metric)."""
    if contrib.empty:
        return contrib, pd.DataFrame()
    parent_cols = [c for c in _PARENT_CONTRIB_COLS if c in contrib.columns]
    parent = contrib[parent_cols].copy()

    mad_cols_present = [c for c in _LEGACY_MAD_COLS if c in contrib.columns]
    if not mad_cols_present:
        return parent, pd.DataFrame()

    key_cols = [c for c in ["experiment_id", "round", "user_id", "user_address"] if c in contrib.columns]
    mad = contrib[key_cols + mad_cols_present].copy()
    # Legacy runs had one inline MAD stream, so infer its measurement from the strategy.
    # New runs log metric directly because accuracy_loss writes both accuracy and loss rows.
    mad.insert(len(key_cols), "metric", _LEGACY_STRATEGY_TO_METRIC.get(strategy))
    return parent, mad


def load_run(path: Path) -> RunData:
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    tables = payload["tables"]
    contributions = tables.get("contributions", pd.DataFrame())
    contributions_mad = tables.get("contributions_mad")
    if contributions_mad is None:
        # Legacy pickle: MAD columns were inline on `contributions`. Split them out.
        strategy = payload["metadata"].get("contribution_score_strategy")
        contributions, contributions_mad = _split_legacy_contributions(contributions, strategy)

    return RunData(
        experiment_id=      payload["experiment_id"],
        metadata=           payload["metadata"],
        setup=              payload.get("setup", {}),
        rounds_global=      tables.get("global",              pd.DataFrame()),
        rounds_users=       tables.get("users",               pd.DataFrame()),
        votes=              tables.get("votes",               pd.DataFrame()),
        receipts=           tables.get("receipts",            pd.DataFrame()),
        contributions=      contributions,
        contributions_mad=  contributions_mad,
        warnings=           tables.get("warnings",            pd.DataFrame()),
        punishments=        tables.get("punishments",         pd.DataFrame()),
        # evaluation_rewards= tables.get("evaluation_rewards",  pd.DataFrame()),
        # evaluation_votes=   tables.get("evaluation_votes",    pd.DataFrame()),
    )


def load_runs(
    root: Path,
    recursive: bool = True,
    prefix: str | None = None,
    experiment_ids: list[str] | None = None,
    contribution_score: str | None = None,
    dataset: str | None = None,
) -> list[RunData]:
    """Recursively walk subdirectories and load all .pkl files.

    Args:
        root:               Root directory to search under.
        prefix:             Optional timestamp prefix (e.g. "26-02-26"). Only .pkl files
                            whose immediate parent folder name starts with this string are
                            loaded. Pass None (default) to load everything.
        experiment_ids:     Optional list of GUIDs to load (e.g. "af165b73-71d2-45b7-b1a4-afcfc07b3af4").
                            Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        contribution_score: Optional contribution score strategy to filter by (e.g. "dotproduct", "accuracy_loss").
                            Matched as a substring of the .pkl filename stem. Pass None (default) to load everything.
        dataset:            Optional dataset to filter by (e.g. "mnist", "cifar.10").
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
        if contribution_score is not None and contribution_score not in pkl_file.stem:
            continue
        if dataset is not None and dataset not in pkl_file.stem:
            continue
        try:
            runs.append(load_run(pkl_file))
        except Exception as e:
            print(f"Warning: could not load {pkl_file}: {e}")
    return runs
