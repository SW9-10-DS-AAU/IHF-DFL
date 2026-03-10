import pandas as pd
from .loader import RunData

# Wei → ETH columns in the global table
_GLOBAL_WEI_COLS  = ["reward_pool", "punishment_pool"]
# Wei → ETH columns in the users table
_USERS_WEI_COLS   = ["grs", "reward_delta", "contribution_score"]
# ratio → % columns
_GLOBAL_PCT_COLS  = ["global_accuracy"]
_USERS_PCT_COLS   = ["accuracy"]

# Metadata keys to propagate as columns when merging
MERGE_META_KEYS = [
    "dataset",
    "contribution_score_strategy",
    "use_outlier_detection",
    "freerider_start_round",
    "freerider_noise_scale",
    "malicious_start_round",
    "malicious_noise_scale",
    "force_merge_all",
    "number_of_good_contributors",
    "number_of_bad_contributors",
    "number_of_freerider_contributors",
    "number_of_inactive_contributors",
    "minimum_rounds",
]


def normalize_run(run: RunData) -> RunData:
    """
    Return a new RunData with unit-converted DataFrames:
      - Wei columns divided by 1e18 (→ ETH)
      - accuracy columns multiplied by 100 (ratio → %)
      - 'is_baseline' boolean column added (True where round == 0)
    """
    g = run.rounds_global.copy() if not run.rounds_global.empty else pd.DataFrame()
    u = run.rounds_users.copy()  if not run.rounds_users.empty  else pd.DataFrame()
    v = run.votes.copy()         if not run.votes.empty         else pd.DataFrame()
    r = run.receipts.copy()      if not run.receipts.empty      else pd.DataFrame()
    c = run.contributions.copy() if not run.contributions.empty else pd.DataFrame()

    # Global table
    if not g.empty:
        for col in _GLOBAL_WEI_COLS:
            if col in g.columns:
                g[col] = g[col] / 1e18
        for col in _GLOBAL_PCT_COLS:
            if col in g.columns:
                g[col] = g[col] * 100
        if "round" in g.columns:
            g["is_baseline"] = g["round"] == 0

    # Users table
    if not u.empty:
        for col in _USERS_WEI_COLS:
            if col in u.columns:
                u[col] = u[col] / 1e18
        for col in _USERS_PCT_COLS:
            if col in u.columns:
                u[col] = u[col] * 100
        if "round" in u.columns:
            u["is_baseline"] = u["round"] == 0

    # Receipts: gas_used stays as-is (integer gas units)

    return RunData(
        experiment_id=run.experiment_id,
        metadata=run.metadata,
        setup=run.setup,
        rounds_global=g,
        rounds_users=u,
        votes=v,
        receipts=r,
        contributions=c
    )


def normalize_runs(runs: list[RunData]) -> list[RunData]:
    return [normalize_run(r) for r in runs]


def merge_runs(runs: list[RunData]) -> dict[str, pd.DataFrame]:
    """
    Concatenate all runs into four flat DataFrames.
    Each row is tagged with metadata columns from MERGE_META_KEYS.
    Returns {"global": df, "users": df, "votes": df, "receipts": df}.
    """
    global_frames   = []
    users_frames    = []
    votes_frames    = []
    receipts_frames = []
    contributions   = []

    for run in runs:
        meta_row = {k: run.metadata.get(k) for k in MERGE_META_KEYS}
        meta_row["experiment_id"] = run.experiment_id

        def _tag(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            for k, v in meta_row.items():
                df = df.copy()
                df[k] = v
            return df

        if not run.rounds_global.empty:
            global_frames.append(_tag(run.rounds_global))
        if not run.rounds_users.empty:
            users_frames.append(_tag(run.rounds_users))
        if not run.votes.empty:
            votes_frames.append(_tag(run.votes))
        if not run.receipts.empty:
            receipts_frames.append(_tag(run.receipts))
        if not run.contributions.empty:
            contributions.append(_tag(run.contributions))

    def _concat(frames):
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    return {
        "global":   _concat(global_frames),
        "users":    _concat(users_frames),
        "votes":    _concat(votes_frames),
        "receipts": _concat(receipts_frames),
        "contributions": _concat(contributions),
    }
