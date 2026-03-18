import numpy as np
import pandas as pd
from .loader import RunData

# Wei → ETH columns in the global table
_GLOBAL_WEI_COLS  = ["reward_pool", "punishment_pool"]
# Wei → ETH columns in the users table
_USERS_WEI_COLS   = ["grs", "reward_delta", "contribution_score"]
# ratio → % columns
_GLOBAL_ACC_COLS  = ["global_accuracy"]
_USERS_ACC_COLS   = ["accuracy"]

_GLOBAL_LOSS_COLS  = ["global_loss"]
_USERS_LOSS_COLS   = ["loss"]




# Metadata keys stored in the "metadata" lookup table returned by merge_runs.
# Data tables only carry experiment_id; join on it when you need config values.
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
    w = run.warnings.copy()      if not run.warnings.empty      else pd.DataFrame()

    # Global table
    if not g.empty:
        for col in _GLOBAL_WEI_COLS:
            if col in g.columns:
                g[col] = g[col] / 1e18
        if "round" in g.columns:
            g["is_baseline"] = g["round"] == 0

    # Users table
    if not u.empty:
        for col in _USERS_WEI_COLS:
            if col in u.columns:
                u[col] = u[col] / 1e18 # TODO: Why not working for contribution_score?

        for col in _USERS_ACC_COLS:
            if col in u.columns:
                u[col] = u[col] / 10000

        for col in _USERS_LOSS_COLS:
            if col in u.columns:
                u[col] = u[col] / 100

        # if "round" in u.columns:
        #     u["is_baseline"] = u["round"] == 0

    # Votes table — flag whether the voted accuracy was excluded as an outlier.
    # Only current_excluded_values is pulled from contributions; it's dropped after use.
    if not v.empty and not c.empty:
        contrib_res = c[['round', 'user_id', 'current_excluded_values']]

        v = v.merge(contrib_res, left_on=['round', 'receiver_id'],
                    right_on=['round', 'user_id'], how='left') \
             .drop(columns='user_id')

        v['is_outlier'] = v.apply(
            lambda row: any(np.isclose(row['votes_accuracy'], val)
                            for val in row['current_excluded_values'])
                        if isinstance(row['current_excluded_values'], list) else False,
            axis=1
        )

        v = v.drop(columns=['current_excluded_values'])

    #   - Dropped experiment_id from merge keys — it doesn't exist yet at this stage (it's added later in merge_runs).
    #   - inner join — matches the original snippet; votes without a corresponding contribution row are dropped. If you
    #   want to preserve all votes, switch to how='left' (then is_outlier will be NaN for unmatched rows).
    #   - Column narrowing — vote_res/contrib_res select only the needed columns up front so v doesn't accumulate
    #   duplicate columns from c.


    # Receipts: gas_used stays as-is (integer gas units)

    return RunData(
        experiment_id=run.experiment_id,
        metadata=run.metadata,
        setup=run.setup,
        rounds_global=g,
        rounds_users=u,
        votes=v,
        receipts=r,
        contributions=c,
        warnings=w
    )


def normalize_runs(runs: list[RunData]) -> list[RunData]:
    return [normalize_run(r) for r in runs]


def merge_runs(runs: list[RunData]) -> dict[str, pd.DataFrame]:
    """
    Concatenate all runs into flat DataFrames.
    Each data table is tagged with only ``experiment_id``.
    A separate ``"metadata"`` DataFrame (one row per run) holds all config
    columns from MERGE_META_KEYS — join on ``experiment_id`` when needed.

    Returns {"metadata": df, "global": df, "users": df, "votes": df,
             "receipts": df, "contributions": df, "warnings": df}.
    """
    metadata_rows   = []
    global_frames   = []
    users_frames    = []
    votes_frames    = []
    receipts_frames = []
    contributions   = []
    warnings        = []

    for run in runs:
        eid = run.experiment_id

        # One metadata row per run
        meta_row = {k: run.metadata.get(k) for k in MERGE_META_KEYS}
        meta_row["experiment_id"] = eid
        metadata_rows.append(meta_row)

        def _tag(df: pd.DataFrame) -> pd.DataFrame:
            """Stamp only experiment_id onto *df*."""
            if df.empty:
                return df
            df = df.copy()
            df["experiment_id"] = eid
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
        if not run.warnings.empty:
            warnings.append(_tag(run.warnings))

    def _concat(frames):
        if not frames:
            return pd.DataFrame()
        # Drop all-NA columns from each frame before concatenating.
        # When runs differ in available columns, pandas fills missing ones with NaN.
        # Concatenating frames that contain all-NA columns triggers a FutureWarning
        # about dtype inference changing in a future pandas version.
        return pd.concat([f.dropna(axis=1, how="all") for f in frames], ignore_index=True)

    return {
        "metadata": pd.DataFrame(metadata_rows),
        "global":   _concat(global_frames),
        "users":    _concat(users_frames),
        "votes":    _concat(votes_frames),
        "receipts": _concat(receipts_frames),
        "contributions": _concat(contributions),
        "warnings": _concat(warnings),
    }
