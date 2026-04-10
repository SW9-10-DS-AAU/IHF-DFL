import numpy as np
import pandas as pd
from .loader import RunData
from .aggregations import _require_consistent_rounds

# Wei → ETH columns
_GLOBAL_WEI_COLS  = ["reward_pool", "punishment_pool"]
_USERS_WEI_COLS   = ["grs", "reward_delta", "round_reputation_assigned"]

# ratio → % columns
_GLOBAL_ACC_COLS  = ["objective_global_accuracy"]
_USERS_ACC_COLS   = ["subjective_personal_accuracy", "subjective_global_accuracy"]

# vote_accuracy/vote_prev_accuracy are stored as int 0..10000 (accuracy * 100 * scalar(100); divide by 100 → %
# vote_loss/vote_prev_loss are stored as loss * scalar(100); divide by 100 → actual loss
# NOTE: these must be converted AFTER the is_outlier merge, which compares vote_accuracy
# against current_excluded_values (both in 0..10000 scale) — converting before would break it.
_VOTES_ACC_COLS  = ["vote_accuracy", "vote_prev_accuracy"]
_VOTES_LOSS_COLS = ["vote_loss", "vote_prev_loss"]


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
    "aggregation_rule",
    "force_merge_all",
    "number_of_good_contributors",
    "number_of_bad_contributors",
    "number_of_freerider_contributors",
    "number_of_inactive_contributors",
    "minimum_rounds",
    "data_distribution",
    "dirichlet_alpha",
]


def normalize_run(run: RunData, make_readable: bool = True) -> RunData:
    """
    Return a new RunData with unit-converted DataFrames:
      - Wei columns divided by 1e18 (→ ETH)
      - accuracy columns multiplied by 100 (ratio → %)
      - 'is_baseline' boolean column added (True where round == 0)

    make_readable: if False, skip all unit conversions and return raw values.
                   The outlier/is_baseline/vote merge logic still runs regardless.
    """
    g = run.rounds_global.copy() if not run.rounds_global.empty else pd.DataFrame()
    u = run.rounds_users.copy()  if not run.rounds_users.empty  else pd.DataFrame()
    v = run.votes.copy()         if not run.votes.empty         else pd.DataFrame()
    r = run.receipts.copy()      if not run.receipts.empty      else pd.DataFrame()
    c = run.contributions.copy() if not run.contributions.empty else pd.DataFrame()
    w = run.warnings.copy()      if not run.warnings.empty      else pd.DataFrame()


    wei_divisor      = 1e18  # Wei → ETH
    acc_multiplier   = 100   # float 0..1 → %
    vote_acc_divisor = 100   # int 0..10000 → % (= / 10000 * 100)
    vote_loss_unscaler  = 100   # vote loss stored as actual_loss * 100; restore by dividing

    if make_readable:

        # Global table
        if not g.empty:
                for col in _GLOBAL_WEI_COLS:
                    if col in g.columns:
                        g[col] = g[col] / wei_divisor
                for col in _GLOBAL_ACC_COLS:
                    if col in g.columns:
                        g[col] = g[col] * acc_multiplier

        # Users table
        if not u.empty:
            for col in _USERS_WEI_COLS:
                if col in u.columns:
                    u[col] = u[col] / wei_divisor

            for col in _USERS_ACC_COLS:
                if col in u.columns:
                    u[col] = u[col] * acc_multiplier


    # Votes table — flag whether the voted accuracy was excluded as an outlier.
    # Only current_excluded_values is pulled from contributions; it's dropped after use.
    if not v.empty and not c.empty:
        contrib_res = c[['round', 'user_id', 'current_excluded_values']]

        v = v.merge(contrib_res, left_on=['round', 'receiver_id'],
                    right_on=['round', 'user_id'], how='left') \
             .drop(columns='user_id')

        v['is_outlier'] = v.apply(
            lambda row: any(np.isclose(row['vote_accuracy'], val)
                            for val in row['current_excluded_values'])
                        if isinstance(row['current_excluded_values'], list) else False,
            axis=1
        )

        v = v.drop(columns=['current_excluded_values'])

    # Vote list conversions — must run after is_outlier merge (see note above)
    if make_readable:
        if not v.empty:
            for col in _VOTES_ACC_COLS:
                if col in v.columns:
                    v[col] = v[col] / vote_acc_divisor
            for col in _VOTES_LOSS_COLS:
                if col in v.columns:
                    v[col] = v[col] / vote_loss_unscaler

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


def normalize_runs(runs: list[RunData], make_readable: bool = True) -> list[RunData]:
    return [normalize_run(r, make_readable=make_readable) for r in runs]


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

    result = {
        "metadata": pd.DataFrame(metadata_rows),
        "global":   _concat(global_frames),
        "users":    _concat(users_frames),
        "votes":    _concat(votes_frames),
        "receipts": _concat(receipts_frames),
        "contributions": _concat(contributions),
        "warnings": _concat(warnings),
    }
