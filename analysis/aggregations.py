import pandas as pd


def _require_nonempty(df: pd.DataFrame, name: str) -> None:
    """Raise a clear error if *df* has no rows, rather than a cryptic KeyError later."""
    if df.empty:
        raise ValueError(
            f"'{name}' is empty — no experiment data was loaded. "
            "Check that load_runs() found .pkl files and that merge_runs() produced non-empty tables."
        )


def _with_meta(df: pd.DataFrame, metadata: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Join only the needed metadata *cols* onto *df* via experiment_id."""
    return df.merge(metadata[["experiment_id"] + cols], on="experiment_id", how="left")


def agg_global_accuracy_loss_by_round(merged_global: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of global accuracy and loss by round.

    The naive single-stage groupby("round") treats every row — regardless of
    which experiment it came from — as an independent observation. This is
    wrong for two reasons:
      1. Each experiment contributes exactly one (accuracy, loss) value per
         round, so pooling all rows computes an unweighted mean that gives more
         influence to experiments that somehow produced more rows for the same
         round (e.g. logging quirks).
      2. The resulting std mixes within-experiment noise with across-experiment
         variance, making the shaded band meaningless.

    Two-stage fix:
      Stage 1 — collapse each experiment to a single (accuracy, loss) per round
                by taking its mean. After this step there is exactly one row per
                (experiment_id, round).
      Stage 2 — across those per-experiment values, compute mean and std per
                round. The std now purely reflects run-to-run variance, which
                is what the shaded band in the plot should show.

    Returns DataFrame with columns: round, accuracy_mean, accuracy_std,
    loss_mean, loss_std.
    """
    _require_nonempty(merged_global, "merged_global")
    per_experiment = (
        merged_global
        .groupby(["experiment_id", "round"])
        .agg(
            accuracy=("objective_global_accuracy", "mean"),
            loss=    ("objective_global_loss",     "mean"),
        )
        .reset_index()
    )
    agg = (
        per_experiment
        .groupby("round")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std= ("accuracy", "std"),
            loss_mean=    ("loss",     "mean"),
            loss_std=     ("loss",     "std"),
        )
        .reset_index()
    )
    return agg


# def agg_accuracy_by_strategy(merged_global: pd.DataFrame) -> pd.DataFrame:
#     """
#     Mean and std of global_accuracy grouped by [contribution_score_strategy, round].
#
#     Returns DataFrame with columns: contribution_score_strategy, round,
#     accuracy_mean, accuracy_std.
#     """
#     agg = (
#         merged_global
#         .groupby(["contribution_score_strategy", "round"])
#         .agg(
#             accuracy_mean=("global_accuracy", "mean"),
#             accuracy_std= ("global_accuracy", "std"),
#         )
#         .reset_index()
#     )
#     return agg


# def agg_final_round_accuracy_by_strategy(merged_global: pd.DataFrame) -> pd.DataFrame:
#     """
#     Final-round accuracy per run, one row per (experiment_id, strategy).
#     Suitable for box plots.
#
#     Returns DataFrame with columns: contribution_score_strategy, experiment_id,
#     final_accuracy.
#     """
#     # Pick the maximum round per experiment
#     max_rounds = (
#         merged_global
#         .groupby("experiment_id")["round"]
#         .max()
#         .reset_index()
#         .rename(columns={"round": "max_round"})
#     )
#     merged = merged_global.merge(max_rounds, on="experiment_id")
#     final = merged[merged["round"] == merged["max_round"]].copy()
#     final = final.rename(columns={"global_accuracy": "final_accuracy"})
#     return final[["contribution_score_strategy", "experiment_id", "final_accuracy"]]


def _require_consistent_activation(merged_users: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Raise if loaded experiments have different activation rounds."""
    eids = merged_users["experiment_id"].unique()
    sub = metadata[metadata["experiment_id"].isin(eids)]
    for col in ("freerider_start_round", "malicious_start_round"):
        if col not in sub.columns:
            continue
        vals = sub[col].dropna().unique()
        if len(vals) > 1:
            raise ValueError(
                f"Experiments have different '{col}' values: {sorted(vals)}. "
                "Use agg_grs_by_role_relative() to aggregate across different activation configs."
            )


def agg_grs_by_role(merged_users: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of GRS by role and round.

    Groups by role (futureAttitude) — the user's eventual identity — rather
    than behavior (current attitude). This shows the full lifetime trajectory
    of each user type: honest users stay high, bad/freerider users decline once
    the protocol detects them, regardless of when they switched.

    Stage 1: mean GRS per (experiment_id, role, round).
    Stage 2: mean and std of those per-experiment means across runs.

    Returns DataFrame with columns: role, round, grs_mean, grs_std.
    """
    _require_nonempty(merged_users, "merged_users")
    _require_consistent_activation(merged_users, metadata)
    per_experiment = (
        merged_users
        .groupby(["experiment_id", "role", "round"])
        .agg(grs=("grs", "mean"))
        .reset_index()
    )
    agg = (
        per_experiment
        .groupby(["role", "round"])
        .agg(
            grs_mean=("grs", "mean"),
            grs_std= ("grs", "std"),
        )
        .reset_index()
    )
    return agg


def agg_grs_by_role_relative(merged_users: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of GRS by role using rounds-since-activation as x-axis.

    relative_round = round - activation_round:
      bad       → malicious_start_round
      freerider → freerider_start_round
      good/inactive → 0 (never switch, serve as baseline)

    x=0 is always the switch moment. Negative x = pre-switch, positive = post-switch.
    Works correctly across experiments with different activation rounds.

    Stage 1: mean GRS per (experiment_id, role, relative_round).
    Stage 2: mean and std across experiments.

    Returns columns: role, relative_round, grs_mean, grs_std.
    """
    _require_nonempty(merged_users, "merged_users")
    cols = [c for c in ("freerider_start_round", "malicious_start_round") if c in metadata.columns]
    df = _with_meta(merged_users, metadata, cols)
    freerider_col = "freerider_start_round" if "freerider_start_round" in df.columns else None
    malicious_col = "malicious_start_round" if "malicious_start_round" in df.columns else None

    def _activation(row):
        if row["role"] == "bad" and malicious_col:
            return row[malicious_col]
        if row["role"] == "freerider" and freerider_col:
            return row[freerider_col]
        return 0

    df["activation_round"] = df.apply(_activation, axis=1)
    df["relative_round"] = df["round"] - df["activation_round"]
    per_experiment = (
        df.groupby(["experiment_id", "role", "relative_round"])
        .agg(grs=("grs", "mean")).reset_index()
    )
    return (
        per_experiment.groupby(["role", "relative_round"])
        .agg(grs_mean=("grs", "mean"), grs_std=("grs", "std"))
        .reset_index()
    )


def grs_by_user(merged_users: pd.DataFrame) -> pd.DataFrame:
    """
    Raw GRS per user per round. Intended for single-experiment inspection only.
    Include experiment_id so user identity is unambiguous when multiple
    experiments are loaded.
    """

    _require_nonempty(merged_users, "merged_users")
    n_experiments = merged_users["experiment_id"].nunique()
    if n_experiments > 1:
        raise ValueError(
            f"grs_by_user() received data from {n_experiments} experiments. "
            "This function is intended for single-experiment inspection only. "
            "Use agg_grs_by_behavior() for multi-experiment data."
        )

    # No aggregation needed, we just select the desired data.
    # we want one GRS per user/per round

    df = merged_users

    # Role: Just fetch from first value on user

    return df[["grs", "user_id", "role", "round"]].sort_values("round")


def global_acc_by_aggregation_strategy(acc_over_agg: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of global accuracy by aggregation rule and round.

    Stage 1: mean accuracy per (experiment_id, aggregation_rule, round).
    Stage 2: mean and std of those per-experiment means across runs.

    Returns DataFrame with columns: aggregation_rule, round,
    accuracy_mean, accuracy_std.
    """
    _require_nonempty(acc_over_agg, "acc_over_agg")
    df = _with_meta(acc_over_agg, metadata, ["aggregation_rule"])
    per_experiment = (
        df
        .groupby(["experiment_id", "aggregation_rule", "round"])
        .agg(accuracy=("objective_global_accuracy", "mean"))
        .reset_index()
    )
    agg = (
        per_experiment
        .groupby(["aggregation_rule", "round"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std= ("accuracy", "std"),
        )
        .reset_index()
    )
    return agg


def global_loss_by_aggregation_strategy(loss_over_agg: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of global loss by aggregation rule and round.

    Stage 1: mean loss per (experiment_id, aggregation_rule, round).
    Stage 2: mean and std of those per-experiment means across runs.

    Returns DataFrame with columns: aggregation_rule, round,
    loss_mean, loss_std.
    """
    _require_nonempty(loss_over_agg, "loss_over_agg")
    df = _with_meta(loss_over_agg, metadata, ["aggregation_rule"])
    per_experiment = (
        df
        .groupby(["experiment_id", "aggregation_rule", "round"])
        .agg(loss=("objective_global_loss", "mean"))
        .reset_index()
    )
    agg = (
        per_experiment
        .groupby(["aggregation_rule", "round"])
        .agg(
            loss_mean=("loss", "mean"),
            loss_std= ("loss", "std"),
        )
        .reset_index()
    )
    return agg




def agg_contribution_score_by_role(merged_users: pd.DataFrame, merged_contributions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of contribution_score by role and round.

    Groups by role (futureAttitude) so the full lifetime of each user type is
    visible. Users excluded from scoring (roundrep < 0, fl_challenge.py:1292)
    have no contributions entry; left-joining from users fills those rounds
    with 0 — excluded = zero contribution.

    Stage 1: mean score per (experiment_id, role, round).
    Stage 2: mean and std of those per-experiment means across runs.

    Returns DataFrame with columns: role, round, score_mean, score_std.

    "how does the protocol treat users who are destined to be bad, across their full lifetime?"


    """
    _require_nonempty(merged_users, "merged_users")
    _require_nonempty(merged_contributions, "merged_contributions")
    _require_consistent_activation(merged_users, metadata)

    role_lookup = merged_users[["experiment_id", "round", "user_id", "role"]].drop_duplicates()
    scores = merged_contributions[["experiment_id", "round", "user_id", "contribution_score"]]
    # Left join from users so excluded users still appear with score=0.
    df = role_lookup.merge(scores, on=["experiment_id", "round", "user_id"], how="left")
    df["contribution_score"] = df["contribution_score"].fillna(0)

    per_experiment = (
        df
        .groupby(["experiment_id", "role", "round"])
        .agg(contribution_score=("contribution_score", "mean"))
        .reset_index()
    )
    agg = (
        per_experiment
        .groupby(["role", "round"])
        .agg(
            score_mean=("contribution_score", "mean"),
            score_std= ("contribution_score", "std"),
        )
        .reset_index()
    )
    return agg


def agg_contribution_score_by_role_relative(
    merged_users: pd.DataFrame,
    merged_contributions: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Two-stage aggregation of contribution_score by role using rounds-since-activation as x-axis.

    relative_round = round - activation_round:
      bad       → malicious_start_round
      freerider → freerider_start_round
      good/inactive → 0 (never switch, serve as baseline)

    Users excluded from scoring appear with score=0 (left-join from users).

    Stage 1: mean score per (experiment_id, role, relative_round).
    Stage 2: mean and std across experiments.

    Returns columns: role, relative_round, score_mean, score_std.
    """
    _require_nonempty(merged_users, "merged_users")
    _require_nonempty(merged_contributions, "merged_contributions")

    role_lookup = merged_users[["experiment_id", "round", "user_id", "role"]].drop_duplicates()
    scores = merged_contributions[["experiment_id", "round", "user_id", "contribution_score"]]
    df = role_lookup.merge(scores, on=["experiment_id", "round", "user_id"], how="left")
    df["contribution_score"] = df["contribution_score"].fillna(0)

    cols = [c for c in ("freerider_start_round", "malicious_start_round") if c in metadata.columns]
    df = _with_meta(df, metadata, cols)
    freerider_col = "freerider_start_round" if "freerider_start_round" in df.columns else None
    malicious_col = "malicious_start_round" if "malicious_start_round" in df.columns else None

    def _activation(row):
        if row["role"] == "bad" and malicious_col:
            return row[malicious_col]
        if row["role"] == "freerider" and freerider_col:
            return row[freerider_col]
        return 0

    # Exclude round 0 (baseline): no contribution scores exist there, so the left-join
    # fills them with 0. Those zeros land at negative relative rounds (e.g. round 0 with
    # freerider_start_round=3 → relative_round=-3), dragging down the pre-activation average
    # and making scores appear to drop before activation. Fixed 2026-03-19.
    # df = df[df["round"] > 0]

    df["activation_round"] = df.apply(_activation, axis=1)
    df["relative_round"] = df["round"] - df["activation_round"]
    per_experiment = (
        df.groupby(["experiment_id", "role", "relative_round"])
        .agg(contribution_score=("contribution_score", "mean"))
        .reset_index()
    )
    return (
        per_experiment.groupby(["role", "relative_round"])
        .agg(score_mean=("contribution_score", "mean"), score_std=("contribution_score", "std"))
        .reset_index()
    )


def agg_gas_used_by_tx_type(merged_receipts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage aggregation of gas used per (tx_type, contribution_score_strategy).

    Stage 1: mean gas per (experiment_id, tx_type, contribution_score_strategy).
    Stage 2: mean and std of those per-experiment means across runs.

    Returns DataFrame with columns: tx_type, contribution_score_strategy,
    gas_mean, gas_std.
    """
    _require_nonempty(merged_receipts, "merged_receipts")
    df = _with_meta(merged_receipts, metadata, ["contribution_score_strategy"])
    per_experiment = (
        df
        .groupby(["experiment_id", "tx_type", "contribution_score_strategy"])
        .agg(gas=("gas_used", "mean"))
        .reset_index()
    )
    agg = (
        per_experiment
        .groupby(["tx_type", "contribution_score_strategy"])
        .agg(
            gas_mean=("gas", "mean"),
            gas_std= ("gas", "std"),
        )
        .reset_index()
    )
    return agg




def agg_round_kicked_by_strategy(
    merged_users: pd.DataFrame,
    metadata: pd.DataFrame,
    freerider_start_round: int | None = None,
) -> pd.DataFrame:
    """
    For each (contribution_score_strategy, role), compute the mean round at
    which participants were disqualified, plus asymmetric min/max error bars.

    Mirrors the kickedGraph() chart from scripts/processData.py.
    """

    # TODO: Check this works

    df = _with_meta(merged_users, metadata, ["contribution_score_strategy", "freerider_start_round"])

    if freerider_start_round is not None:
        df = df[df["freerider_start_round"] == freerider_start_round]

    # Only rows where a user was disqualified
    disq = df[df["state"] == "disqualified"]

    if disq.empty:
        return pd.DataFrame(columns=[
            "contribution_score_strategy", "role",
            "mean_round_kicked", "low_err", "high_err",
        ])

    # First round kicked per (experiment, user)
    first_kicked = (
        disq
        .groupby(["experiment_id", "user_id"])
        .agg(
            round_kicked=("round", "min"),
            role=("role", "first"),
            contribution_score_strategy=("contribution_score_strategy", "first"),
        )
        .reset_index()
    )

    # Aggregate across runs: mean, min, max per (strategy, role)
    agg = (
        first_kicked
        .groupby(["contribution_score_strategy", "role"])
        .agg(
            mean_round_kicked=("round_kicked", "mean"),
            min_round_kicked= ("round_kicked", "min"),
            max_round_kicked= ("round_kicked", "max"),
        )
        .reset_index()
    )

    # Asymmetric error bars matching original kickedGraph style
    agg["low_err"]  = agg["mean_round_kicked"] - agg["min_round_kicked"]
    agg["high_err"] = agg["max_round_kicked"]  - agg["mean_round_kicked"]

    return agg[["contribution_score_strategy", "role",
                "mean_round_kicked", "low_err", "high_err"]]
