import pandas as pd


def _with_meta(df: pd.DataFrame, metadata: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Join only the needed metadata *cols* onto *df* via experiment_id."""
    return df.merge(metadata[["experiment_id"] + cols], on="experiment_id", how="left")


def agg_global_accuracy_loss_by_round(merged_global: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std of global_accuracy and global_loss grouped by round.

    Returns DataFrame with columns: round, accuracy_mean, accuracy_std,
    loss_mean, loss_std.
    """
    agg = (
        merged_global
        .groupby("round")
        .agg(
            accuracy_mean=("objective_global_accuracy", "mean"),
            accuracy_std= ("objective_global_accuracy", "std"),
            loss_mean=    ("objective_global_loss",     "mean"),
            loss_std=     ("objective_global_loss",     "std"),
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


# Displays their behaviour. We know if they are malicious or freerider, that they are going to switch. So not displaying role.
def agg_grs_by_behavior(merged_users: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std of GRS grouped by [behavior, round], excluding round 0.

    Returns DataFrame with columns: behavior, round, grs_mean, grs_std.
    """
    # df = merged_users[merged_users["round"] > 0]
    df = merged_users
    agg = (
        df
        .groupby(["behavior", "round"])
        .agg(
            grs_mean=("grs", "mean"),
            grs_std= ("grs", "std"),
        )
        .reset_index()
    )
    return agg


def grs_by_user(merged_users: pd.DataFrame) -> pd.DataFrame:
    # No aggregation needed, we just select the desired data.
    # we want one GRS per user/per round

    # TODO: Check that it works. Mal. and FR's GRS decline after round two. But they first switch in round 3. Does the switching logic work? Are we logging correctly, and not round zero?

    # df = merged_users[merged_users["round"] > 0]
    df = merged_users

    # Role: Just fetch from first value on user

    return df[["grs", "user_id", "role", "round"]].sort_values("round")


# All, Only positives,
# Accuracy_only, loss_only

# Take global accuracy (What it actually is) over rounds, show for each aggregation strategy.


def global_acc_by_aggregation_strategy(acc_over_agg: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    # df = acc_over_agg[acc_over_agg["round"] > 0]
    df = acc_over_agg
    df = _with_meta(df, metadata, ["contribution_score_strategy"])
    # TODO: Change to aggregation_strategy when implemented
    return df[['round', 'objective_global_accuracy', 'contribution_score_strategy']].sort_values(["contribution_score_strategy", "round"])


def global_loss_by_aggregation_strategy(loss_over_agg: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    # df = loss_over_agg[loss_over_agg["round"] > 0]
    df = loss_over_agg
    df = _with_meta(df, metadata, ["contribution_score_strategy"])
    # TODO: Change to aggregation_strategy when implemented
    return df[['round', 'objective_global_loss', 'contribution_score_strategy']].sort_values(["contribution_score_strategy", "round"])




def agg_contribution_score_by_behavior(merged_users: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std of contribution_score grouped by [behavior, round].

    Returns DataFrame with columns: behavior, round, score_mean, score_std.
    """
    agg = (
        merged_users
        .groupby(["behavior", "round"])
        .agg(
            score_mean=("contribution_score", "mean"),
            score_std= ("contribution_score", "std"),
        )
        .reset_index()
    )
    return agg


def agg_gas_used_by_tx_type(merged_receipts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    df = _with_meta(merged_receipts, metadata, ["contribution_score_strategy"])
    agg = (
        df
        .groupby(["tx_type"])
        .agg(
            gas_mean=("gas_used", "mean"),
            gas_std= ("gas_used", "std"),
            strategy= ("contribution_score_strategy", "first")
        )
        .reset_index() # Used for making columns instead of DF indexes.
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

    Parameters
    ----------
    merged_users : merged["users"] DataFrame (after normalize_runs + merge_runs)
    metadata : merged["metadata"] DataFrame with experiment config columns
    freerider_start_round : optional int — filter to experiments with this
        freerider_start_round value (matches graph_one_one=1, one_two=3, one_three=5)

    Returns DataFrame with columns:
        contribution_score_strategy, role,
        mean_round_kicked, low_err (mean−min), high_err (max−mean)
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
