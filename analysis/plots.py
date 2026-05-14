from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis.uuid_extractor import extract_uuid_from_filename

matplotlib.rcParams.update({"figure.dpi": 200})

ROLE_LABELS = {
    "good": "Honest",
    "bad": "Malicious",
    "freerider": "Freerider",
    "inactive": "Inactive",
}

BEHAVIOR_COLORS = {
    "good":      "#2196F3",
    "bad":       "#d62728",
    "freerider": "#9467bd",
    "inactive":  "#7f7f7f",
}

STRATEGY_COLORS = {
    "dotproduct":    "#2196F3",
    "naive":         "#FF9800",
    "accuracy_loss": "#E91E63",
    "accuracy_only": "#4CAF50",
    "loss_only":     "#9C27B0",
}


def plot_accuracy_loss_over_rounds(agg_global: pd.DataFrame) -> plt.Figure:
    """
    Dual-axis line chart: accuracy (left y-axis) + loss (right y-axis)
    with ±1 std shading.

    Expects columns: round, accuracy_mean, accuracy_std, loss_mean, loss_std.
    """
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    rounds = agg_global["round"]

    # Accuracy
    ax1.plot(rounds, agg_global["accuracy_mean"], color="#2196F3",
             linewidth=2, label="Accuracy")
    if "accuracy_std" in agg_global.columns:
        ax1.fill_between(
            rounds,
            agg_global["accuracy_mean"] - agg_global["accuracy_std"],
            agg_global["accuracy_mean"] + agg_global["accuracy_std"],
            alpha=0.2, color="#2196F3",
        )

    # Loss
    ax2.plot(rounds, agg_global["loss_mean"], color="#FF5722",
             linewidth=2, linestyle="--", label="Loss")
    if "loss_std" in agg_global.columns:
        ax2.fill_between(
            rounds,
            agg_global["loss_mean"] - agg_global["loss_std"],
            agg_global["loss_mean"] + agg_global["loss_std"],
            alpha=0.2, color="#FF5722",
        )

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Global Accuracy", color="#2196F3")
    ax2.set_ylabel("Global Loss", color="#FF5722")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig._plot_name = "accuracy_loss_over_rounds"
    fig._uuids = agg_global.attrs.get("experiment_ids", [])
    return fig


def plot_strategy_comparison_lines(agg_by_strategy: pd.DataFrame) -> plt.Figure:
    """
    One line per strategy, mean accuracy over rounds with ±1 std error bands.

    Expects columns: contribution_score_strategy, round, accuracy_mean,
    accuracy_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for strategy, group in agg_by_strategy.groupby("contribution_score_strategy"):
        color = STRATEGY_COLORS.get(strategy, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["accuracy_mean"],
                label=strategy, color=color, linewidth=2)
        if "accuracy_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["accuracy_mean"] - group["accuracy_std"],
                group["accuracy_mean"] + group["accuracy_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig._plot_name = "strategy_comparison_lines"
    fig._uuids = agg_by_strategy.attrs.get("experiment_ids", [])
    return fig


def plot_strategy_comparison_boxplot(agg_final: pd.DataFrame) -> plt.Figure:
    """
    One box per strategy showing final-round accuracy distribution.

    Expects columns: contribution_score_strategy, final_accuracy.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    strategies = sorted(agg_final["contribution_score_strategy"].unique())
    data = [
        agg_final.loc[
            agg_final["contribution_score_strategy"] == s, "final_accuracy"
        ].values
        for s in strategies
    ]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strategies]

    bp = ax.boxplot(data, patch_artist=True, labels=strategies)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Final-Round Accuracy (%)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig._plot_name = "strategy_comparison_boxplot"
    fig._uuids = agg_final.attrs.get("experiment_ids", [])
    return fig


def plot_grs_by_role(agg_grs: pd.DataFrame) -> plt.Figure:
    """
    One line per role (eventual user type), GRS over rounds with ±1 std shading.

    Expects columns: role, round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_grs.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "grs_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["grs_mean"] - group["grs_std"],
                group["grs_mean"] + group["grs_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig._plot_name = "grs_by_role"
    fig._uuids = agg_grs.attrs.get("experiment_ids", [])
    return fig


def plot_contribution_score_by_role(agg_scores: pd.DataFrame) -> plt.Figure:
    """
    One line per role (eventual user type), contribution score over rounds
    with ±1 std shading.

    Expects columns: role, round, score_mean, score_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_scores.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["score_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "score_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["score_mean"] - group["score_std"],
                group["score_mean"] + group["score_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Contribution Score")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig._plot_name = "contribution_score_by_role"
    fig._uuids = agg_scores.attrs.get("experiment_ids", [])
    return fig


def plot_grs_by_role_relative(agg_grs: pd.DataFrame) -> plt.Figure:
    """
    One line per role, GRS over rounds-since-activation with ±1 std shading.
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_grs.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("relative_round")
        ax.plot(group["relative_round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "grs_std" in group.columns:
            ax.fill_between(
                group["relative_round"],
                group["grs_mean"] - group["grs_std"],
                group["grs_mean"] + group["grs_std"],
                alpha=0.15, color=color,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Activation")
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig._plot_name = "grs_by_role_relative"
    fig._uuids = agg_grs.attrs.get("experiment_ids", [])
    return fig


def plot_contribution_score_by_role_relative(agg_scores: pd.DataFrame) -> plt.Figure:
    """
    One line per role, contribution score over rounds-since-activation with ±1 std shading.
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, score_mean, score_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_scores.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("relative_round")
        ax.plot(group["relative_round"], group["score_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "score_std" in group.columns:
            ax.fill_between(
                group["relative_round"],
                group["score_mean"] - group["score_std"],
                group["score_mean"] + group["score_std"],
                alpha=0.15, color=color,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Activation")
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Contribution Score")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig._plot_name = "contribution_score_by_role_relative"
    fig._uuids = agg_scores.attrs.get("experiment_ids", [])
    return fig


def plot_grs_by_user(
    grs_users: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    One line per user, GRS over rounds.

    metadata: optional full metadata DataFrame. If provided, vertical dashed lines
    are drawn at malicious_start_round / freerider_start_round for roles present
    in the data, extracted automatically from the matching experiment row.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for (user_id, behavior), group in grs_users.groupby(["user_id", "role"]):
        ax.plot(group["round"], group["grs"], label=f"User {user_id} ({ROLE_LABELS[behavior]})", alpha=0.5) # alpha: 50% transparency, so overlapping lines show through each other

    if metadata is not None:
        experiment_id = grs_users["experiment_id"].iloc[0]
        meta = metadata[metadata["experiment_id"] == experiment_id].iloc[0]
        roles_in_data = grs_users["role"].unique()
        for role, col in (("bad", "malicious_start_round"), ("freerider", "freerider_start_round")):
            if role not in roles_in_data or col not in meta.index:
                continue
            color = BEHAVIOR_COLORS.get(role, "black")
            ax.axvline(
                int(meta[col]),
                linestyle="--",
                color=color,
                linewidth=2,
                alpha=1.0,
                label=f"{ROLE_LABELS.get(role, role)} activation",
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Users", loc="lower left")
    ax.grid(True, alpha=0.3) # alpha: makes the grid subtle/faint so it doesn't compete with the data
    fig.tight_layout()
    fig._plot_name = "grs_by_user"
    fig._uuids = grs_users.attrs.get("experiment_ids", [])
    return fig


def plot_gas_cost_by_tx_type(agg_gas: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart of mean gas used per transaction type, one bar group per
    tx_type and one bar per contribution_score_strategy, with ±1 std error bars.

    Expects columns: tx_type, contribution_score_strategy, gas_mean, gas_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    tx_types = sorted(agg_gas["tx_type"].unique())
    strategies = sorted(agg_gas["contribution_score_strategy"].unique())
    n_tx = len(tx_types)
    n_strategies = len(strategies)
    width = 0.8 / n_strategies
    x = range(n_tx)

    for i, strategy in enumerate(strategies):
        group = agg_gas[agg_gas["contribution_score_strategy"] == strategy]
        means = []
        stds = []
        for tx in tx_types:
            row = group[group["tx_type"] == tx]
            means.append(row["gas_mean"].iloc[0] if not row.empty else float("nan"))
            stds.append(row["gas_std"].iloc[0] if not row.empty else 0)

        xpos = [xi - 0.4 + i * width + width / 2 for xi in x]
        color = STRATEGY_COLORS.get(strategy, "#607c8a")
        ax.bar(xpos, means, width, yerr=stds, capsize=4,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.8,
               label=strategy)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tx_types, rotation=10, ha="right")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Mean Gas Used")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig._plot_name = "gas_cost_by_tx_type"
    fig._uuids = agg_gas.attrs.get("experiment_ids", [])
    return fig



def plot_round_kicked_by_strategy(
    agg_kicked: pd.DataFrame,
    title: str = "Effectiveness of Strategies in Removing Dishonest Participants",
    max_rounds: int | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: for each contribution score strategy, show at which
    round each user role was disqualified (lower = removed sooner = better).
    Asymmetric error bars show min/max range across runs.

    Inspired by kickedGraph() in scripts/processData.py.

    Expects columns: contribution_score_strategy, role,
                     mean_round_kicked, low_err, high_err.
    """
    if agg_kicked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No disqualified users", ha="center", va="center", transform=ax.transAxes)
        fig._plot_name = "round_kicked_by_strategy"
        fig._uuids = agg_kicked.attrs.get("experiment_ids", [])
        return fig

    strategies = sorted(agg_kicked["contribution_score_strategy"].unique())
    roles = sorted(agg_kicked["role"].unique())

    n_strategies = len(strategies)
    n_roles = len(roles)
    x = range(n_strategies)
    width = 0.8 / n_roles

    fig, ax = plt.subplots(figsize=(max(7, n_strategies * 1.8), 5))

    for role_idx, role in enumerate(roles):
        role_data = agg_kicked[agg_kicked["role"] == role]
        color = BEHAVIOR_COLORS.get(role, "#888888")

        means   = []
        low_err = []
        high_err = []
        missing = []

        for strategy in strategies:
            row = role_data[role_data["contribution_score_strategy"] == strategy]
            if row.empty:
                means.append(float("nan"))
                low_err.append(0)
                high_err.append(0)
                missing.append(True)
            else:
                means.append(row["mean_round_kicked"].iloc[0])
                low_err.append(row["low_err"].iloc[0])
                high_err.append(row["high_err"].iloc[0])
                missing.append(False)

        xpos = [xi - 0.4 + role_idx * width + width / 2 for xi in x]

        bar_means = [m if not missing[i] else float("nan") for i, m in enumerate(means)]
        show_err = any(l != 0 or h != 0 for l, h in zip(low_err, high_err))

        ax.bar(
            xpos,
            bar_means,
            width,
            yerr=[low_err, high_err] if show_err else None,
            capsize=4,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            label=ROLE_LABELS[role],
        )

        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (max_rounds or 1)
        for xi, is_missing in zip(xpos, missing):
            if is_missing:
                ax.text(
                    xi, y_top * 0.02, "N/A",
                    ha="center", va="bottom",
                    fontsize=8, color="gray", rotation=90,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=10, ha="right")
    ax.set_ylabel("Round Kicked (lower = removed sooner)")
    ax.set_title(title)
    ax.legend(title="Role")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig._plot_name = "round_kicked_by_strategy"
    fig._uuids = agg_kicked.attrs.get("experiment_ids", [])
    return fig


def _next_graph_id(directory: Path) -> str:
    existing = [p.name for p in directory.glob("*.svg")]
    ids = []
    for name in existing:
        part = name.split("-")[0]
        if part.isdigit():
            ids.append(int(part))
    next_id = max(ids) + 1 if ids else 1
    return f"{next_id:03d}"


def delete_figure(directory: str | Path, graph_id: str) -> None:
    directory = Path(directory)
    matches = list(directory.glob(f"{graph_id}-*.svg"))
    if not matches:
        raise FileNotFoundError(f"No SVG with graph_id '{graph_id}' in {directory}")

    for svg in matches:
        svg.unlink()

    mappings_path = directory / "mappings.txt"

    if mappings_path.exists():
        lines = mappings_path.read_text().splitlines(keepends=True)
        kept = [l for l in lines if not l.startswith(f"{graph_id}:") and l.strip()]
        if kept:
            mappings_path.write_text("".join(kept))
        else:
            mappings_path.unlink()


def save_figure(fig: plt.Figure, base_dir, experiment_name=None, suffix: str = "", dpi: int = 300):
    plot_name = getattr(fig, "_plot_name", "figure")
    uuids = getattr(fig, "_uuids", [])
    directory = Path(base_dir) / experiment_name if experiment_name is not None else Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)

    graph_id = _next_graph_id(directory)
    stem = f"{graph_id}-{plot_name}{f'-{suffix}' if suffix else ''}"
    fig.savefig(directory / f"{stem}.svg", dpi=dpi, bbox_inches="tight")

    if uuids:
        with open(directory / "mappings.txt", "a") as f:
            for uid in uuids:
                try:
                    uid = extract_uuid_from_filename(uid)
                except ValueError:
                    pass  # already a bare UUID or unrecognised format — write as-is
                f.write(f"{graph_id}: {uid}\n")
