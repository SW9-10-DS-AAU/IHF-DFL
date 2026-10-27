"""Rebuild Fig. 3-8 of the FAIRFL paper from the stored runs.

Usage:
    python -m analysis.paper_figures [RUNS_DIR] [OUT_DIR]
Defaults: data/runs/ and data/output_graphs/paper/.

Definitions (these reproduce the paper):
  - Permanent removal: `state` is `disqualified` in some round; the removal round
    is the first such round.
  - Post-activation delay: removal round minus free-rider activation round.
  - Round-one gas: sum of `gas_used` over receipts with `round == 1`, averaged over runs.
  - Within-round variance: population variance (ddof=0) of the contribution scores
    in a round, averaged over the rounds of a run.
Bootstrap intervals resample runs with a fixed seed, so output is identical on every run.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.loader import load_runs
from analysis.transform import merge_runs, normalize_runs

DEFAULT_RUNS_DIR = Path("data/runs")
DEFAULT_OUT_DIR = Path("data/output_graphs/paper")

BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 2000

STRATEGIES = ["naive", "accuracy_only", "loss_only", "accuracy_loss"]
STRATEGY_LABELS = {
    "naive": "Naive", "accuracy_only": "Accuracy",
    "loss_only": "Loss", "accuracy_loss": "Accuracy+Loss",
}
DATASETS = ["mnist", "cifar.10"]
DATASET_LABELS = {"mnist": "MNIST", "cifar.10": "CIFAR-10"}
ROLES = ["good", "bad", "freerider"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def removal_rounds(users: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment_id, user_id): role, removed (bool), removed_round (NaN if kept)."""
    disqualified = users[users["state"] == "disqualified"]
    first_removal = (disqualified.groupby(["experiment_id", "user_id"])["round"].min()
                     .rename("removed_round"))
    roles = users.groupby(["experiment_id", "user_id"])["role"].first()
    out = roles.to_frame().join(first_removal).reset_index()
    out["removed"] = out["removed_round"].notna()
    return out


def post_activation_delay(removed_round: float, activation_round: float) -> float:
    return removed_round - activation_round


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    # Clamp to [0, p] and [p, 1]: at p = 0 or 1 floating-point error would otherwise put the
    # bound a hair on the wrong side of p and give matplotlib a negative error bar.
    return min(p, max(0.0, float(centre - half))), max(p, min(1.0, float(centre + half)))


def bootstrap_ci(values, level: float = 0.95, seed: int = BOOTSTRAP_SEED,
                 n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def mean_within_round_variance(contributions: pd.DataFrame) -> pd.DataFrame:
    """Per run: contribution-score variance within each round, averaged over rounds."""
    per_round = (contributions.groupby(["experiment_id", "round"])["contribution_score"]
                 .var(ddof=0))
    return per_round.groupby("experiment_id").mean().rename("variance").reset_index()


def round_one_gas_per_run(receipts: pd.DataFrame, by_tx_type: bool = False) -> pd.DataFrame:
    first = receipts[receipts["round"] == 1]
    keys = ["experiment_id", "tx_type"] if by_tx_type else ["experiment_id"]
    return first.groupby(keys)["gas_used"].sum().reset_index()


# ---------------------------------------------------------------------------
# Tables (one per figure)
# ---------------------------------------------------------------------------

def _with_config(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    cols = ["experiment_id", "dataset", "contribution_score_strategy",
            "freerider_start_round", "freerider_noise_scale"]
    return df.merge(metadata[cols], on="experiment_id").rename(
        columns={"contribution_score_strategy": "strategy",
                 "freerider_start_round": "activation_round",
                 "freerider_noise_scale": "noise"})


def freerider_removal_table(users: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Fig. 3: removals (of n free-riders) and median post-activation delay per cell."""
    removal = _with_config(removal_rounds(users), metadata)
    fr = removal[removal["role"] == "freerider"].copy()
    fr["delay"] = [post_activation_delay(r, a)
                   for r, a in zip(fr["removed_round"], fr["activation_round"])]
    return (fr.groupby(["dataset", "noise", "strategy", "activation_round"])
            .agg(removed=("removed", "sum"), total=("removed", "size"),
                 median_delay=("delay", "median"))
            .reset_index())


def removal_by_role_table(users: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Fig. 4: removals per dataset x strategy x role with Wilson 95% interval."""
    removal = _with_config(removal_rounds(users), metadata)
    t = (removal.groupby(["dataset", "strategy", "role"])
         .agg(removed=("removed", "sum"), total=("removed", "size")).reset_index())
    bounds = [wilson_interval(int(k), int(n)) for k, n in zip(t["removed"], t["total"])]
    t["lo"], t["hi"] = zip(*bounds)
    t["rate"] = t["removed"] / t["total"]
    return t


def mean_grs_by_role_table(users: pd.DataFrame, metadata: pd.DataFrame,
                           noise: float = 0.01, activation_round: int = 3) -> pd.DataFrame:
    """Fig. 5: mean GRS (ETH) per role over rounds, bootstrap over runs."""
    u = _with_config(users, metadata)
    u = u[(u["noise"] == noise) & (u["activation_round"] == activation_round)]
    per_run = (u.groupby(["dataset", "strategy", "role", "round", "experiment_id"])["grs"]
               .mean().reset_index())
    return _bootstrap_by_group(per_run, ["dataset", "strategy", "role", "round"], "grs")


def global_accuracy_table(global_df: pd.DataFrame, metadata: pd.DataFrame,
                          by_noise: bool = False) -> pd.DataFrame:
    """Fig. 7: global accuracy (%) over rounds, bootstrap over runs.

    Runs of all activation rounds are pooled. The paper draws one line per noise level
    (`by_noise=True`); without it the noise levels are pooled too.
    """
    g = _with_config(global_df, metadata)
    keys = ["dataset", "strategy"] + (["noise"] if by_noise else []) + ["round"]
    return _bootstrap_by_group(g, keys, "objective_global_accuracy")


def round_one_gas_table(receipts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Fig. 8: round-one gas by tx type (mean per run) plus per-strategy totals, with bootstrap CIs."""
    per_type = _with_config(round_one_gas_per_run(receipts, by_tx_type=True), metadata)
    by_type = _bootstrap_by_group(per_type, ["dataset", "strategy", "tx_type"], "gas_used")
    totals = _with_config(round_one_gas_per_run(receipts), metadata)
    by_total = _bootstrap_by_group(totals, ["dataset", "strategy"], "gas_used")
    by_total["tx_type"] = "total"
    return pd.concat([by_type, by_total], ignore_index=True)


def _bootstrap_by_group(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(keys):
        lo, hi = bootstrap_ci(grp[value].to_numpy())
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "mean": grp[value].mean(), "lo": lo, "hi": hi, "n": len(grp)})
    return pd.DataFrame(rows)


def variance_table(contributions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Fig. 6: per-run mean within-round variance of contribution scores."""
    return _with_config(mean_within_round_variance(contributions), metadata)


# ---------------------------------------------------------------------------
# Figures (layout follows the paper's Fig. 3-8)
# ---------------------------------------------------------------------------

# Okabe-Ito palette, as in the paper.
STRATEGY_COLORS = {"naive": "#7f7f7f", "accuracy_only": "#0072B2",
                   "loss_only": "#D55E00", "accuracy_loss": "#009E73"}
ROLE_COLORS = {"good": "#0072B2", "bad": "#D55E00", "freerider": "#009E73"}
ROLE_HATCHES = {"good": "", "bad": "//", "freerider": "///"}
ROLE_MARKERS = {"good": "o", "bad": "o", "freerider": "^"}
ROLE_LEGEND_LABELS = {"good": "Honest", "bad": "Malicious", "freerider": "Free-Rider"}
# Stacking order of Fig. 8, bottom to top.
GAS_STACK = [("slot", "Slot", "#56B4E9", ""), ("weights", "Weights", "#E69F00", "//"),
             ("feedback", "Feedback", "#009E73", "\\\\"), ("contrib", "Contrib", "#CC79A7", "xx"),
             ("close", "Close", "#999999", "..")]


def plot_fig3(table: pd.DataFrame) -> plt.Figure:
    noises = sorted(table["noise"].unique())
    fig, axes = plt.subplots(len(DATASETS), len(noises), figsize=(12, 8), squeeze=False,
                             constrained_layout=True)
    fig.suptitle("Free-rider removal: count and median post-activation delay")
    for i, ds in enumerate(DATASETS):
        for j, noise in enumerate(noises):
            ax = axes[i][j]
            cell = table[(table["dataset"] == ds) & (table["noise"] == noise)]
            acts = sorted(cell["activation_round"].unique())
            grid = cell.set_index(["strategy", "activation_round"])
            fraction = np.array([[grid.loc[(s, a), "removed"] / grid.loc[(s, a), "total"]
                                  for a in acts] for s in STRATEGIES])
            image = ax.imshow(fraction, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for r, s in enumerate(STRATEGIES):
                for c, a in enumerate(acts):
                    row = grid.loc[(s, a)]
                    text = ("no removals" if row["removed"] == 0
                            else f"median {row['median_delay']:g} r")
                    ax.text(c, r, f"{int(row['removed'])}/{int(row['total'])}\n{text}",
                            ha="center", va="center", fontsize=8, fontweight="bold",
                            color="white" if fraction[r, c] > 0.5 else "black")
            ax.set_xticks(range(len(acts)), [str(a) for a in acts])
            ax.set_yticks(range(len(STRATEGIES)), [STRATEGY_LABELS[s] for s in STRATEGIES])
            ax.set_title(f"{DATASET_LABELS[ds]} · noise {noise:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Contribution strategy")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Free-rider activation round")
    fig.colorbar(image, ax=axes, label="Fraction permanently removed", shrink=0.9)
    return fig


def plot_fig4(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), squeeze=False,
                             constrained_layout=True)
    fig.suptitle("Permanent removal by participant role")
    width = 0.8 / len(ROLES)
    for ax, ds in zip(axes[0], DATASETS):
        for k, role in enumerate(ROLES):
            sub = (table[(table["dataset"] == ds) & (table["role"] == role)]
                   .set_index("strategy").reindex(STRATEGIES))
            xs = np.arange(len(STRATEGIES)) + k * width
            err = [(sub["rate"] - sub["lo"]).to_numpy(), (sub["hi"] - sub["rate"]).to_numpy()]
            ax.bar(xs, sub["rate"], width, yerr=err, capsize=2, color=ROLE_COLORS[role],
                   hatch=ROLE_HATCHES[role], edgecolor="black", linewidth=0.5,
                   label=ROLE_LEGEND_LABELS[role])
        ax.set_xticks(np.arange(len(STRATEGIES)) + width, [STRATEGY_LABELS[s] for s in STRATEGIES],
                      rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Permanent-removal proportion")
        ax.set_title(DATASET_LABELS[ds], fontsize=9)
    axes[0][-1].legend(fontsize=7, ncol=3, loc="upper center")
    return fig


def _strategy_grid(title: str, ylabel: str) -> tuple[plt.Figure, np.ndarray]:
    """2 x 4 grid: dataset rows, strategy columns, shared look of Fig. 5 and Fig. 7."""
    fig, axes = plt.subplots(len(DATASETS), len(STRATEGIES), figsize=(15, 7), squeeze=False,
                             constrained_layout=True)
    fig.suptitle(title)
    for i, ds in enumerate(DATASETS):
        for j, s in enumerate(STRATEGIES):
            ax = axes[i][j]
            ax.set_title(f"{DATASET_LABELS[ds]}\n{STRATEGY_LABELS[s]}", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel(ylabel)
    return fig, axes


def plot_fig5(table: pd.DataFrame, activation_round: int = 3) -> plt.Figure:
    fig, axes = _strategy_grid(
        f"Reputation trajectories (noise=0.01, activation round {activation_round})",
        "Mean GRS (ETH)")
    for i, ds in enumerate(DATASETS):
        for j, s in enumerate(STRATEGIES):
            ax = axes[i][j]
            cell = table[(table["dataset"] == ds) & (table["strategy"] == s)]
            for role in ROLES:
                grp = cell[cell["role"] == role].sort_values("round")
                ax.plot(grp["round"], grp["mean"], color=ROLE_COLORS[role],
                        marker=ROLE_MARKERS[role], markersize=3, linewidth=1,
                        label=ROLE_LEGEND_LABELS[role])
                ax.fill_between(grp["round"], grp["lo"], grp["hi"], color=ROLE_COLORS[role], alpha=0.2)
            ax.axvline(activation_round, color="black", linestyle=":", linewidth=0.8)
            ax.set_ylim(0, 2.2)
    axes[0][-1].legend(fontsize=6, loc="upper left")
    return fig


def plot_fig7(table: pd.DataFrame) -> plt.Figure:
    """`table` has one row per dataset x strategy x noise x round (accuracy in %)."""
    fig, axes = _strategy_grid("Global model accuracy (averaged over activation rounds)",
                               "Global accuracy")
    styles = {0.01: "-", 0.1: "--"}
    for i, ds in enumerate(DATASETS):
        for j, s in enumerate(STRATEGIES):
            ax = axes[i][j]
            for noise, style in styles.items():
                grp = (table[(table["dataset"] == ds) & (table["strategy"] == s)
                             & (table["noise"] == noise)].sort_values("round"))
                ax.plot(grp["round"], grp["mean"] / 100, style, color=STRATEGY_COLORS[s],
                        marker="o", markersize=2, linewidth=1, label=f"noise={noise:g}")
                ax.fill_between(grp["round"], grp["lo"] / 100, grp["hi"] / 100,
                                color=STRATEGY_COLORS[s], alpha=0.2)
    axes[0][-1].legend(fontsize=6, loc="lower right")
    return fig


def plot_fig6(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), squeeze=False,
                             constrained_layout=True)
    fig.suptitle("Contribution-score dispersion across runs")
    for ax, ds in zip(axes[0], DATASETS):
        data = [table[(table["dataset"] == ds) & (table["strategy"] == s)]["variance"].to_numpy()
                for s in STRATEGIES]
        boxes = ax.boxplot(data, positions=range(len(STRATEGIES)), showfliers=False,
                           patch_artist=True)
        for box, s in zip(boxes["boxes"], STRATEGIES):
            box.set(facecolor=STRATEGY_COLORS[s], alpha=0.7)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(len(STRATEGIES)), [STRATEGY_LABELS[s] for s in STRATEGIES],
                      rotation=20, ha="right")
        ax.set_ylabel("Mean within-round score variance")
        ax.set_title(DATASET_LABELS[ds], fontsize=9)
    return fig


def plot_fig8(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), squeeze=False,
                             constrained_layout=True)
    fig.suptitle("Round-one gas by transaction type")
    for ax, ds in zip(axes[0], DATASETS):
        cell = table[table["dataset"] == ds]
        xs = np.arange(len(STRATEGIES))
        bottom = np.zeros(len(STRATEGIES))
        for tx, label, colour, hatch in GAS_STACK:
            part = (cell[cell["tx_type"] == tx].set_index("strategy")
                    .reindex(STRATEGIES)["mean"].to_numpy() / 1e6)
            ax.bar(xs, part, 0.8, bottom=bottom, color=colour, hatch=hatch, label=label)
            bottom += part
        totals = cell[cell["tx_type"] == "total"].set_index("strategy").reindex(STRATEGIES)
        ax.errorbar(xs, totals["mean"] / 1e6, fmt="none", ecolor="black", capsize=2,
                    yerr=[(totals["mean"] - totals["lo"]) / 1e6, (totals["hi"] - totals["mean"]) / 1e6])
        ax.set_xticks(xs, [STRATEGY_LABELS[s] for s in STRATEGIES], rotation=20, ha="right")
        ax.set_ylabel("Round-one gas (millions)")
        ax.set_title(DATASET_LABELS[ds], fontsize=9)
    axes[0][-1].legend(fontsize=6, ncol=2, loc="center")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_merged(runs_dir: Path) -> dict[str, pd.DataFrame]:
    return merge_runs(normalize_runs(load_runs(Path(runs_dir))))


def build_tables(merged: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    meta = merged["metadata"]
    return {
        "fig3": freerider_removal_table(merged["users"], meta),
        "fig4": removal_by_role_table(merged["users"], meta),
        "fig5": mean_grs_by_role_table(merged["users"], meta),
        "fig6": variance_table(merged["contributions"], meta),
        "fig7": global_accuracy_table(merged["global"], meta),
        "fig7_by_noise": global_accuracy_table(merged["global"], meta, by_noise=True),
        "fig8": round_one_gas_table(merged["receipts"], meta),
    }


def build_figures(runs_dir: Path = DEFAULT_RUNS_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    tables = build_tables(load_merged(runs_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotters = {"fig3": plot_fig3, "fig4": plot_fig4, "fig5": plot_fig5,
                "fig6": plot_fig6, "fig7": plot_fig7, "fig8": plot_fig8}
    for name, plot in plotters.items():
        fig = plot(tables["fig7_by_noise" if name == "fig7" else name])
        fig.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close(fig)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Fig. 3-8 of the paper from stored runs.")
    parser.add_argument("runs_dir", nargs="?", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build_figures(args.runs_dir, args.out_dir)
    print(f"Wrote Fig. 3-8 to {args.out_dir}")


if __name__ == "__main__":
    main()
