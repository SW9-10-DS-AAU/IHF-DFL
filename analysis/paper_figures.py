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
ROLES = ["good", "freerider", "bad"]
ROLE_LABELS = {"good": "Honest", "freerider": "Free-rider", "bad": "Malicious"}


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


def global_accuracy_table(global_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Fig. 7: global accuracy (%) over rounds, bootstrap over runs."""
    g = _with_config(global_df, metadata)
    return _bootstrap_by_group(g, ["dataset", "strategy", "round"], "objective_global_accuracy")


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
# Figures
# ---------------------------------------------------------------------------

def plot_fig3(table: pd.DataFrame) -> plt.Figure:
    noises = sorted(table["noise"].unique())
    fig, axes = plt.subplots(len(DATASETS), len(noises), figsize=(5 * len(noises), 8), squeeze=False)
    for i, ds in enumerate(DATASETS):
        for j, noise in enumerate(noises):
            ax = axes[i][j]
            cell = table[(table["dataset"] == ds) & (table["noise"] == noise)]
            acts = sorted(cell["activation_round"].unique())
            width = 0.8 / len(STRATEGIES)
            for k, s in enumerate(STRATEGIES):
                sub = cell[cell["strategy"] == s].set_index("activation_round").reindex(acts)
                xs = np.arange(len(acts)) + k * width
                ax.bar(xs, sub["removed"], width, label=STRATEGY_LABELS[s])
                for x, rem, med in zip(xs, sub["removed"], sub["median_delay"]):
                    label = f"{int(rem)}" if pd.isna(med) else f"{int(rem)}\n{med:g}"
                    ax.text(x, rem, label, ha="center", va="bottom", fontsize=7)
            ax.set_xticks(np.arange(len(acts)) + 0.4 - width / 2)
            ax.set_xticklabels([str(a) for a in acts])
            ax.set_ylim(0, 12.5)
            ax.set_xlabel("Free-rider activation round")
            ax.set_ylabel("Removed free-riders (of 10)")
            ax.set_title(f"{DATASET_LABELS[ds]}, noise {noise:g}")
    axes[0][0].legend(fontsize=7, title="count / median delay")
    fig.tight_layout()
    return fig


def plot_fig4(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), squeeze=False)
    for ax, ds in zip(axes[0], DATASETS):
        width = 0.8 / len(STRATEGIES)
        for k, s in enumerate(STRATEGIES):
            sub = table[(table["dataset"] == ds) & (table["strategy"] == s)].set_index("role").reindex(ROLES)
            xs = np.arange(len(ROLES)) + k * width
            err = [(sub["rate"] - sub["lo"]).to_numpy(), (sub["hi"] - sub["rate"]).to_numpy()]
            ax.bar(xs, sub["rate"], width, yerr=err, capsize=2, label=STRATEGY_LABELS[s])
        ax.set_xticks(np.arange(len(ROLES)) + 0.4 - width / 2)
        ax.set_xticklabels([ROLE_LABELS[r] for r in ROLES])
        ax.set_ylabel("Permanently removed (share)")
        ax.set_title(DATASET_LABELS[ds])
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _line_grid(table: pd.DataFrame, group_col: str | None, labels: dict, ylabel: str,
               value_label: str = "mean") -> plt.Figure:
    fig, axes = plt.subplots(len(DATASETS), len(STRATEGIES), figsize=(16, 7),
                             sharex=True, squeeze=False)
    for i, ds in enumerate(DATASETS):
        for j, s in enumerate(STRATEGIES):
            ax = axes[i][j]
            cell = table[(table["dataset"] == ds) & (table["strategy"] == s)]
            groups = [(None, cell)] if group_col is None else list(cell.groupby(group_col))
            for name, grp in groups:
                grp = grp.sort_values("round")
                ax.plot(grp["round"], grp[value_label], label=labels.get(name, name))
                ax.fill_between(grp["round"], grp["lo"], grp["hi"], alpha=0.25)
            ax.set_title(f"{DATASET_LABELS[ds]} / {STRATEGY_LABELS[s]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Round")
    if group_col is not None:
        axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_fig5(table: pd.DataFrame) -> plt.Figure:
    return _line_grid(table, "role", ROLE_LABELS, "Mean GRS (ETH)")


def plot_fig7(table: pd.DataFrame) -> plt.Figure:
    return _line_grid(table, None, {}, "Global accuracy (%)")


def plot_fig6(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), squeeze=False)
    for ax, ds in zip(axes[0], DATASETS):
        data = [table[(table["dataset"] == ds) & (table["strategy"] == s)]["variance"].to_numpy()
                for s in STRATEGIES]
        positions = np.arange(len(STRATEGIES))
        ax.boxplot(data, positions=positions, showfliers=False)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for pos, vals in zip(positions, data):
            ax.scatter(pos + rng.uniform(-0.15, 0.15, len(vals)), vals, s=6, alpha=0.4)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xticks(positions)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], fontsize=8)
        ax.set_ylabel("Mean within-round score variance")
        ax.set_title(DATASET_LABELS[ds])
    fig.tight_layout()
    return fig


def plot_fig8(table: pd.DataFrame) -> plt.Figure:
    tx_types = [t for t in table["tx_type"].unique() if t != "total"] + ["total"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4.5), squeeze=False)
    for ax, ds in zip(axes[0], DATASETS):
        width = 0.8 / len(STRATEGIES)
        for k, s in enumerate(STRATEGIES):
            sub = (table[(table["dataset"] == ds) & (table["strategy"] == s)]
                   .set_index("tx_type").reindex(tx_types))
            xs = np.arange(len(tx_types)) + k * width
            err = [(sub["mean"] - sub["lo"]).to_numpy(), (sub["hi"] - sub["mean"]).to_numpy()]
            ax.bar(xs, sub["mean"], width, yerr=err, capsize=2, label=STRATEGY_LABELS[s])
        ax.set_xticks(np.arange(len(tx_types)) + 0.4 - width / 2)
        ax.set_xticklabels(tx_types, fontsize=8)
        ax.set_ylabel("Round-one gas per run")
        ax.set_title(DATASET_LABELS[ds])
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
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
        "fig8": round_one_gas_table(merged["receipts"], meta),
    }


def build_figures(runs_dir: Path = DEFAULT_RUNS_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    tables = build_tables(load_merged(runs_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotters = {"fig3": plot_fig3, "fig4": plot_fig4, "fig5": plot_fig5,
                "fig6": plot_fig6, "fig7": plot_fig7, "fig8": plot_fig8}
    for name, plot in plotters.items():
        fig = plot(tables[name])
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
