from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from parser.parseExports import runProcessor
from parser.helpers.mehods import Method
from parser.types.participant import MetaAttitude

data = defaultdict(lambda: defaultdict(list))
Y_SCALE = 1e18

def prepare_data_for_contribution_variance(
    rounds,
    participants,
    experiment_specs,
    gasStats,
    outDir,
):
    method = Method.from_string(
        experiment_specs.contribution_score_strategy,
        experiment_specs.use_outlier_detection,
    )

    # need at least two rounds to compute delta
    for r_idx in range(1, len(rounds)):
        prev = rounds[r_idx - 1]
        curr = rounds[r_idx]

        for uid, p in participants.items():
            # user missing → disqualified, skip
            if len(p.states) <= r_idx:
                continue

            attitude = p.states[r_idx].attitude

            try:
                #prev_score = prev.contributionScores[uid - 1]
                curr_score = curr.contributionScores[uid - 1]
            except (IndexError, TypeError):
                continue

            delta = curr_score# - prev_score
            data[method][attitude].append(delta)


def annotate_clipped(ax, values, y_range):
    ymin, ymax = y_range
    yrange = ymax - ymin

    top_y = ymax - 0.03 * yrange
    bottom_y = ymin + 0.03 * yrange

    for i, vals in enumerate(values, start=1):
        vmin = min(vals)
        vmax = max(vals)

        if vmax > ymax:
            ax.text(
                i,
                top_y,
                f"↑ max={vmax / Y_SCALE:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
            )

        if vmin < ymin:
            ax.text(
                i,
                bottom_y,
                f"↓ min={vmin / Y_SCALE:.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color="red",
            )

def plot_contribution_score_variance(title, methods: list[Method], attitudes: list[MetaAttitude], usePreviousTests: bool, windowAndFileName: str, RESULTDATAFOLDER, y_range: tuple[float, float] | None = None):
    runProcessor(
        RESULTDATAFOLDER,
        usePreviousTests,
        prepare_data_for_contribution_variance
    )

    #methods = list(data.keys())
    # attitudes = [
    #     MetaAttitude.GOOD,
    #     MetaAttitude.MALICIOUS,
    #     MetaAttitude.FREERIDER,
    # ]

    values = []
    labels = []

    for method in methods:
        for att in attitudes:
            raw_vals = data[method].get(att, [])

            vals = [
                float(v)
                for v in raw_vals
                if v is not None
            ]

            if not vals:
                vals = [0.0]

            values.append(vals)
            labels.append(f"{method.display_name}\n{att.display_name}")

    fig, ax = plt.subplots(figsize=(12, 5))

    # --- plot violins ---
    ax.violinplot(
        values,
        showmeans=True,
        showmedians=False,
        showextrema=True,
    )

    # --- x axis ---
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=10, ha="right")

    # --- y axis ---
    ax.set_ylabel("Contribution Score per Round")
    ax.set_title(title)

    # scientific notation: 1e18
    ax.ticklabel_format(axis="y", style="sci", scilimits=(18, 18))

    # grid
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if y_range is not None:
        ax.set_ylim(y_range)
        annotate_clipped(ax, values, y_range)
    else:
        ax.relim()
        ax.autoscale_view()

    # --- draw zero line without affecting limits ---
    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        clip_on=False,
        zorder=0,
    )

    fig.canvas.manager.set_window_title(windowAndFileName)

    plt.tight_layout()
    plt.show(block=True)