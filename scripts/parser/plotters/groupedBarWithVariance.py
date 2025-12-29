import numpy as np
import matplotlib.pyplot as plt

def  grouped_bar_with_variance(
    labels,
    means,
    variances,
    group_names,
    missing,
    ylabel="Value",
    title=None,
    figsize=(8, 5)
):
    """
    labels: list[str]            -> x-axis categories
    means: list[list[float]]     -> shape (groups, categories)
    variances: list[list[float]] -> same shape as means
    group_names: list[str]
    """

    means = np.asarray(means)
    #variances = np.asarray(variances)

    n_groups, n_cats = means.shape
    x = np.arange(n_cats)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=figsize)

    

    for i in range(n_groups):
        lows = np.array([v[0] for v in variances[i]], dtype=float)
        ups  = np.array([v[1] for v in variances[i]], dtype=float)

        show_yerr = any(low != 0 or up != 0 for low, up in zip(lows, ups))

        xpos = x - 0.4 + i * width + width / 2
        ax.bar(
            xpos,
            means[i],
            width,
            yerr = [lows, ups] if show_yerr else None,
            capsize=4,
            edgecolor="black",
            linewidth=0.8,
            label=group_names[i]
        )
    
        for j, is_missing in enumerate(missing[i]):
            if is_missing:
                y_offset = ax.get_ylim()[1] * 0.02
                ax.text(
                    xpos[j],
                    y_offset,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="gray",
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show(block=True)
