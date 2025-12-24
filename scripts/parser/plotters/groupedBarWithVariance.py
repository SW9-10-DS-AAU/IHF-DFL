import numpy as np
import matplotlib.pyplot as plt

def grouped_bar_with_variance(
    labels,
    means,
    variances,
    group_names,
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
    variances = np.asarray(variances)

    n_groups, n_cats = means.shape
    x = np.arange(n_cats)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_groups):
        ax.bar(
            x - 0.4 + i * width + width / 2,
            means[i],
            width,
            yerr=variances[i],
            capsize=4,
            edgecolor="black",
            linewidth=0.8,
            label=group_names[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


labels = ['Naive', 'Dot', 'Dot+Outlier', 'Accuracy']
means = [
    [10, 12, 9, 4],
    [8, 11, 7, 2],
    [8, 11, 7, 8],
    [8, 11, 7, 1]
]
variances = [
    [2, 1.5, 1, 1],
    [1, 2, 1.2, 2],
    [2, 1.5, 1, 4],
    [2, 1.5, 1, 0.5]
]
group_names = ['Model A', 'Model B', 'Model B', 'Model B']

grouped_bar_with_variance(labels, means, variances, group_names)