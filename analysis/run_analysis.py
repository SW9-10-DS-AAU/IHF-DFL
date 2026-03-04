import sys
from pathlib import Path

# Repo root is one level above this script (analysis/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from analysis import load_runs_recursive, normalize_runs, merge_runs, aggregations as agg, plots

# Set to a specific timestamp folder, or leave as None to scan all folders
FOLDER = "26-02-26--11_04_41"

data_dir = REPO_ROOT / "experiment" / "data" / "experimentData"
if FOLDER:
    data_dir = data_dir / FOLDER

print(f"Scanning: {data_dir}")

runs = load_runs_recursive(data_dir)
print(f"Loaded {len(runs)} run(s)")

if not runs:
    print(
        "\nNo .pkl files found. Possible reasons:\n"
        "  1. No experiments have been run yet with the updated code.\n"
        "  2. The data folder doesn't exist yet.\n"
        "\nRun an experiment first:\n"
        "  ENV=ganache python experiment/experiment_runner.py\n"
        "\nThe .pkl file will appear alongside the .csv in the timestamped folder."
    )
    sys.exit(0)

# Normalize units: wei → ETH, ratio → %
runs = normalize_runs(runs)

# Merge into flat DataFrames tagged with config metadata
merged = merge_runs(runs)

# Aggregations
agg_global   = agg.agg_global_accuracy_by_round(merged["global"])
agg_strategy = agg.agg_accuracy_by_strategy(merged["global"])
agg_final    = agg.agg_final_round_accuracy_by_strategy(merged["global"])
agg_grs      = agg.agg_grs_by_behavior(merged["users"])
agg_kicked   = agg.agg_round_kicked_by_strategy(merged["users"])

# Plots
fig1 = plots.plot_accuracy_loss_over_rounds(agg_global)
fig2 = plots.plot_strategy_comparison_lines(agg_strategy)
fig3 = plots.plot_strategy_comparison_boxplot(agg_final)
fig4 = plots.plot_grs_by_behavior(agg_grs)
fig5 = plots.plot_round_kicked_by_strategy(agg_kicked)

# Save
figures_dir = REPO_ROOT / "figures"
figures_dir.mkdir(exist_ok=True)
plots.save_figure(fig1, figures_dir / "accuracy_loss.png")
plots.save_figure(fig2, figures_dir / "strategy_lines.png")
plots.save_figure(fig3, figures_dir / "strategy_boxplot.png")
plots.save_figure(fig4, figures_dir / "grs_by_behavior.png")
plots.save_figure(fig5, figures_dir / "round_kicked.png")
print(f"Figures saved to: {figures_dir}")

plt.show()
