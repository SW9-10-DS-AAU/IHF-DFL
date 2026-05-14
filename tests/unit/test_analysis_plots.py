"""Smoke tests for analysis/plots.py.

Each test verifies:
  - The function returns a matplotlib Figure without raising.
  - Key axis labels are set.
  - (Where applicable) The correct number of lines is drawn.

matplotlib is forced to the non-interactive Agg backend so tests run headlessly.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import analysis.plots as plots
import analysis.aggregations as agg


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after every test to avoid memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helper fixtures — pre-aggregated DataFrames matching each plot's expectations
# ---------------------------------------------------------------------------

@pytest.fixture
def agg_global(two_exp_global):
    return agg.agg_global_accuracy_loss_by_round(two_exp_global)


@pytest.fixture
def agg_grs(two_exp_users, two_exp_metadata):
    return agg.agg_grs_by_role(two_exp_users, two_exp_metadata)

@pytest.fixture
def agg_gas(two_exp_metadata):
    receipts = pd.DataFrame([
        {"experiment_id": exp, "round": r, "tx_type": tx, "gas_used": 50000}
        for exp in ["exp-A", "exp-B"]
        for r in [1, 2]
        for tx in ["register", "slot", "weights"]
    ])
    return agg.agg_gas_used_by_tx_type(receipts, two_exp_metadata)

# ---------------------------------------------------------------------------
# plot_accuracy_loss_over_rounds
# ---------------------------------------------------------------------------

def test_plot_accuracy_loss_returns_figure(agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    assert isinstance(fig, plt.Figure)


def test_plot_accuracy_loss_has_two_axes(agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    assert len(fig.axes) == 2  # primary + twinx


def test_plot_accuracy_loss_xlabel(agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    assert fig.axes[0].get_xlabel() == "Round"


# ---------------------------------------------------------------------------
# plot_grs_by_role
# ---------------------------------------------------------------------------

def test_plot_grs_by_role_returns_figure(agg_grs):
    fig = plots.plot_grs_by_role(agg_grs)
    assert isinstance(fig, plt.Figure)


def test_plot_grs_by_role_xlabel(agg_grs):
    fig = plots.plot_grs_by_role(agg_grs)
    assert fig.axes[0].get_xlabel() == "Round"


def test_plot_grs_by_role_has_four_lines(agg_grs):
    fig = plots.plot_grs_by_role(agg_grs)
    # One Line2D per role (4 roles)
    lines = [l for l in fig.axes[0].lines if len(l.get_xdata()) > 0]
    assert len(lines) == 4


# ---------------------------------------------------------------------------
# plot_gas_cost_by_tx_type
# ---------------------------------------------------------------------------

def test_save_figure_creates_file(tmp_path, agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    plots.save_figure(fig, tmp_path, experiment_name="test")

    out_dir = tmp_path / "test"
    files = list(out_dir.glob("*.svg"))
    assert len(files) == 1
    assert files[0].stat().st_size > 0


def test_save_figure_auto_names_svg_files(tmp_path, agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    plots.save_figure(fig, tmp_path, experiment_name="test")
    plots.save_figure(fig, tmp_path, experiment_name="test", suffix="second")

    files = sorted((tmp_path / "test").glob("*.svg"))
    assert [file.name for file in files] == [
        "001-accuracy_loss_over_rounds.svg",
        "002-accuracy_loss_over_rounds-second.svg",
    ]


def test_save_figure_png_and_pdf(tmp_path, agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    for ext in ["png", "pdf"]:
        out = tmp_path / f"output.{ext}"
        plots.save_figure(fig, out)
        assert out.exists()
