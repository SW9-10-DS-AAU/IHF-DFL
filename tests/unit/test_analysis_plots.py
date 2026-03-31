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
from analysis_helpers import make_users, make_metadata


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
def agg_weights(two_exp_users):
    return agg.agg_merge_weights_by_behavior(two_exp_users)


@pytest.fixture
def agg_stats(two_exp_users):
    return agg.agg_merge_stats_by_behavior(two_exp_users)


@pytest.fixture
def agg_acc_by_strategy(two_exp_global, two_exp_metadata):
    return agg.global_acc_by_aggregation_strategy(two_exp_global, two_exp_metadata)


@pytest.fixture
def agg_loss_by_strategy(two_exp_global, two_exp_metadata):
    return agg.global_loss_by_aggregation_strategy(two_exp_global, two_exp_metadata)


@pytest.fixture
def agg_gas(two_exp_metadata):
    receipts = pd.DataFrame([
        {"experiment_id": exp, "round": r, "tx_type": tx, "gas_used": 50000}
        for exp in ["exp-A", "exp-B"]
        for r in [1, 2]
        for tx in ["register", "slot", "weights"]
    ])
    return agg.agg_gas_used_by_tx_type(receipts, two_exp_metadata)


@pytest.fixture
def agg_kicked():
    pass  # imported at top
    users = make_users(["exp-A"])
    meta  = make_metadata(["exp-A"])
    mask = (users["user_id"] == 1) & (users["round"] >= 4)
    users.loc[mask, "state"] = "disqualified"
    return agg.agg_round_kicked_by_strategy(users, meta)


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
# plot_merge_weights_by_behavior
# ---------------------------------------------------------------------------

def test_plot_merge_weights_no_stats_returns_figure(agg_weights):
    fig = plots.plot_merge_weights_by_behavior(agg_weights)
    assert isinstance(fig, plt.Figure)


def test_plot_merge_weights_with_stats_returns_figure(agg_weights, agg_stats):
    fig = plots.plot_merge_weights_by_behavior(agg_weights, stats=agg_stats)
    assert isinstance(fig, plt.Figure)


def test_plot_merge_weights_xlabel(agg_weights):
    fig = plots.plot_merge_weights_by_behavior(agg_weights)
    assert fig.axes[0].get_xlabel() == "Round"


def test_plot_merge_weights_ylabel(agg_weights):
    fig = plots.plot_merge_weights_by_behavior(agg_weights)
    assert fig.axes[0].get_ylabel() == "Merge Weight"


def test_plot_merge_weights_legend_has_title_when_stats(agg_weights, agg_stats):
    fig = plots.plot_merge_weights_by_behavior(agg_weights, stats=agg_stats)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() != ""


# ---------------------------------------------------------------------------
# plot_global_acc_by_aggregation_strategy
# ---------------------------------------------------------------------------

def test_plot_global_acc_by_aggregation_strategy_returns_figure(agg_acc_by_strategy):
    fig = plots.plot_global_acc_by_aggregation_strategy(agg_acc_by_strategy)
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_global_loss_by_aggregation_strategy
# ---------------------------------------------------------------------------

def test_plot_global_loss_by_aggregation_strategy_returns_figure(agg_loss_by_strategy):
    fig = plots.plot_global_loss_by_aggregation_strategy(agg_loss_by_strategy)
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_gas_cost_by_tx_type
# ---------------------------------------------------------------------------

def test_plot_gas_cost_by_tx_type_returns_figure(agg_gas):
    fig = plots.plot_gas_cost_by_tx_type(agg_gas)
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_round_kicked_by_strategy
# ---------------------------------------------------------------------------

def test_plot_round_kicked_by_strategy_returns_figure(agg_kicked):
    fig = plots.plot_round_kicked_by_strategy(agg_kicked)
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

def test_save_figure_creates_file(tmp_path, agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    out = tmp_path / "output.png"
    plots.save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_png_and_pdf(tmp_path, agg_global):
    fig = plots.plot_accuracy_loss_over_rounds(agg_global)
    for ext in ["png", "pdf"]:
        out = tmp_path / f"output.{ext}"
        plots.save_figure(fig, out)
        assert out.exists()
