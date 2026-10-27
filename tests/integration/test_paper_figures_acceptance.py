"""Acceptance numbers of the paper's Fig. 3-8, recomputed from the 480 stored runs.

Checks the computed values, not the images. Reads the runs from $IHF_RUNS_DIR,
falling back to data/runs_paper_v1 in the repo root; skipped when neither exists (it is gitignored).
The paper's runs live apart from data/runs, where new runs land, so a rerun never mixes into them.
"""
import os
from pathlib import Path

import pytest

from analysis import paper_figures as pf

PAPER_RUNS_DIR = pf.DEFAULT_RUNS_DIR.parent / "runs_paper_v1"
RUNS_DIR = Path(os.environ.get("IHF_RUNS_DIR", PAPER_RUNS_DIR))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUNS_DIR.exists(), reason=f"stored runs not found at {RUNS_DIR}"),
]


@pytest.fixture(scope="module")
def tables():
    return pf.build_tables(pf.load_merged(RUNS_DIR))


def _removed(table, **where):
    sel = table
    for col, val in where.items():
        sel = sel[sel[col] == val]
    assert len(sel) == 1, where
    return int(sel["removed"].iloc[0]), int(sel["total"].iloc[0])


def test_fig3_has_48_cells_of_ten_freeriders(tables):
    f3 = tables["fig3"]
    assert len(f3) == 48
    assert set(f3["total"]) == {10}


# (removed of 10, median delay) per activation round 1/3/5, transcribed from the paper's Fig. 3.
# None = the paper prints "no removals".
PAPER_FIG3 = {
    ("mnist", 0.01): {
        "naive": [(8, 2), (0, None), (0, None)], "accuracy_only": [(10, 2), (7, 5), (4, 3.5)],
        "loss_only": [(10, 2), (10, 3), (9, 3)], "accuracy_loss": [(10, 2), (4, 7), (1, 5)]},
    ("mnist", 0.1): {
        "naive": [(10, 2), (10, 3), (10, 3)], "accuracy_only": [(10, 2), (10, 3), (10, 3)],
        "loss_only": [(10, 2), (10, 3), (10, 3)], "accuracy_loss": [(10, 2), (10, 3), (10, 3)]},
    ("cifar.10", 0.01): {
        "naive": [(0, None), (0, None), (0, None)], "accuracy_only": [(10, 2), (10, 5), (10, 5)],
        "loss_only": [(10, 2.5), (10, 6.5), (10, 4)], "accuracy_loss": [(10, 3.5), (1, 7), (1, 14)]},
    ("cifar.10", 0.1): {
        "naive": [(10, 3), (10, 3), (10, 3)], "accuracy_only": [(10, 3), (10, 3), (10, 3)],
        "loss_only": [(10, 2), (10, 3), (10, 3)], "accuracy_loss": [(10, 2), (10, 3), (10, 3)]},
}


def test_fig3_every_cell_matches_the_paper(tables):
    f3 = tables["fig3"].set_index(["dataset", "noise", "strategy", "activation_round"])
    for (dataset, noise), strategies in PAPER_FIG3.items():
        for strategy, cells in strategies.items():
            for activation_round, (removed, delay) in zip([1, 3, 5], cells):
                row = f3.loc[(dataset, noise, strategy, activation_round)]
                assert int(row["removed"]) == removed, (dataset, noise, strategy, activation_round)
                if delay is not None:
                    assert row["median_delay"] == delay, (dataset, noise, strategy, activation_round)


def test_fig3_low_noise_removal_totals(tables):
    f3 = tables["fig3"][lambda d: d["noise"] == 0.01]
    assert f3.groupby("strategy")["removed"].sum().to_dict() == {
        "loss_only": 59, "accuracy_only": 51, "accuracy_loss": 27, "naive": 8}


def test_fig3_high_noise_removes_every_freerider(tables):
    f3 = tables["fig3"][lambda d: d["noise"] == 0.1]
    assert f3.groupby("strategy")["removed"].sum().to_dict() == {
        "loss_only": 60, "accuracy_only": 60, "accuracy_loss": 60, "naive": 60}


def test_fig3_loss_median_delay_on_cifar_activation_round_3(tables):
    f3 = tables["fig3"]
    cell = f3[(f3["dataset"] == "cifar.10") & (f3["strategy"] == "loss_only")
              & (f3["activation_round"] == 3) & (f3["noise"] == 0.01)]
    assert cell["median_delay"].iloc[0] == 6.5


@pytest.mark.parametrize("dataset, strategy, expected", [
    ("mnist", "accuracy_loss", 4), ("mnist", "accuracy_only", 0),
    ("mnist", "loss_only", 0), ("mnist", "naive", 0),
    ("cifar.10", "accuracy_only", 11), ("cifar.10", "accuracy_loss", 10),
    ("cifar.10", "loss_only", 7), ("cifar.10", "naive", 0),
])
def test_fig4_malicious_removals(tables, dataset, strategy, expected):
    assert _removed(tables["fig4"], dataset=dataset, strategy=strategy, role="bad") == (expected, 60)


@pytest.mark.parametrize("dataset, strategy, expected, total", [
    ("mnist", "accuracy_loss", 19, 240), ("mnist", "accuracy_only", 1, 240),
    ("mnist", "loss_only", 1, 240), ("mnist", "naive", 0, 240),
    ("cifar.10", "accuracy_loss", 55, 360), ("cifar.10", "accuracy_only", 43, 360),
    ("cifar.10", "loss_only", 50, 360), ("cifar.10", "naive", 0, 360),
])
def test_fig4_honest_removals(tables, dataset, strategy, expected, total):
    assert _removed(tables["fig4"], dataset=dataset, strategy=strategy, role="good") == (expected, total)


def test_fig6_naive_variance_is_zero(tables):
    f6 = tables["fig6"]
    assert (f6[f6["strategy"] == "naive"]["variance"] == 0).all()


@pytest.mark.parametrize("dataset, expected_max", [("mnist", 13.2), ("cifar.10", 54.3)])
def test_fig6_accuracy_loss_maximum(tables, dataset, expected_max):
    f6 = tables["fig6"]
    cell = f6[(f6["dataset"] == dataset) & (f6["strategy"] == "accuracy_loss")]
    assert cell["variance"].max() == pytest.approx(expected_max, abs=0.05)


@pytest.mark.parametrize("dataset, strategy, expected", [
    ("mnist", "naive", 94.4), ("mnist", "accuracy_only", 94.4),
    ("mnist", "loss_only", 94.4), ("mnist", "accuracy_loss", 94.4),
    ("cifar.10", "naive", 64.6), ("cifar.10", "accuracy_only", 65.3),
    ("cifar.10", "loss_only", 65.0), ("cifar.10", "accuracy_loss", 64.6),
])
def test_fig7_final_accuracy(tables, dataset, strategy, expected):
    f7 = tables["fig7"]
    cell = f7[(f7["dataset"] == dataset) & (f7["strategy"] == strategy)]
    final = cell[cell["round"] == cell["round"].max()]["mean"].iloc[0]
    # The ticket gives MNIST as "about 94.4%"; the four strategies span 94.39-94.45.
    assert final == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("dataset, strategy, expected_pct", [
    ("mnist", "accuracy_only", 39.8), ("mnist", "loss_only", 40.1),
    ("mnist", "accuracy_loss", 52.9),
    ("cifar.10", "accuracy_only", 38.6), ("cifar.10", "loss_only", 39.6),
    ("cifar.10", "accuracy_loss", 49.1),
])
def test_fig8_round_one_gas_versus_naive(tables, dataset, strategy, expected_pct):
    f8 = tables["fig8"]
    total = f8[(f8["tx_type"] == "total") & (f8["dataset"] == dataset)].set_index("strategy")["mean"]
    assert 100 * (total[strategy] / total["naive"] - 1) == pytest.approx(expected_pct, abs=0.05)
