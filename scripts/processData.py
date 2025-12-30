import os
from pathlib import Path
from typing import Optional

from matplotlib import cm, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from parser import *
from parser.helpers.setLegendLocation import LegendPosition

#RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")
RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")

LISTOFALLMETHODS = [Method.ACCURACY, Method.DOTPRODUCT, Method.DOTPRODUCTANDOUTLIER, Method.NAIVE]
LISTOFALLATTIDUDESASMETA = [MetaAttitude.GOOD, MetaAttitude.FREERIDER, MetaAttitude.MALICIOUS]


def graph_one_one(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraph(1, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activation Round: 1", usePreviousTests, "Graph_1-1", legend_position, RESULTDATAFOLDER)

def graph_one_two(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraph(3, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 3", usePreviousTests, "Graph_1-2", legend_position, RESULTDATAFOLDER)

def graph_one_three(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraph(5, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 5", usePreviousTests, "Graph_1-3", legend_position, RESULTDATAFOLDER)



def graph_two_one(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraphMethodNoise(3, "Freerider noise comparisons - Freerider Activated Round: 3", usePreviousTests, "Graph_2-1", legend_position, RESULTDATAFOLDER)


def graph_three_one(usePreviousTests: bool = True):
    grsGraph(Attitude.FREERIDER, "Freerider GRS", 3, usePreviousTests, "Graph_3-1", RESULTDATAFOLDER)


def graph_four_one(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_UPPER_RIGHT):
    gasCostGraphMethods("Gas Comparision - Contribution Score Strategies", usePreviousTests, "Graph_4-1", legend_position, RESULTDATAFOLDER)


def graph_five_one(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.FREERIDER,
        "Dot Product GRS (Freeriders) – With vs Without Outlier Detection",
        usePreviousTests, 
        "Graph_5-1",
        RESULTDATAFOLDER,
        True,
    )


def graph_five_two(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.MALICIOUS,
        "Dot Product GRS (Malicious) – With vs Without Outlier Detection",
        usePreviousTests,
        "Graph_5-2",
        RESULTDATAFOLDER,
        True,
    )

def graph_five_three(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.GOOD,
        "Dot Product GRS (Honest Users)",
        usePreviousTests,
        "Graph_5-3",
        RESULTDATAFOLDER,
        True,
    )

def graph_six_one(usePreviousTests: bool = True):
    # accuracy_grs_gain: 0-11 57-67 110-112
    accuracyGRSGainGraph(
        "Accuracy Scoring – GRS Gained Per Round",
        usePreviousTests,
        "Graph_6-1",
        RESULTDATAFOLDER,
    )

def cifar_graph_seven_one(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Accuracy", [Method.ACCURACY], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-1", RESULTDATAFOLDER, y_range=(-3.5e17, 4.5e17))

def cifar_graph_seven_two(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product Without Outlier Detection", [Method.DOTPRODUCT], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-2", RESULTDATAFOLDER, y_range=(0.0e18, 0.3e18))

def cifar_graph_seven_three(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product With Outlier Detection", [Method.DOTPRODUCTANDOUTLIER], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-3", RESULTDATAFOLDER, y_range=(0.0e18, 0.3e18))

def cifar_graph_seven_four(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Naive", [Method.NAIVE], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-4", RESULTDATAFOLDER, y_range=(0.12e18, 0.18e18))



def minist_graph_seven_one(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Accuracy", [Method.ACCURACY], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-1", RESULTDATAFOLDER, y_range=(-1.0e18, 1.2e18))

def minist_graph_seven_two(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product Without Outlier Detection", [Method.DOTPRODUCT], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-2", RESULTDATAFOLDER, y_range=(0.15e18, 0.26e18))

def minist_graph_seven_three(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product With Outlier Detection", [Method.DOTPRODUCTANDOUTLIER], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-3", RESULTDATAFOLDER, y_range=(0.15e18, 0.26e18))

def minist_graph_seven_four(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Naive", [Method.NAIVE], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-4", RESULTDATAFOLDER, y_range=(0.16e18, 0.26e18))


# graph_one_one()
# graph_one_two()
# graph_one_three()

# graph_two_one()

# graph_three_one()

# graph_four_one()

# graph_five_one()
# graph_five_two()
# graph_five_three()

# graph_six_one()

# cifar_graph_seven_one()
# cifar_graph_seven_two()
# cifar_graph_seven_three()
# cifar_graph_seven_four()

minist_graph_seven_one()
minist_graph_seven_two()
minist_graph_seven_three()
minist_graph_seven_four()

def test(rounds: list[Round], participants: dict[int, Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir):
    if Method.from_config(experiment_specs) != Method.NAIVE or experiment_specs.freerider_start_round != 1:
        return
    print(f"{experiment_specs.contribution_score_strategy}-{experiment_specs.freerider_start_round}-{experiment_specs.freerider_noise_scale}-{experiment_specs.use_outlier_detection}")
    for i, r in enumerate(rounds):
        print(f"{participants[7].futureAttitude.display_name}-{r.nr}-{r.GRS[7]}")
        if r.GRS[7] == 0:
            break

        

#runProcessor(RESULTDATAFOLDER, True, test)

#graph_three_one()
