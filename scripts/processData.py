import os
from pathlib import Path
from typing import Optional

from matplotlib import cm, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from parser import *
from parser.dataProcessors.plotter import plotData
from parser.dataProcessors.gasCostExtractor import print_gas, get_totals
from parser.dataProcessors.findInavlidExperiments import save_experiment_name_if_invalid, get_invalid_experiments
from parser.dataProcessors.graphs.GRSOverRounds import grsGraph
from parser.dataProcessors.graphs.kickedUsersExperiment import kickedGraph, prepare_data_for_graph, get_round_kicked, format_for_grouped_bar
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance
from parser.dataProcessors.graphs.noiseGraph import kickedGraphMethodNoise
from parser.dataProcessors.graphs.gasGraph import gasCostGraphMethods
from parser.dataProcessors.graphs.contributionScoreVarianceGraph import plot_contribution_score_variance
from parser.participant import MetaAttitude

#RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")
RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")

LISTOFALLMETHODS = [Method.ACCURACY, Method.DOTPRODUCT, Method.DOTPRODUCTANDOUTLIER, Method.NAIVE]
LISTOFALLATTIDUDESASMETA = [MetaAttitude.GOOD, MetaAttitude.FREERIDER, MetaAttitude.MALICIOUS]


def graph_one_one(usePreviousTests: bool = True):
    kickedGraph(1, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activation Round: 1", usePreviousTests, RESULTDATAFOLDER)

def graph_one_two(usePreviousTests: bool = True):
    kickedGraph(3, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 3", usePreviousTests, RESULTDATAFOLDER)

def graph_one_three(usePreviousTests: bool = True):
    kickedGraph(5, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 5", usePreviousTests, RESULTDATAFOLDER)



def graph_two(usePreviousTests: bool = True):
    kickedGraphMethodNoise(3, "Freerider noise comparisons - Freerider Activated Round: 3", usePreviousTests, RESULTDATAFOLDER)


def graph_three_one(usePreviousTests: bool = True):
    grsGraph(Attitude.FREERIDER, "Freerider GRS", 3, usePreviousTests, RESULTDATAFOLDER)


def graph_four_one(usePreviousTests: bool = True):
    gasCostGraphMethods("Gas Comparision - Contribution Score Strategies", usePreviousTests, RESULTDATAFOLDER)

def graph_seven_one(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Accuracy", [Method.ACCURACY], LISTOFALLATTIDUDESASMETA, usePreviousTests, RESULTDATAFOLDER)

def graph_seven_two(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product Without Outlier Detection", [Method.DOTPRODUCT], LISTOFALLATTIDUDESASMETA, usePreviousTests, RESULTDATAFOLDER)

def graph_seven_three(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product With Outlier Detection", [Method.DOTPRODUCTANDOUTLIER], LISTOFALLATTIDUDESASMETA, usePreviousTests, RESULTDATAFOLDER)

def graph_seven_four(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Naive", [Method.NAIVE], LISTOFALLATTIDUDESASMETA, usePreviousTests, RESULTDATAFOLDER)

def graph_four():
    gasCostGraph(
        "Gas Cost of Contribution Scoring Methods",
        RESULTDATAFOLDER
    )

def graph_five_one():
    # Dotproduct data
    # 11-33 49-52

    dotProductOutlierGRSGraph(
        Attitude.FREERIDER,
        "Dot Product GRS (Freeriders) – With vs Without Outlier Detection",
        RESULTDATAFOLDER,
    )


def graph_five_two():
    dotProductOutlierGRSGraph(
        Attitude.BAD,
        "Dot Product GRS (Malicious) – With vs Without Outlier Detection",
        RESULTDATAFOLDER,
    )

def graph_five_three():
    dotProductOutlierGRSGraph(
        Attitude.GOOD,
        "Dot Product GRS (Honest Users)",
        RESULTDATAFOLDER,
    )

def graph_six():
    # accuracy_grs_gain: 0-11 57-67 110-112
    accuracyGRSGainGraph(
        "Accuracy Scoring – GRS Gained Per Round",
        RESULTDATAFOLDER,
    )

# graph_one_one()
# graph_one_two()
# graph_one_three()

#graph_two()

#graph_three_one()

#graph_four_one()

graph_seven_one()
graph_seven_two()
graph_seven_three()
graph_seven_four()