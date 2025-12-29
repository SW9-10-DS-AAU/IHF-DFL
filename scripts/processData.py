import os
from pathlib import Path
from typing import Optional

from matplotlib import cm, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from parser import *

#RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")
RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")

LISTOFALLMETHODS = [Method.ACCURACY, Method.DOTPRODUCT, Method.DOTPRODUCTANDOUTLIER, Method.NAIVE]
LISTOFALLATTIDUDESASMETA = [MetaAttitude.GOOD, MetaAttitude.FREERIDER, MetaAttitude.MALICIOUS]


def graph_one_one(usePreviousTests: bool = True):
    kickedGraph(1, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activation Round: 1", usePreviousTests, "Graph_1-1", RESULTDATAFOLDER)

def graph_one_two(usePreviousTests: bool = True):
    kickedGraph(3, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 3", usePreviousTests, "Graph_1-2" ,RESULTDATAFOLDER)

def graph_one_three(usePreviousTests: bool = True):
    kickedGraph(5, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 5", usePreviousTests, "Graph_1-3",  RESULTDATAFOLDER)



def graph_two_one(usePreviousTests: bool = True):
    kickedGraphMethodNoise(3, "Freerider noise comparisons - Freerider Activated Round: 3", usePreviousTests, "Graph_2-1", RESULTDATAFOLDER)


def graph_three_one(usePreviousTests: bool = True):
    grsGraph(Attitude.FREERIDER, "Freerider GRS", 3, usePreviousTests, "Graph_3-1", RESULTDATAFOLDER)


def graph_four_one(usePreviousTests: bool = True):
    gasCostGraphMethods("Gas Comparision - Contribution Score Strategies", usePreviousTests, "Graph_4-1", RESULTDATAFOLDER)


def graph_five_one(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.FREERIDER,
        "Dot Product GRS (Freeriders) – With vs Without Outlier Detection",
        usePreviousTests, 
        "Graph_5-1",
        RESULTDATAFOLDER,
    )


def graph_five_two(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.MALICIOUS,
        "Dot Product GRS (Malicious) – With vs Without Outlier Detection",
        usePreviousTests,
        "Graph_5-2",
        RESULTDATAFOLDER,
    )

def graph_five_three(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.GOOD,
        "Dot Product GRS (Honest Users)",
        usePreviousTests,
        "Graph_5-3",
        RESULTDATAFOLDER,
    )

def graph_six_one(usePreviousTests: bool = True):
    # accuracy_grs_gain: 0-11 57-67 110-112
    accuracyGRSGainGraph(
        "Accuracy Scoring – GRS Gained Per Round",
        usePreviousTests,
        "Graph_6-1",
        RESULTDATAFOLDER,
    )

def graph_seven_one(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Accuracy", [Method.ACCURACY], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-1", RESULTDATAFOLDER)

def graph_seven_two(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product Without Outlier Detection", [Method.DOTPRODUCT], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-2", RESULTDATAFOLDER)

def graph_seven_three(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product With Outlier Detection", [Method.DOTPRODUCTANDOUTLIER], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-3", RESULTDATAFOLDER)

def graph_seven_four(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Naive", [Method.NAIVE], LISTOFALLATTIDUDESASMETA, usePreviousTests, "Graph_7-4", RESULTDATAFOLDER)


graph_one_one()
graph_one_two()
graph_one_three()

graph_two_one()

graph_three_one()

graph_four_one()

graph_five_one()
graph_five_two()
graph_five_three()

graph_six_one()

graph_seven_one()
graph_seven_two()
graph_seven_three()
graph_seven_four()