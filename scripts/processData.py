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
from parser.dataProcessors.graphs.gasCostComparison import gasCostGraph

#RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")
RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")


def graph_one_one():
    kickedGraph(1, "Freerider Activated Round: 3", RESULTDATAFOLDER)

def graph_one_two():
    kickedGraph(3, "Freerider Activated Round: 3", RESULTDATAFOLDER)

def graph_one_three():
    kickedGraph(5, "Freerider Activated Round: 5", RESULTDATAFOLDER)


def graph_two():
    kickedGraphMethodNoise(3, "Freerider noise comparisons", RESULTDATAFOLDER)

def graph_three_one():
    grsGraph(Attitude.BAD, "Malicious GRS", 3, RESULTDATAFOLDER)

def graph_four():
    gasCostGraph(
        "Gas Cost of Contribution Scoring Methods",
        RESULTDATAFOLDER
    )

# graph_one_one()
# graph_one_two()
# graph_one_three()
# graph_two()
#graph_three_one()
graph_four()




