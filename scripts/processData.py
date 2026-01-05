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
DATASET = "cifar"
#DATASET = "mnist"


def graph_one_one(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraph(1, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activation Round: 1", usePreviousTests, f"Graph11{DATASET}", legend_position, RESULTDATAFOLDER)

def graph_one_two(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraph(3, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 3", usePreviousTests, f"Graph12{DATASET}", legend_position, RESULTDATAFOLDER)

def graph_one_three(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraph(5, "Effectiveness of Strategies in Removing Dishonest Participants - Freerider Activated Round: 5", usePreviousTests, f"Graph13{DATASET}", legend_position, RESULTDATAFOLDER)


def graph_two_one(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraphMethodNoise(1, "Freerider noise comparisons - Freerider Activated Round: 1", usePreviousTests, f"Graph21{DATASET}", legend_position, RESULTDATAFOLDER)

def graph_two_two(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraphMethodNoise(3, "Freerider noise comparisons - Freerider Activated Round: 3", usePreviousTests, f"Graph22{DATASET}", legend_position, RESULTDATAFOLDER)

def graph_two_three(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_LOWER_LEFT):
    kickedGraphMethodNoise(5, "Freerider noise comparisons - Freerider Activated Round: 5", usePreviousTests, f"Graph23{DATASET}", legend_position, RESULTDATAFOLDER)



def graph_three_one(usePreviousTests: bool = True):
    grsGraph(Attitude.FREERIDER, "Freerider GRS", 3, usePreviousTests, f"Graph31{DATASET}", RESULTDATAFOLDER)

def graph_three_two(usePreviousTests: bool = True):
    grsGraph(Attitude.MALICIOUS, "Malicious GRS", 3, usePreviousTests, f"Graph32{DATASET}", RESULTDATAFOLDER)

def graph_three_three(usePreviousTests: bool = True):
    grsGraph(Attitude.GOOD, "Honest GRS", 3, usePreviousTests, f"Graph33{DATASET}", RESULTDATAFOLDER)



def graph_three_investigation(usePreviousTests: bool = True):  
    def onlyNoise10(experimentSpec: ExperimentSpec):
        return experimentSpec.freerider_noise_scale == 1.0
    grsGraph(Attitude.FREERIDER, "Freerider GRS - Noise 1.0 ", 3, usePreviousTests, f"Graph3x1{DATASET}", RESULTDATAFOLDER, filter=onlyNoise10)



def graph_four_one(usePreviousTests: bool = True, legend_position: LegendPosition = LegendPosition.INSIDE_UPPER_RIGHT):
    gasCostGraphMethods("Gas Comparision - Contribution Score Strategies", usePreviousTests, f"Graph41{DATASET}", legend_position, RESULTDATAFOLDER)


def graph_five_one(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.FREERIDER,
        "Dot Product GRS (Freeriders) – With vs Without Outlier Detection",
        usePreviousTests, 
        f"Graph51{DATASET}",
        RESULTDATAFOLDER,
        True,
    )


def graph_five_two(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.MALICIOUS,
        "Dot Product GRS (Malicious) – With vs Without Outlier Detection",
        usePreviousTests,
        f"Graph52{DATASET}",
        RESULTDATAFOLDER,
        True,
    )

def graph_five_three(usePreviousTests: bool = True):
    dotProductOutlierGRSGraph(
        Attitude.GOOD,
        "Dot Product GRS (Honest Users)",
        usePreviousTests,
        f"Graph53{DATASET}",
        RESULTDATAFOLDER,
        True,
    )

def graph_six_one(usePreviousTests: bool = True):
    # accuracy_grs_gain: 0-11 57-67 110-112
    accuracyGRSGainGraph(
        "Accuracy Scoring – GRS Gained Per Round",
        usePreviousTests,
        f"Graph61{DATASET}",
        RESULTDATAFOLDER,
    )

def cifar_graph_seven_one(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Accuracy", [Method.ACCURACY], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph71{DATASET}", RESULTDATAFOLDER, y_range=(-3.5e17, 4.5e17))

def cifar_graph_seven_two(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product Without Outlier Detection", [Method.DOTPRODUCT], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph72{DATASET}", RESULTDATAFOLDER, y_range=(0.0e18, 0.3e18))

def cifar_graph_seven_three(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product With Outlier Detection", [Method.DOTPRODUCTANDOUTLIER], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph73{DATASET}", RESULTDATAFOLDER, y_range=(0.0e18, 0.3e18))

def cifar_graph_seven_four(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Naive", [Method.NAIVE], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph74{DATASET}", RESULTDATAFOLDER, y_range=(0.12e18, 0.18e18))



def minist_graph_seven_one(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Accuracy", [Method.ACCURACY], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph71{DATASET}", RESULTDATAFOLDER, y_range=(-1.0e18, 1.2e18))

def minist_graph_seven_two(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product Without Outlier Detection", [Method.DOTPRODUCT], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph72{DATASET}", RESULTDATAFOLDER, y_range=(0.15e18, 0.26e18))

def minist_graph_seven_three(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Dot Product With Outlier Detection", [Method.DOTPRODUCTANDOUTLIER], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph73{DATASET}", RESULTDATAFOLDER, y_range=(0.15e18, 0.26e18))

def minist_graph_seven_four(usePreviousTests: bool = True):
    plot_contribution_score_variance("Variance Of Contribution Score - Naive", [Method.NAIVE], LISTOFALLATTIDUDESASMETA, usePreviousTests, f"Graph74{DATASET}", RESULTDATAFOLDER, y_range=(0.16e18, 0.26e18))


# graph_one_one()
# graph_one_two()
# graph_one_three()

# graph_two_one()
# graph_two_two()
# graph_two_three()

# graph_three_one()
# graph_three_two()
# graph_three_three()

# graph_four_one()



# graph_six_one()

#cifar_graph_seven_one()
#cifar_graph_seven_two()
#cifar_graph_seven_three()
#cifar_graph_seven_four()

# minist_graph_seven_one()
# minist_graph_seven_two()
# minist_graph_seven_three()
# minist_graph_seven_four()

#graph_three_investigation()


## REMOVED
#graph_five_one()
#graph_five_two()
#graph_five_three()

eCounter = 0
errCounter = 0
def test(rounds: list[Round], participants: dict[int, Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir):
    global eCounter
    global errCounter
    if Method.from_config(experiment_specs) != Method.NAIVE: # or experiment_specs.freerider_start_round != 1:
        return
    eCounter += 1

    def getIndexesWithAttitude(attitude: Attitude):
        return [k for k, p in participants.items() if p.futureAttitude == attitude]
    
    targetIds = getIndexesWithAttitude(Attitude.FREERIDER)

    print(f"{experiment_specs.contribution_score_strategy}-{experiment_specs.freerider_start_round}-{experiment_specs.freerider_noise_scale}-{experiment_specs.use_outlier_detection}")
    found = False
    for i, r in enumerate(rounds):
        if found:
            break
        for id in targetIds:
            if r.GRS[id] <= 0.333333e18 and r.GRS[id] != 0:
                errCounter += 1 
                found = True
                break
                



runProcessor(RESULTDATAFOLDER, True, test)
print(f"{errCounter}/{eCounter}")
#graph_three_one()
