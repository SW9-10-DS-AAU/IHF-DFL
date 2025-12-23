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
from parser.dataProcessors.kickedUsersExperiment import create_kicked_graph, get_round_kicked

#RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")
RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")

print(RESULTDATAFOLDER)

runProcessor(RESULTDATAFOLDER, create_kicked_graph)

print(get_round_kicked())




## For deleting files with the 333 bug
# invalid:list[Path] = get_invalid_experiments()
#
# print([i.name for i in invalid])
# invalidFiles = [RESULTDATAFOLDER / (str(Path(*i.parts[-2:])) + ".csv") for i in invalid]
# for p in invalidFiles:
#     if p.exists():
#         print(f"deleting: {str(p)}")
#         #p.unlink() ## DANGEROUS, DELETES A FILE
