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

#RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")
RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/sample")

print(RESULTDATAFOLDER)

runProcessor(RESULTDATAFOLDER, save_experiment_name_if_invalid)

#totals = get_totals()
#avg = sum(totals) / len(totals)

#print(avg)
invalid:list[Path] = get_invalid_experiments()

print([i.name for i in invalid])
invalidFiles = [RESULTDATAFOLDER / (str(Path(*i.parts[-2:])) + ".csv") for i in invalid]
for p in invalidFiles:
    if p.exists():
        print(f"deleting: {str(p)}")
        #p.unlink() ## DANGEROUS, DELETES A FILE
