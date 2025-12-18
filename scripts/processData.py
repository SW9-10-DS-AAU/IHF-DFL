import os
from pathlib import Path
from typing import Optional

from matplotlib import cm, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from parser import *
from parser.dataProcessors.plotter import plotData
from parser.dataProcessors.gasCostExtractor import print_gas, get_totals
from parser.dataProcessors.findInavlidExperiments import save_round_if_invalid, getInvalidRounds

RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")

print(RESULTDATAFOLDER)

runProcessor(RESULTDATAFOLDER, print_gas)

totals = get_totals()
avg = sum(totals) / len(totals)

print(avg)