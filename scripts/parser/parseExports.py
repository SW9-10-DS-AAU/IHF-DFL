from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from typing import Any, Callable

from openfl.ml.pytorch_model import Participant
from .gasCosts import GasStats
from .round import Round
from .parser import load_data
from .selector import choose_from_list

def runProcessor(RESULTDATAFOLDER: Path, processor: Callable[[list[Round], list[Participant], GasStats, str], Any]):
    dirs = sorted([d for d in RESULTDATAFOLDER.iterdir() if d.is_dir()])

    chosenDir = choose_from_list(dirs, "Choose test", True)
    

    tests = sorted([d for d in chosenDir[0].iterdir() if not d.is_dir()])

    chosenTests = choose_from_list(tests, "Chose tests to process")

    for dir in chosenTests:
        outdir = RESULTDATAFOLDER.parent.joinpath("compiledData").joinpath(dir.parts[-2]).joinpath(Path(dir.parts[-1]).stem)
        print(outdir)
        rounds, participants, gasCosts = load_data(dir)
        
        processor(rounds, participants, gasCosts, outdir)

