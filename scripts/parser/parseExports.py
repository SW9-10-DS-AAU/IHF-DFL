from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from typing import Any, Callable

from openfl.ml.pytorch_model import Participant
from .experiment_specs import ExperimentSpec
from .gasCosts import GasStats
from .types.round import Round
from .parser import load_data
from .selector import choose_from_list

chosenTests = []

def runProcessor(RESULTDATAFOLDER: Path, useSameTests: bool, processor: Callable[[list[Round], list[Participant], ExperimentSpec, GasStats, str], Any]):
    global chosenTests
    if not chosenTests or not useSameTests:
        dirs = sorted([d for d in RESULTDATAFOLDER.iterdir() if d.is_dir()])

        chosenDirs = choose_from_list(dirs, "Chose test run(s)", False)

        files: list[Path] = []
        for d in chosenDirs:
            files.extend([p for p in d.iterdir() if p.is_file() and p.suffix == ".csv"])

        tests = sorted(files)

        chosenTests = choose_from_list(tests, "Chose tests to process")


    for test in chosenTests:
        outdir = (
            RESULTDATAFOLDER.parent
            / "compiledData"
            / test.parent.name
            / test.stem
        )

        rounds, participants, gasCosts, experimentConfig = load_data(test)

        disqualifiedUsers = _detect_disqualifications(rounds, participants)

        for roundNr, participant in disqualifiedUsers:
            rounds[roundNr].addDisqualifiedUser(participant)

        processor(rounds, participants, experimentConfig, gasCosts, outdir)


def _detect_disqualifications(rounds: list[Round], participants: dict[int, Participant]):
    disqualified = []

    # total_rounds = len(rounds)
    markedIndexes = []
    for round in rounds:
        for i, userGrs in enumerate(round.GRS):
            if (userGrs == 0 and i not in markedIndexes):
                markedIndexes.append(i)
                disqualified.append((round.nr, participants[i]))

    return disqualified