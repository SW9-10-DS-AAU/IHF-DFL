from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import re
import traceback
from unittest import skip
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from itertools import product
from dataclasses import dataclass
import argparse

from openfl.utils.async_writer import AsyncWriter
from selector import choose_from_list 

DATASETSLOW = "cifar.10"
DATASETFAST = "mnist"
RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/experimentData")

datasets = [ DATASETSLOW ]
#strategy_options = ["accuracy", "naive", "dotproduct"]
strategy_options = ["naive" ]
outlier_detection_options = [ False ]
free_rider_activation_round_options = [1]
free_rider_noise_options = [0.0]


#datasets = [ DATASETSLOW, DATASETFAST ]
##strategy_options = ["accuracy", "naive", "dotproduct"]
#strategy_options = ["naive", "dotproduct"]
#outlier_detection_options = [ True, False ]
#free_rider_activation_round_options = [1, 3, 5]
#free_rider_noise_options = [0.0, 0.1, 0.5, 1.0]

#strategy_options = ["accuracy"]
#free_rider_activation_round_options = [1]
#free_rider_noise_options = [0.1]

OUTPUTHEADERS = [
    "round",
    "time",
    "globalAcc",
    "globalLoss",
    "GRS",
    "accAvgPerUser",
    "lossAvgPerUser",
    "rewards",
    "conctractBalanceRewards",
    "punishments",
    "contributionScores",
    "feedbackMatrix",
    "disqualifiedUsers",
    "userStatuses",
    ]



# OUTPUTHEADERS = [
#     "GRS"
# ]
WRITERBUFFERSIZE = 200

def main(author):
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")
    for strategy, outlier_detection, free_rider_activation_round, free_rider_noise, dataset in product(strategy_options, outlier_detection_options, free_rider_activation_round_options, free_rider_noise_options, datasets):
        # Set up configuration for the experiment run
        if (strategy == "accuracy" and outlier_detection == True or (strategy == "naive" and outlier_detection == True)):
            continue #As accuracy mode always uses making having both on redundent
        
        # Auto skips
        if (args.skipFolder is not None):
            parseSkips()
        if (shouldSkip(Skip(strategy, outlier_detection, free_rider_activation_round, free_rider_noise, dataset))):
            print(f"Skipping: {strategy} {outlier_detection} {free_rider_activation_round} {free_rider_activation_round} {dataset}")
            continue

        config = ExperimentConfiguration(
            fork=True,
            min_buy_in=int(1e18),
            max_buy_in=int(1e18),
            contribution_score_strategy = strategy,
            use_outlier_detection = outlier_detection,
            freerider_start_round = free_rider_activation_round,
            freerider_noise_scale = free_rider_noise,
        )
        
        path = getPath(config, startTime, dataset)
        try:
            writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
            experiment = ExperimentRunner.run_experiment(dataset, config, writer)
            writer.finish()
        except Exception as e:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            err_file = path.parent / f"error-{ts}.txt"

            config_as_dict = config.__dict__

            text  = "CONFIG:\n"
            text += json.dumps(config_as_dict, indent=4, default=str)
            text += "\n\nEXCEPTION:\n"
            text += "".join(traceback.format_exception(e))

            with open(err_file, "w") as f:
                f.write(text)

            print(f"Error logged to: {err_file}")

        #experiment.model.visualize_simulation("experiment/figures")

        #ExperimentRunner.print_transactions(experiment)

def getPath(experimentConfig: ExperimentConfiguration, time: datetime, dataset):

    filename = f"{dataset}-{experimentConfig.contribution_score_strategy}-{experimentConfig.freerider_start_round}-{experimentConfig.freerider_noise_scale}-{experimentConfig.use_outlier_detection}.csv"

    path = Path(RESULTDATAFOLDER).joinpath(time).joinpath(filename)

    return path

parser = argparse.ArgumentParser()
parser.add_argument("--skipFolder", type=str)
parser.add_argument("--author", type=str)
args = parser.parse_args()

@dataclass(frozen=True)
class Skip:
    strategy: str
    outlier_detection: bool
    free_rider_activation_round: int
    free_rider_noise: float
    dataset: str

skips: list[Skip] = []

def parseSkips():
    if not Path.exists(RESULTDATAFOLDER):
        return

    dirs = sorted([d for d in RESULTDATAFOLDER.iterdir() if d.is_dir()])
    chosenDirs =choose_from_list(dirs, "IDK man", False)

    files: list[Path] = []
    for dir in chosenDirs:
        files.extend([p.name for p in Path(dir).iterdir() if p.is_file()])
    
    for file in files:
        m = re.fullmatch(
            r"(?P<dataset>[^-]+)-(?P<strategy>[^-]+)-(?P<activationRound>[^-]+)-"
            r"(?P<noise>[^-]+)-(?P<outlierDetection>[^-]+)\.csv", file)
        m2 = re.fullmatch(
            r"(?P<strategy>[^-]+)-(?P<activationRound>[^-]+)-"
            r"(?P<noise>[^-]+)-(?P<outlierDetection>[^-]+)\.csv", file)
        if m:
            dataset = m.group("dataset")
            groups = m
        elif m2:
            dataset = DATASETSLOW
            groups = m2
        else:
            print("Did not match")
            continue
        
        print("FoundSkip")
        skips.append(
            Skip(
                strategy=groups.group("strategy"),
                outlier_detection=groups.group("outlierDetection") == "True",
                free_rider_activation_round=int(groups.group("activationRound")),
                free_rider_noise=float(groups.group("noise")),
                dataset=dataset,
            )
        )

def shouldSkip(config: Skip):
    print(len(skips))
    for skip in skips:
        if skip == config:
            print("Skipping")
            return True
    print("not skipping")
    print(skip)
    print(config)
    return False

if __name__ == "__main__":
    author = args.author if args.author is not None else input("Author?\n")
    mp.freeze_support()
    main(author)
    for p in mp.active_children():
        print("Terminating:", p.pid)
        p.terminate()
    print("main finished")

