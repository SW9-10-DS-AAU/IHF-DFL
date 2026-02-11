from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import re
import traceback
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

datasets = [ DATASETFAST ]
#strategy_options = ["dotproduct", "naive", "accuracy"]
strategy_options = [ "accuracy" ]
outlier_detection_options = [ True, False ]
free_rider_activation_round_options = [1, 3, 5]
#malicious_activation_round_options = [1, 3, 5]
free_rider_noise_options = [1.0, 0.5, 0.1, 0.01, 0.0]
#malicious_noise_options = [1.0, 0.5, 0.05, 0.01]
#forced_ones = [ True, False ]
forced_ones = [ False ]

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
    "GasTransactions",
    ]

# OUTPUTHEADERS = [
#     "GRS"
# ]
WRITERBUFFERSIZE = 200
@dataclass(frozen=True)
class Skip:
    strategy: str
    outlier_detection: bool
    free_rider_activation_round: int
    free_rider_noise: float
    malicious_activation_round: int
    malicious_noise: float
    dataset: str
skips: list[Skip] = []

def main(author):
    global skips
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    if (args.skipFolder is not None):
        parseSkips()

    oldProduct = product(
        strategy_options,
        outlier_detection_options,
        free_rider_activation_round_options,
        free_rider_noise_options,
        #malicious_activation_round_options,
        #malicious_noise_options,
        datasets,
        forced_ones)
    
    productVar = [
        (strategy, outlier_detection, free_rider_activation_round, free_rider_noise, dataset, forced)
        for (strategy, outlier_detection, free_rider_activation_round, free_rider_noise, dataset, forced)
        in oldProduct
        if not (outlier_detection and strategy in {"accuracy", "naive"})
    ]
    total = len(productVar)
    skipsCount = len(skips)

    for i, (strategy, outlier_detection, free_rider_activation_round, free_rider_noise, dataset, forced) in enumerate(productVar, start=1):
        #malicious_activation_round, malicious_noise, -- removed
        malicious_activation_round = free_rider_activation_round
        malicious_noise = free_rider_noise
        
        
        # Auto skips
        if (shouldSkip(Skip(strategy, outlier_detection, free_rider_activation_round, free_rider_noise, malicious_activation_round, malicious_noise, dataset))):
            print(f"Skipping: {strategy} {outlier_detection} {free_rider_activation_round} {free_rider_activation_round} {malicious_activation_round} {malicious_noise} {dataset}")
            continue

        config = ExperimentConfiguration(
            fork=True,
            min_buy_in=int(1e18),
            max_buy_in=int(1e18),
            contribution_score_strategy = strategy,
            use_outlier_detection = outlier_detection,
            freerider_start_round = free_rider_activation_round,
            freerider_noise_scale = free_rider_noise,
            malicious_start_round = malicious_activation_round,
            malicious_noise_scale = malicious_noise,
            force_merge_all = forced
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
        percent = (i / total) * 100
        totalPercent = ((i + skipsCount) / total) * 100
        print(f"Progress: {i}/{total} ({percent:.2f}%)\nTotal Progress: {i + skipsCount}/{total} ({totalPercent:.2f}%)\n", flush=True)

def getPath(experimentConfig: ExperimentConfiguration, time: datetime, dataset):

    filename = f"{dataset}-{experimentConfig.contribution_score_strategy}-{experimentConfig.freerider_start_round}-{experimentConfig.freerider_noise_scale}-{experimentConfig.malicious_start_round}-{experimentConfig.malicious_noise_scale}-{experimentConfig.use_outlier_detection}-{experimentConfig.force_merge_all}.csv"

    path = Path(RESULTDATAFOLDER).joinpath(time).joinpath(filename)

    return path

parser = argparse.ArgumentParser()
parser.add_argument("--skipFolder", type=str)
parser.add_argument("--author", type=str)
args = parser.parse_args()

def parseSkips():
    global skips
    if not Path.exists(RESULTDATAFOLDER):
        return

    dirs = sorted([d for d in RESULTDATAFOLDER.iterdir() if d.is_dir()])
    chosenDirs = choose_from_list(dirs, "Select directories to scan for configs to skip", False)

    files: list[Path] = []
    for dir in chosenDirs:
        files.extend([p.name for p in Path(dir).iterdir() if p.is_file()])
    
    for file in files:
        m = re.fullmatch(
            r"(?P<dataset>[^-]+)-"
            r"(?P<strategy>[^-]+)-"
            r"(?P<activationRound>[^-]+)-"
            r"(?P<noise>[^-]+)-"
            r"(?P<maliciousRound>[^-]+)-"
            r"(?P<maliciousNoise>[^-]+)-"
            r"(?P<outlierDetection>[^-]+)-"
            r"(?P<forced>[^-]+)\.csv",
            file
        )
        if m:
            dataset = m.group("dataset")
            groups = m
        else:
            print("Did not match")
            continue
        
        skips.append(
            Skip(
                strategy=groups.group("strategy"),
                outlier_detection=groups.group("outlierDetection") == "True",
                free_rider_activation_round=int(groups.group("activationRound")),
                free_rider_noise=float(groups.group("noise")),
                malicious_activation_round=int(groups.group("maliciousRound")),
                malicious_noise=float(groups.group("maliciousNoise")),
                dataset=dataset,
            )
        )

def shouldSkip(config: Skip):
    for skip in skips:
        if skip == config:
            print("Skipping")
            return True
    print("not skipping")
    return False

if __name__ == "__main__":
    author = args.author if args.author is not None else input("Author?\n")
    mp.freeze_support()
    main(author)
    for p in mp.active_children():
        print("Terminating:", p.pid)
        p.terminate()
    print("Done :)")