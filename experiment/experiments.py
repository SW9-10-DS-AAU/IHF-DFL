from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
import traceback
from itertools import product
from dataclasses import dataclass
from helper import getPath
import argparse
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from experiment_presets import PRESETS
from openfl.utils.async_writer import AsyncWriter
from selector import choose_from_list

# Add the repo root to sys.path so `analysis` package is importable from here
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import ExperimentLogger

DATASETSLOW = "cifar.10"
DATASETFAST = "mnist"
RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/experimentData")

# ---------------- PRESET SEARCH SPACE ----------------

# preset = "test"
preset = "aggregation_rules_test_model_performance_people_get_kicked_now_mnist"
_use_defaults = False
datasets = [ DATASETFAST ]


# ---------------- OUTPUT ----------------

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

WRITERBUFFERSIZE = 200


# ---------------- SKIP STRUCTURE ----------------

@dataclass(frozen=True)
class Skip:
    preset: str
    dataset: str
    strategy: str
    outlier_detection: bool
    free_rider_activation_round: int
    free_rider_noise: float
    malicious_activation_round: int | None
    malicious_noise: float | None
    aggregation_rule: str

skips: list[Skip] = []


# ---------------- MAIN ----------------

def main(author): # single preset
    preset_config = PRESETS[preset]
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    if args.skipFolder is not None:
        parseSkips()

    oldProduct = product(
        preset_config.contribution_score_strategy,
        preset_config.use_outlier_detection,
        preset_config.freerider_start_round,
        preset_config.freerider_noise_scale,
        preset_config.malicious_start_round if preset_config.malicious_start_round is not None else [None],
        preset_config.malicious_noise_scale if preset_config.malicious_noise_scale is not None else [None],
        datasets,
        preset_config.aggregation_rule,
    )

    productVar = list(oldProduct)

    total = len(productVar)
    skipsCount = len(skips)

    time = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    for i, (
        strategy,
        outlier_detection,
        freerider_round,
        freerider_noise,
        malicious_activation_round,
        malicious_noise,
        dataset,
        aggregation_rule,
    ) in enumerate(productVar, start=1):

        progress_bar(i - 1, skipsCount, total)

        config = ExperimentConfiguration(preset=preset, use_defaults=_use_defaults)
        config.preset_name = preset
        config.dataset = dataset

        skipConfig = Skip(
            preset=preset,
            dataset=dataset,
            strategy=strategy,
            outlier_detection=outlier_detection,
            free_rider_activation_round=freerider_round,
            free_rider_noise=freerider_noise,
            malicious_activation_round=malicious_activation_round,
            malicious_noise=malicious_noise,
            aggregation_rule=aggregation_rule,
        )

        if shouldSkip(skipConfig):
            print(f"Skipping: {skipConfig}")
            continue


        # override parameters for this run
        config.contribution_score_strategy = strategy
        config.use_outlier_detection = outlier_detection
        config.freerider_start_round = freerider_round
        config.freerider_noise_scale = freerider_noise
        config.malicious_start_round = malicious_activation_round if malicious_activation_round is not None else freerider_round
        config.malicious_noise_scale = malicious_noise if malicious_noise is not None else freerider_noise
        config.aggregation_rule = aggregation_rule

        path = getPath(config, time, dataset, preset, RESULTDATAFOLDER)

        try:

            writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
            metadata = {**vars(config), "dataset": dataset, "timestamp": startTime}
            logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
            ExperimentRunner.run_experiment(dataset, config, writer, logger)
            writer.finish()

            logger.save(path.with_suffix(".pkl"))
        except Exception as e:

            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            err_file = path.parent / f"error-{ts}.txt"

            config_as_dict = config.__dict__

            text = "CONFIG:\n"
            text += json.dumps(config_as_dict, indent=4, default=str)
            text += "\n\nEXCEPTION:\n"
            text += "".join(traceback.format_exception(e))

            with open(err_file, "w") as f:
                f.write(text)

            print(f"Error logged to: {err_file}")

        #experiment.model.visualize_simulation("experiment/figures")

    #ExperimentRunner.print_transactions(experiment)


# ---------------- ARGUMENTS ----------------

parser = argparse.ArgumentParser()
parser.add_argument("--skipFolder", type=str)
parser.add_argument("--author", type=str)
args = parser.parse_args()


# ---------------- SKIP PARSING ----------------

def parseSkips():
    if not RESULTDATAFOLDER.exists():
        return

    dirs = sorted([d for d in RESULTDATAFOLDER.iterdir() if d.is_dir()])

    chosenDirs = choose_from_list(
        dirs,
        "Select directories to scan for configs to skip",
        False
    )

    files: list[str] = []

    for dir in chosenDirs:
        files.extend([p.name for p in dir.iterdir() if p.is_file()])

    for file in files:
        m = re.fullmatch(
            r"(?P<preset>[^-]+)-"
            r"(?P<dataset>[^-]+)-"
            r"(?P<strategy>[^-]+)-"
            r"(?P<activationRound>[^-]+)-"
            r"(?P<noise>[^-]+)-"
            r"(?P<maliciousRound>[^-]+)-"
            r"(?P<maliciousNoise>[^-]+)-"
            r"(?P<outlierDetection>[^-]+)-"
            r"(?P<aggregationRule>[^-]+)"
            r"(?:-\{[0-9a-fA-F-]+\})?"  # <-- optional UUID part
            r"\.csv",
            file,
        )


        if not m:
            print("Did not match:", file)
            continue

        mal_round = m.group("maliciousRound")
        mal_noise = m.group("maliciousNoise")


        skips.append(
            Skip(
                preset=m.group("preset"),
                dataset=m.group("dataset"),
                strategy=m.group("strategy"),
                outlier_detection=m.group("outlierDetection") == "True",
                free_rider_activation_round=int(m.group("activationRound")),
                free_rider_noise=float(m.group("noise")),
                malicious_activation_round=int(mal_round) if mal_round != "None" else None,
                malicious_noise=float(mal_noise) if mal_noise != "None" else None,
                aggregation_rule=m.group("aggregationRule"),
            )
        )


# ---------------- SKIP CHECK ----------------

def shouldSkip(config: Skip):
    config = normalize_skip(config)
    for skip in skips:
        if normalize_skip(skip) == config:
            return True
    return False


# ---------------- PROGRESS ----------------

def progress_bar(i, skipsCount, total):

    percent = (i / total) * 100
    totalPercent = ((i + skipsCount) / total) * 100

    print(
        f"Progress: {i}/{total} ({percent:.2f}%)\n"
        f"Total Progress: {i + skipsCount}/{total} ({totalPercent:.2f}%)\n",
        flush=True,
    )



def normalize_skip(skip: Skip):
    return Skip(
        preset=skip.preset,
        dataset=skip.dataset,
        strategy=skip.strategy,
        outlier_detection=skip.outlier_detection,
        free_rider_activation_round=skip.free_rider_activation_round,
        free_rider_noise=skip.free_rider_noise,
        malicious_activation_round=(
            skip.malicious_activation_round 
            if skip.malicious_activation_round is not None 
            else skip.free_rider_activation_round
        ),
        malicious_noise=(
            skip.malicious_noise 
            if skip.malicious_noise is not None 
            else skip.free_rider_noise
        ),
        aggregation_rule=skip.aggregation_rule,
    )


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    author = args.author if args.author is not None else input("Author?\n")
    mp.freeze_support()
    main(author)
    for p in mp.active_children():
        print("Terminating:", p.pid)
        p.terminate()
    print("Done :)")