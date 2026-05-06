from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
import traceback
import argparse

# Running this file directly puts experiment/ on sys.path.
# Insert src/ and repo root before any project imports so they resolve
# regardless of whether the editable install is present.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

from utils.paths import repo_root
REPO_ROOT = repo_root(Path(__file__))

import experiment.experiment_runner as ExperimentRunner
from itertools import product
from dataclasses import dataclass
from experiment.helper import getPath, create_run_ids
from experiment.experiment_configuration import ExperimentConfiguration
from experiment.experiment_presets import PRESETS
from utils.async_writer import AsyncWriter
from selector import choose_from_list
from analysis import ExperimentLogger

DATA_ROOT = REPO_ROOT / "data"
DATASETSLOW = "cifar.10"
DATASETFAST = "mnist"
RESULTDATAFOLDER = REPO_ROOT / "data" / "runs" / "experiments"

# ---------------- PRESET SEARCH SPACE ----------------

# preset = "test"
preset = "test"
_use_defaults = True
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
    freerider_activation_round: int
    freerider_noise: float
    freerider_attack_type: str
    malicious_activation_round: int | None
    malicious_noise: float | None
    malicious_attack_type: str
    aggregation_rule: str
    data_distribution: str
    dirichlet_alpha: float | None
    run_id : int

skips: list[Skip] = []


# ---------------- MAIN ----------------

def main(author): # single preset
    preset_config = PRESETS[preset]
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    if args.skipFolder is not None:
        parseSkips()

    productVar = []

    malicious_rounds = (
        preset_config.malicious_start_round
        if preset_config.malicious_start_round is not None
        else [None]
    )

    malicious_noises = (
        preset_config.malicious_noise_scale
        if preset_config.malicious_noise_scale is not None
        else [None]
    )

    malicious_attack_types = (
        preset_config.malicious_attack_type
        if preset_config.malicious_attack_type is not None
        else [None]
    )

    freerider_attack_types = (
        preset_config.freerider_attack_type
        if preset_config.freerider_attack_type is not None
        else [None]
    )

    runs = create_run_ids(preset_config.number_of_runs)

    for (
            strategy,
            outlier_detection,
            freerider_round,
            freerider_noise,
            freerider_attack_type,
            malicious_activation_round,
            malicious_noise,
            malicious_attack_type,
            dataset,
            aggregation_rule,
            data_distribution,
            run_id
    ) in product(
        preset_config.contribution_score_strategy,
        preset_config.use_outlier_detection,
        preset_config.freerider_start_round,
        preset_config.freerider_noise_scale,
        freerider_attack_types,
        malicious_rounds,
        malicious_noises,
        malicious_attack_types,
        datasets,
        preset_config.aggregation_rule,
        preset_config.data_distribution,
        runs
    ):

        # fallback to freerider values if any of the malicious params are None
        if malicious_activation_round is None:
            malicious_activation_round = freerider_round
        if malicious_noise is None:
            malicious_noise = freerider_noise
        if malicious_attack_type is None:
            malicious_attack_type = "noise"

        if freerider_attack_type is None:
            freerider_attack_type = "noise"

        # only add dirichlet alpha to the product if the data distribution is dirichlet, otherwise set it to None
        if data_distribution in {"dirichlet_split", "dirichlet_split_42"}:
            for alpha in preset_config.dirichlet_alpha:
                productVar.append((
                    strategy,
                    outlier_detection,
                    freerider_round,
                    freerider_noise,
                    freerider_attack_type,
                    malicious_activation_round,
                    malicious_noise,
                    malicious_attack_type,
                    dataset,
                    aggregation_rule,
                    data_distribution,
                    alpha,
                    run_id
                ))
        else:
            productVar.append((
                strategy,
                outlier_detection,
                freerider_round,
                freerider_noise,
                freerider_attack_type,
                malicious_activation_round,
                malicious_noise,
                malicious_attack_type,
                dataset,
                aggregation_rule,
                data_distribution,
                None,
                run_id
            ))

    total = len(productVar)
    skipsCount = len(skips)

    time = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    for i, (
        strategy,
        outlier_detection,
        freerider_round,
        freerider_noise,
        freerider_attack_type,
        malicious_activation_round,
        malicious_noise,
        malicious_attack_type,
        dataset,
        aggregation_rule,
        data_distribution,
        dirichlet_alpha,
        run_id
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
            freerider_activation_round=freerider_round,
            freerider_noise=freerider_noise,
            freerider_attack_type=freerider_attack_type,
            malicious_activation_round=malicious_activation_round,
            malicious_noise=malicious_noise,
            malicious_attack_type=malicious_attack_type,
            aggregation_rule=aggregation_rule,
            data_distribution=data_distribution,
            dirichlet_alpha=dirichlet_alpha,
            run_id=run_id
        )

        if shouldSkip(skipConfig):
            print(f"Skipping: {skipConfig}")
            continue


        # override parameters for this run
        config.contribution_score_strategy = strategy
        config.use_outlier_detection = outlier_detection
        config.freerider_start_round = freerider_round
        config.freerider_noise_scale = freerider_noise
        config.freerider_attack_type = freerider_attack_type
        config.malicious_start_round = malicious_activation_round if malicious_activation_round is not None else freerider_round
        config.malicious_noise_scale = malicious_noise if malicious_noise is not None else freerider_noise
        config.malicious_attack_type = malicious_attack_type
        config.aggregation_rule = aggregation_rule
        config.data_distribution = data_distribution
        config.dirichlet_alpha = dirichlet_alpha

        path = getPath(config, time, dataset, preset, RESULTDATAFOLDER, run_id=run_id)

        try:

            writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
            metadata = {**vars(config), "dataset": dataset, "timestamp": startTime}
            logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
            ExperimentRunner.run_experiment(dataset, config, run_id, writer, logger)
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


        # experiment.model.visualize_simulation(DATA_ROOT / "figures")

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
            r"(?P<outlierDetection>[^-]+)-"
            r"(?P<activationRound>[^-]+)-"
            r"(?P<noise>[^-]+)-"
            r"(?P<freeriderAttackType>[^-]+)-"
            r"(?P<maliciousRound>[^-]+)-"
            r"(?P<maliciousNoise>[^-]+)-"
            r"(?P<maliciousAttackType>[^-]+)-"
            r"(?P<aggregationRule>[^-]+)"
            r"-(?P<dataDistribution>[^-]+)"
            r"(?:-(?P<dirichletAlpha>[^-]+))"
            r"-?(?P<runId>[0-9]+)?"  # <-- optional run ID part
            r"(?:-\{[0-9a-fA-F-]+\})?"  # <-- optional UUID part
            r"\.pkl",
            file,
        )


        if not m:
            print("Did not match:", file)
            continue

        mal_round = m.group("maliciousRound")
        mal_noise = m.group("maliciousNoise")
        alpha = m.group("dirichletAlpha")


        skips.append(
            Skip(
                preset=m.group("preset"),
                dataset=m.group("dataset"),
                strategy=m.group("strategy"),
                outlier_detection=m.group("outlierDetection") == "True",
                freerider_activation_round=int(m.group("activationRound")),
                freerider_noise=float(m.group("noise")),
                freerider_attack_type=m.group("freeriderAttackType"),
                malicious_activation_round=int(mal_round) if mal_round != "None" else None,
                malicious_noise=float(mal_noise) if mal_noise != "None" else None,
                malicious_attack_type=m.group("maliciousAttackType"),
                aggregation_rule=m.group("aggregationRule"),
                data_distribution=m.group("dataDistribution"),
                dirichlet_alpha=float(alpha) if alpha != "None" else None,
                run_id=int(m.group("runId")) if m.group("runId") is not None else 0
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
        freerider_activation_round=skip.freerider_activation_round,
        freerider_noise=skip.freerider_noise,
        freerider_attack_type=skip.freerider_attack_type,
        malicious_activation_round=(
            skip.malicious_activation_round
            if skip.malicious_activation_round is not None
            else skip.freerider_activation_round
        ),
        malicious_noise=(
            skip.malicious_noise
            if skip.malicious_noise is not None
            else skip.freerider_noise
        ),
        malicious_attack_type=skip.malicious_attack_type,
        aggregation_rule=skip.aggregation_rule,
        data_distribution=skip.data_distribution,
        dirichlet_alpha=skip.dirichlet_alpha if skip.data_distribution in {"dirichlet_split", "dirichlet_split_42"} else None,
        run_id=skip.run_id
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


