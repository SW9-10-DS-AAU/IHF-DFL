from datetime import datetime
import sys
import multiprocessing as mp
from pathlib import Path

# Running this file directly puts experiment/ on sys.path.
# Insert src/ and repo root before any project imports so they resolve
# regardless of whether the editable install is present.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

from utils.paths import repo_root
REPO_ROOT = repo_root(Path(__file__))

import experiment.experiment_runner as ExperimentRunner
from experiment.experiment_configuration import ExperimentConfiguration
from experiment.helper import getPath, resolve_attack_params
from utils.async_writer import AsyncWriter

from analysis import ExperimentLogger

DATA_ROOT = REPO_ROOT / "data"

preset = "test"
_use_defaults = True


config = ExperimentConfiguration(preset=preset, use_defaults=_use_defaults)

DATASETSLOW = "cifar.10"
DATASETFAST = "mnist"
RESULTDATAFOLDER = REPO_ROOT / "data" / "runs" / "sample"

DATASET = DATASETFAST

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

(
    config.freerider_start_round,
    config.freerider_noise_scale,
    # config.freerider_attack_type,
    config.malicious_start_round,
    config.malicious_noise_scale,
    # config.malicious_attack_type,
) = resolve_attack_params(
    has_bad=config.number_of_bad_contributors > 0,
    has_freerider=config.number_of_freerider_contributors > 0,
    freerider_round=config.freerider_start_round,
    freerider_noise=config.freerider_noise_scale,
    # freerider_attack_type=config.freerider_attack_type,
    malicious_activation_round=config.malicious_start_round,
    malicious_noise=config.malicious_noise_scale,
    # malicious_attack_type=config.malicious_attack_type,
    warn=True,
)



def main():
    time = datetime.now().strftime("%d-%m-%y--%H_%M_%S")
    #
    # try:
    path = getPath(config, time, DATASET, preset, RESULTDATAFOLDER, run_id=0)
    writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
    metadata = {**vars(config), "dataset": DATASET, "timestamp": time}
    logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
    experiment = ExperimentRunner.run_experiment(DATASET, config, 0, writer, logger)
    writer.finish()
    logger.save(path.with_suffix(".pkl"))

    # experiment.model.visualize_simulation(DATA_ROOT / "figures")
    ExperimentRunner.print_transactions(experiment)
    # except Exception as e:
    #     print(f"An error occurred during the experiment: {e}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
    print("Done :)")
