from datetime import datetime
import sys
import multiprocessing as mp
from pathlib import Path
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from helper import getPath
from openfl.utils.async_writer import AsyncWriter

# Add the repo root to sys.path so `analysis` package is importable from here
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import ExperimentLogger

preset = "data_distribution_mnist"
_use_defaults = False
# preset = "test"

config = ExperimentConfiguration(preset=preset, use_defaults=_use_defaults)

DATASETSLOW = "cifar.10"
DATASETFAST = "mnist"
RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/sample")

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

if config.malicious_noise_scale is None:
    config.malicious_noise_scale = config.freerider_noise_scale
if config.malicious_start_round is None:
    config.malicious_start_round = config.freerider_start_round


def main():
    time = datetime.now().strftime("%d-%m-%y--%H_%M_%S")
    #
    # try:
    path = getPath(config, time, DATASET, preset, RESULTDATAFOLDER)
    writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
    metadata = {**vars(config), "dataset": DATASET, "timestamp": time}
    logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
    experiment = ExperimentRunner.run_experiment(DATASET, config, writer, logger)
    writer.finish()
    logger.save(path.with_suffix(".pkl"))

    experiment.model.visualize_simulation("figures")
    ExperimentRunner.print_transactions(experiment)
    # except Exception as e:
    #     print(f"An error occurred during the experiment: {e}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
    for p in mp.active_children():
        #print("Terminating:", p.pid)
        p.terminate()
    print("Done :)")