from datetime import datetime
import sys
import multiprocessing as mp
from pathlib import Path
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from openfl.utils.async_writer import AsyncWriter

# Add the repo root to sys.path so `analysis` package is importable from here
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import ExperimentLogger

config = ExperimentConfiguration(
    min_buy_in=int(1e18),
    max_buy_in=int(1e18),
    contribution_score_strategy="loss_only",
    use_outlier_detection=True,
    minimum_rounds=10,
    freerider_noise_scale=0.1,
    malicious_noise_scale=0.1,
    punish_factor=3,
    punish_factor_contrib=3,
    freerider_start_round=1,
    malicious_start_round=1,
    use_nobody_is_kicked=True,
    force_merge_all=True,
    number_of_good_contributors=4,
    number_of_bad_contributors=1,
    number_of_freerider_contributors=1,
    number_of_inactive_contributors=0,
    aggregation_rule="positives_only"
)
PRESET = "mnist_openfl_w_outlier"
config = ExperimentConfiguration(preset=PRESET, use_defaults=True)

# OVERSKRIV variabler her for testing. eksempel: config = ExperimentConfiguration(minimum_rounds=1), hvis du kun vil køre een round#DATASET = "cifar-10"
RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/sample")
DATASET = "mnist"

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

def main():
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    try:
        path = getPath(config)
        writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
        metadata = {**vars(config), "dataset": DATASET, "timestamp": startTime}
        logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
        experiment = ExperimentRunner.run_experiment(DATASET, config, writer, logger)
        writer.finish()
        logger.save(path.with_suffix(".pkl"))

        experiment.model.visualize_simulation("figures")
        ExperimentRunner.print_transactions(experiment)
    except Exception as e:
        print(f"An error occurred during the experiment: {e}")

    writer.finish()


def getPath(experimentConfig: ExperimentConfiguration):
    time = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    filename = (
        f"{PRESET}-"
        f"{DATASET}-"
        f"{config.contribution_score_strategy}-"
        f"{config.freerider_start_round}-"
        f"{config.freerider_noise_scale}-"
        f"{config.malicious_start_round}-"
        f"{config.malicious_noise_scale}-"
        f"{config.use_outlier_detection}-"
        f"{config.aggregation_rule}.csv"
    )

    path = Path(RESULTDATAFOLDER).joinpath(time).joinpath(filename)

    return path

if __name__ == "__main__":
    mp.freeze_support()
    main()
    for p in mp.active_children():
        #print("Terminating:", p.pid)
        p.terminate()
    print("Done :)")