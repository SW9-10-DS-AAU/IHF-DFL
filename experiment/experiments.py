from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from itertools import product

from openfl.utils.async_writer import AsyncWriter 

#DATASET = "cifar-10"
DATASET = "mnist"
RESULTDATAFOLDER = "/home/wired/dev/openFL-2.0/experiment/data"

#strategy_options = ["accuracy", "legacy", "None"]
#outlier_detection_options = ["mad"]
#outlier_detection_options = ["none"]
#free_rider_activation_round_options = [1, 3, 5]
#free_rider_noise_options = [0.0, 0.1, 0.5, 1.0]

strategy_options = ["accuracy"]
outlier_detection_options = ["mad", "None"]
free_rider_activation_round_options = [1]
free_rider_noise_options = [0.1]

OUTPUTHEADERS = [
    "round",
    "time",
    "globalAcc",
    "GRS",
    "accAvgPerUser",
    "lossAvgPerUser",
    "rewards",
    "conctractBalanceRewards",
    "punishments",
    "contributionScores",
    "feedbackMatrix",
    "disqualifiedUsers",
    ]
WRITERBUFFERSIZE = 200

def main(author):
    for strategy, outlier_detection, free_rider_activation_round, free_rider_noise in product(strategy_options, outlier_detection_options, free_rider_activation_round_options, free_rider_noise_options):
        # Set up configuration for the experiment run
        if (strategy == "accuracy" and outlier_detection == "mad"):
            continue #As accuracy mode always uses making having both on redundent
        config = ExperimentConfiguration(
            fork=True,
            min_buy_in=int(1e18),
            max_buy_in=int(1e18),
            contribution_score_strategy = strategy,
            use_outlier_detection = outlier_detection,
            freerider_start_round = free_rider_activation_round,
            freerider_noise_scale = free_rider_noise,
        )
        
        path = getPath(config)
        
        writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
        experiment = ExperimentRunner.run_experiment(DATASET, config, writer)

        #experiment.model.visualize_simulation("experiment/figures")

        #ExperimentRunner.print_transactions(experiment)
        writer.finish()


def getPath(experimentConfig: ExperimentConfiguration):
    
    time = datetime.now().strftime("%d-%m-%y--%H:%M:%S")

    filename = f"{experimentConfig.contribution_score_strategy}-{experimentConfig.freerider_start_round}-{experimentConfig.freerider_noise_scale}-{experimentConfig.use_outlier_detection}.csv"

    path = Path(RESULTDATAFOLDER).joinpath(time).joinpath(filename)

    return path


if __name__ == "__main__":
    author = input("Author?\n")
    mp.freeze_support()
    main(author)