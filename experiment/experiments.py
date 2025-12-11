from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import traceback
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from itertools import product

from openfl.utils.async_writer import AsyncWriter 

DATASET = "cifar-10"
#DATASET = "mnist"
RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/experimentData")

#strategy_options = ["accuracy", "naive", "dotproduct"]
strategy_options = ["naive", "dotproduct"]
outlier_detection_options = [ True, False ]
free_rider_activation_round_options = [1, 3, 5]
free_rider_noise_options = [0.0, 0.1, 0.5, 1.0]

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
    for strategy, outlier_detection, free_rider_activation_round, free_rider_noise in product(strategy_options, outlier_detection_options, free_rider_activation_round_options, free_rider_noise_options):
        # Set up configuration for the experiment run
        if (strategy == "accuracy" and outlier_detection == True or (strategy == "naive" and outlier_detection == True)):
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
        
        path = getPath(config, startTime)
        try:
            writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
            experiment = ExperimentRunner.run_experiment(DATASET, config, writer)
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


def getPath(experimentConfig: ExperimentConfiguration, time: datetime):

    filename = f"{experimentConfig.contribution_score_strategy}-{experimentConfig.freerider_start_round}-{experimentConfig.freerider_noise_scale}-{experimentConfig.use_outlier_detection}.csv"

    path = Path(RESULTDATAFOLDER).joinpath(time).joinpath(filename)

    return path


if __name__ == "__main__":
    author = input("Author?\n")
    mp.freeze_support()
    main(author)
    for p in mp.active_children():
        print("Terminating:", p.pid)
        p.terminate()
    print("main finished")