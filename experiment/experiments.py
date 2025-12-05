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

#strategy_options = ["naive", "legacy", "mad"]
#outlier_detection_options = ["mad", "none"]
#free_rider_activation_round_options = [1, 3, 5]
#free_rider_noise_options = [0.0, 0.1, 0.5, 1.0]

strategy_options = ["naive", "legacy", "mad"]
outlier_detection_options = ["mad", "none"]
free_rider_activation_round_options = [1]
free_rider_noise_options = [0.1]

OUTPUTHEADERS = [
    "time",
    "round",
    "globalAcc",
    "globalLoss",
    "GRS",
    ]
WRITERBUFFERSIZE = 200

def main():
    author = input("Author?\n")

    for strategy, outlier_detection, free_rider_activation_round, free_rider_noise in product(strategy_options, outlier_detection_options, free_rider_activation_round_options, free_rider_noise_options):
        # Set up configuration for the experiment run
        config = ExperimentConfiguration(
            fork=True,
            min_buy_in=int(1e18),
            max_buy_in=int(1e18),
            contribution_score_strategy = strategy,
            #outlier_detection
            #free_rider_activation_round
            #free_rider_noise
        )
        
        path = getPath(config, outlier_detection)
        
        writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
        experiment = ExperimentRunner.run_experiment(DATASET, config, writer)

        experiment.model.visualize_simulation("experiment/figures")

        #ExperimentRunner.print_transactions(experiment)

        writer.finish()


def getPath(experimentConfig: ExperimentConfiguration,extra):
    
    time = datetime.now().strftime("%d-%m-%y--%H:%M")

    filename = f"{time}_{experimentConfig.contribution_score_strategy}-{extra}.csv"

    path = Path(RESULTDATAFOLDER).joinpath(filename)

    return path

def print_config(cfg):
    fields = [
        ("good_contributors", cfg.number_of_good_contributors, "honest participants"),
        ("bad_contributors", cfg.number_of_bad_contributors, "malicious participants"),
        ("freeriders", cfg.number_of_freerider_contributors, "contribute 0"),
        ("inactive", cfg.number_of_inactive_contributors, "never join"),
        ("reward", cfg.reward, "total reward pool"),
        ("minimum_rounds", cfg.minimum_rounds, "rounds to simulate"),
        ("min_buy_in", cfg.min_buy_in, "lower buy-in bound"),
        ("max_buy_in", cfg.max_buy_in, "upper buy-in bound"),
        ("standard_buy_in", cfg.standard_buy_in, "default buy-in"),
        ("epochs", cfg.epochs, "local epochs per round"),
        ("batch_size", cfg.batch_size, "training batch size"),
        ("punish_factor", cfg.punish_factor, "penalty multiplier"),
        ("first_round_fee", cfg.first_round_fee, "fee for first round"),
        ("fork", cfg.fork, "True=local fork, False=real net"),
        ("contribution_score_strategy", cfg.contribution_score_strategy, "scoring method"),
    ]
    for name, value, desc in fields:
        print(f"{name}: {value} ({desc})")


if __name__ == "__main__":
    mp.freeze_support()
    main()