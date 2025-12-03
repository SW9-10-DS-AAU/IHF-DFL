import multiprocessing as mp
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from itertools import product 

#DATASET = "cifar-10"
DATASET = "mnist"

strategy_options = [
    "naive",
    "legacy",
    "mad"
]

outlider_detection = ["mad", "none"]

free_rider_activation_round = [1, 3, 5]
free_rider_notice = [0.0, 0.1, 0.5, 1.0]
# max_buy_in = min_buy_in

def main():
    for strategy in product(strategy_options, outlider_detection, free_rider_activation_round, free_rider_notice):
        # Set up configuration for the experiment run
        config = ExperimentConfiguration(
            fork=True,
        )

        config.contribution_score_strategy = strategy[0]

        print_config(config)

        #experiment = ExperimentRunner.run_experiment(DATASET, config)

        #experiment.model.visualize_simulation("experiment/figures")

        #ExperimentRunner.print_transactions(experiment)


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