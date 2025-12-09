from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from openfl.utils.async_writer import AsyncWriter

config = ExperimentConfiguration() # OVERSKRIV variabler her for testing. eksempel: config = ExperimentConfiguration(minimum_rounds=1), hvis du kun vil køre een round

#DATASET = "cifar-10"
RESULTDATAFOLDER = "/home/wired/dev/openFL-2.0/experiment/data"
DATASET = "mnist"

OUTPUTHEADERS = [
    "time",
    "round",
    "accAvgPerUser",
    "globalAcc",
    "GRS",
    "conctractBalanceRewards",
    "rewards"
    "punishments",
    "lossAvgPerUser"
    ]
WRITERBUFFERSIZE = 200

def main():
    path = getPath(config)
    writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "Schnyks")

    experiment = ExperimentRunner.run_experiment(DATASET, config, writer)

    experiment.model.visualize_simulation("experiment/figures")

    ExperimentRunner.print_transactions(experiment)

    writer.finish()

def getPath(experimentConfig: ExperimentConfiguration):
    
    time = datetime.now().strftime("%d-%m-%y--%H:%M")

    filename = f"{experimentConfig.contribution_score_strategy}-{experimentConfig.freerider_start_round}-{experimentConfig.freerider_noise_scale}.csv"

    path = Path(RESULTDATAFOLDER).joinpath(time).joinpath(filename)

    return path

if __name__ == "__main__":
    mp.freeze_support()
    main()