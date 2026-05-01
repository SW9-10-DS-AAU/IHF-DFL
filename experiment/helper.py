import uuid
from datetime import datetime
from pathlib import Path
from experiment.experiment_configuration import ExperimentConfiguration


def getPath(config: ExperimentConfiguration, time: str, dataset: str, preset: str, resultDataFolder: Path,
            run_id: int = 0) -> Path:
    # Filename for csv and pickle logging files
    parts = [
        preset,
        dataset,
        config.contribution_score_strategy,
        config.use_outlier_detection,
        config.freerider_start_round,
        config.freerider_noise_scale,
        config.freerider_attack_type,
        config.malicious_start_round,
        config.malicious_noise_scale,
        config.malicious_attack_type,
        config.aggregation_rule,
        config.data_distribution,
        config.dirichlet_alpha,
    ]

    if run_id != 0:
        parts.append(run_id)

    filename = "-".join(map(str, parts)) + "{" + str(uuid.uuid4()) + "}" + ".csv"

    path = Path(resultDataFolder).joinpath(time).joinpath(filename)

    return path



def create_run_ids (number_of_runs: int) -> list[int]:
    run_ids = []
    run_id = 1
    while run_id <= number_of_runs:
        run_ids.append(run_id)
        run_id += 1
    return run_ids
