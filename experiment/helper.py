import uuid
from datetime import datetime
from pathlib import Path
from experiment_configuration import ExperimentConfiguration


def getPath(config: ExperimentConfiguration, time: str, dataset: str, preset: str, resultDataFolder: Path):
    # Filename for csv and pickle logging files

    filename = (
        f"{preset}-"
        f"{dataset}-"
        f"{config.contribution_score_strategy}-"
        f"{config.freerider_start_round}-"
        f"{config.freerider_noise_scale}-"
        f"{config.malicious_start_round}-"
        f"{config.malicious_noise_scale}-"
        f"{config.use_outlier_detection}-"
        f"{config.aggregation_rule}-"
        f"{{{uuid.uuid4()}}}.csv"
    )

    path = Path(resultDataFolder).joinpath(time).joinpath(filename)

    return path