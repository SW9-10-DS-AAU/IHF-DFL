from dataclasses import dataclass
from itertools import product

from experiment.experiment_presets import PRESETS
from experiment.helper import create_run_ids


@dataclass(frozen=True)
class RunSpec:
    strategy: str
    outlier_detection: bool
    freerider_round: int | None
    freerider_noise: float | None
    malicious_round: int | None
    malicious_noise: float | None
    dataset: str
    run_id: int


def _values_or_none(values: list | None) -> list:
    # A preset that leaves a value out still needs one grid cell for it.
    return values if values is not None else [None]


def expand_preset(preset_name: str, dataset: str) -> list[RunSpec]:
    """Expand a preset into the per-run configurations of its grid.

    Lives outside the driver because the driver parses the command line on import.
    """
    preset_config = PRESETS[preset_name]

    number_of_runs = preset_config.number_of_runs if preset_config.number_of_runs is not None else 1
    runs = create_run_ids(number_of_runs)
    if not runs:
        runs = [1]

    return [
        RunSpec(*values)
        for values in product(
            preset_config.contribution_score_strategy,
            preset_config.use_outlier_detection,
            _values_or_none(preset_config.freerider_start_round),
            _values_or_none(preset_config.freerider_noise_scale),
            _values_or_none(preset_config.malicious_start_round),
            _values_or_none(preset_config.malicious_noise_scale),
            [dataset],
            runs,
        )
    ]
