import uuid
from pathlib import Path
from experiment.experiment_configuration import ExperimentConfiguration
from utils.colors import red


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
        config.malicious_start_round,
        config.malicious_noise_scale,
    ]

    if run_id != 0:
        parts.append(run_id)

    filename = "-".join(map(str, parts)) + "-{" + str(uuid.uuid4()) + "}" + ".csv"

    path = Path(resultDataFolder).joinpath(time).joinpath(filename)

    return path



def create_run_ids (number_of_runs: int) -> list[int]:
    run_ids = []
    run_id = 1
    while run_id <= number_of_runs:
        run_ids.append(run_id)
        run_id += 1
    return run_ids


def _warn(message: str, enable: bool) -> None:
    if enable:
        print(red(f"[WARN] {message}"))


def resolve_attack_params(
        has_bad: bool,
        has_freerider: bool,
        freerider_round: int | None,
        freerider_noise: float | None,
        malicious_activation_round: int | None,
        malicious_noise: float | None,
        warn: bool = False,
) -> tuple[int | None, float | None , int | None, float | None]:
    # Only when both malicious and freerider users exist, allow one-way
    # cross-fallback from freerider params to malicious params.
    if has_bad and has_freerider:
        if malicious_activation_round is None:
            _warn(
                f"malicious_start_round is None; using freerider_start_round={freerider_round}",
                warn,
            )
            malicious_activation_round = freerider_round
        if malicious_noise is None:
            _warn(
                f"malicious_noise_scale is None; using freerider_noise_scale={freerider_noise}",
                warn,
            )
            malicious_noise = freerider_noise

    # Apply role-local hard defaults only for roles that are present.
    # If a role count is zero, its params are intentionally left untouched.
    if has_bad:
        if malicious_activation_round is None:
            _warn("malicious_start_round unresolved; forcing default=1", warn)
            malicious_activation_round = 1
        if malicious_noise is None:
            _warn("malicious_noise_scale unresolved; forcing default=0.0", warn)
            malicious_noise = 0.0

    if has_freerider:
        if freerider_round is None:
            _warn("freerider_start_round unresolved; forcing default=1", warn)
            freerider_round = 1
        if freerider_noise is None:
            _warn("freerider_noise_scale unresolved; forcing default=0.0", warn)
            freerider_noise = 0.0

    # Return normalized values in the same order expected by callers.
    return (
        freerider_round,
        freerider_noise,
        malicious_activation_round,
        malicious_noise,
    )
