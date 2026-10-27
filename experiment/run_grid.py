import argparse
import subprocess
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from experiment.experiment_configuration import ExperimentConfiguration, require_malicious_values
from experiment.experiment_presets import PRESETS, ExperimentPreset
from experiment.helper import create_run_ids

DATASETS = ["mnist", "cifar.10"]


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


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool
    # Why the tree counts as dirty, for the refusal message. Not part of the recorded state.
    reason: str = field(default="", compare=False)


@dataclass(frozen=True)
class PlannedRun:
    spec: RunSpec
    config: ExperimentConfiguration
    metadata: dict


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
        RunSpec(
            strategy,
            outlier_detection,
            freerider_round,
            freerider_noise,
            # A preset that leaves the malicious values out reuses this cell's free-rider values;
            # a None here would mean the malicious participant never activates.
            malicious_round if malicious_round is not None else freerider_round,
            malicious_noise if malicious_noise is not None else freerider_noise,
            dataset,
            run_id,
        )
        for strategy, outlier_detection, freerider_round, freerider_noise, malicious_round, malicious_noise, run_id
        in product(
            preset_config.contribution_score_strategy,
            preset_config.use_outlier_detection,
            _values_or_none(preset_config.freerider_start_round),
            _values_or_none(preset_config.freerider_noise_scale),
            _values_or_none(preset_config.malicious_start_round),
            _values_or_none(preset_config.malicious_noise_scale),
            runs,
        )
    ]


def read_git_state(repo: Path) -> GitState:
    """Read the checked-out commit and whether tracked files differ from it.

    Untracked files do not count as dirty. Without git, or when git fails, the commit is unknown and the
    state counts as dirty; the reason says which case it was.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                                check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo, capture_output=True, text=True)
    except OSError as e:
        return GitState(commit="unknown", dirty=True, reason=f"git is not installed or not runnable ({e})")
    except subprocess.CalledProcessError as e:
        return GitState(commit="unknown", dirty=True, reason=f"git failed: {(e.stderr or '').strip()}")
    # git diff --quiet exits 1 for changes; any other non-zero code is an error, not a dirty tree.
    if diff.returncode not in (0, 1):
        return GitState(commit="unknown", dirty=True,
                        reason=f"git diff failed with exit code {diff.returncode}: {(diff.stderr or '').strip()}")
    if diff.returncode == 1:
        return GitState(commit=commit, dirty=True, reason="tracked files have uncommitted changes")
    return GitState(commit=commit, dirty=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run every configuration of one preset on one dataset.")
    parser.add_argument("--skipFolder", type=str)
    parser.add_argument("--author", type=str)
    parser.add_argument("--preset", default="test",
                        choices=[name for name, p in PRESETS.items() if isinstance(p, ExperimentPreset)])
    parser.add_argument("--dataset", default="mnist", choices=DATASETS)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run even when tracked files have uncommitted changes or git is unavailable.")
    return parser


def plan_runs(flags: argparse.Namespace, git: GitState, start_time: str) -> list[PlannedRun]:
    """Build the configuration and metadata of every run the flags select.

    Refuses a dirty tree unless --allow-dirty is set, because the runs could not be tied to a commit.
    """
    if git.dirty and not flags.allow_dirty:
        raise SystemExit(
            f"Refusing to start: {git.reason or 'the working tree is dirty'} (commit: {git.commit}). "
            "Commit your changes, or pass --allow-dirty to record the runs as dirty."
        )
    runs = [_plan_run(flags.preset, spec, git, start_time) for spec in expand_preset(flags.preset, flags.dataset)]
    # Validated here, not only in the runner: the driver catches per-run errors, so a misconfigured
    # preset would otherwise fail run by run instead of stopping the batch.
    for run in runs:
        require_malicious_values(run.config)
    return runs


def _plan_run(preset: str, spec: RunSpec, git: GitState, start_time: str) -> PlannedRun:
    config = ExperimentConfiguration(preset=preset)
    config.preset_name = preset
    config.dataset = spec.dataset
    config.contribution_score_strategy = spec.strategy
    config.use_outlier_detection = spec.outlier_detection
    config.freerider_start_round = spec.freerider_round
    config.freerider_noise_scale = spec.freerider_noise
    config.malicious_start_round = spec.malicious_round
    config.malicious_noise_scale = spec.malicious_noise

    metadata = {
        **vars(config),
        "dataset": spec.dataset,
        "timestamp": start_time,
        "git_commit": git.commit,
        "git_dirty": git.dirty,
    }
    return PlannedRun(spec, config, metadata)
