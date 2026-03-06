from dataclasses import dataclass

# Parameters controlled by the default preset
@dataclass(frozen=True)
class DefaultPreset:
    fork: bool
    reward: int
    standard_buy_in: int
    min_buy_in: int
    max_buy_in: int
    first_round_fee: int
    punish_factor: int
    punish_factor_contrib: int
    force_merge_all: bool
    use_nobody_is_kicked: bool
    number_of_inactive_contributors: int


# Parameters specific to experiments
@dataclass(frozen=True)
class ExperimentPreset:
    number_of_good_contributors: int
    number_of_bad_contributors: int
    number_of_freerider_contributors: int
    minimum_rounds: int
    epochs: int
    batch_size: int
    use_outlier_detection: list[bool]
    contribution_score_strategy: list[str]
    freerider_noise_scale: list[float]
    freerider_start_round: list[int]
    malicious_start_round: list[int]
    malicious_noise_scale: list[float]
    aggregation_rule: list[str]


# Full preset (used when use_defaults=False)
@dataclass(frozen=True)
class FullPreset(DefaultPreset, ExperimentPreset):
    pass


PRESETS = {
    "default": DefaultPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=False,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,
    ),

    "mnist_openfl": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True, False],
        contribution_score_strategy=["loss_only", "accuracy_only"],
        freerider_noise_scale=[1.0],
        freerider_start_round=[3],
        malicious_start_round=[3],
        malicious_noise_scale=[1.0],
        aggregation_rule= ["FedAVG"],
    ),
}

# Hvis du vil overwrite værdi fra default preset, så skal du lave en FullPreset class inde i PRESETS. .
