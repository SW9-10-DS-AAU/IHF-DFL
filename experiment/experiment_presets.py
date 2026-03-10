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

    "mnist_openfl_w_outlier": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive", "dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1, 3, 5],
        malicious_start_round=[1, 3, 5],
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        aggregation_rule= ["FedAVG"],
    ),

    "mnist_openfl_w/o_outlier": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[False],
        contribution_score_strategy=["dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1,3,5],
        malicious_start_round=[1,3,5],
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        aggregation_rule= ["FedAVG"],
    ),

    "cifar_openfl_w_outlier": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive", "dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1, 3, 5],
        malicious_start_round=[1, 3, 5],
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        aggregation_rule= ["FedAVG"],
    ),

"cifar_openfl_w/o_outlier": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[False],
        contribution_score_strategy=["dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1, 3, 5],
        malicious_start_round=[1, 3, 5],
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        aggregation_rule= ["FedAVG"],
    ),

    "aggregation_rules_test_mode_performance_mnist": FullPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=True,
        number_of_inactive_contributors=0,
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=50,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only"],
        freerider_noise_scale=[0, 0.1,1.0],
        freerider_start_round=[1, 5, 10],
        malicious_start_round=[1, 5, 10],
        malicious_noise_scale=[0, 0.1,1.0],
        aggregation_rule=["positives_only", "plus_one_normalize", "FedAVG"],
    )

}

# Hvis du vil overwrite værdi fra default preset, så skal du lave en FullPreset class inde i PRESETS. .
