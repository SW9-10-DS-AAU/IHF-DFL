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
    malicious_noise_scale: list[float] | None
    malicious_start_round: list[int] | None
    aggregation_rule: list[str]
    data_distribution: list[str]
    dirichlet_alpha: list[float] | None


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

    "test": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=3,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0],
        freerider_start_round=[2],
        malicious_noise_scale=[0.1],
        malicious_start_round=[2],
        aggregation_rule=["FedAVG"],
        data_distribution= ["random_split_42"],
        dirichlet_alpha= None,
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
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        malicious_start_round=[1, 3, 5],
        aggregation_rule= ["FedAVG"],
        data_distribution= ["random_split_42"],
        dirichlet_alpha= None,
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
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        malicious_start_round=[1, 3, 5],
        aggregation_rule= ["FedAVG"],
        data_distribution= ["random_split_42"],
        dirichlet_alpha= None,
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
        malicious_noise_scale=None,
        malicious_start_round=None,
        aggregation_rule= ["FedAVG"],
        data_distribution= ["random_split_42"],
        dirichlet_alpha= None,
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
        malicious_noise_scale=None,
        malicious_start_round=None,
        aggregation_rule= ["FedAVG"],
        data_distribution= ["random_split_42"],
        dirichlet_alpha= None,
    ),

    "aggregation_rules_test_model_performance_mnist": FullPreset(
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
        minimum_rounds=5,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only"],
        freerider_noise_scale=[0, 0.1,1.0],
        freerider_start_round=[1, 5, 10],
        malicious_noise_scale=None,
        malicious_start_round=None,
        aggregation_rule=["positives_only", "FedAVG", "plus_one_normalize"],
        data_distribution= ["random_split_42"],
        dirichlet_alpha= None,
    ),

    "aggregation_rules_test_model_performance_people_get_kicked_now_mnist": FullPreset(
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
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"], #loss_only is the only loss'os
        freerider_noise_scale=[0.01], # 0.0
        freerider_start_round=[1], # 1
        malicious_noise_scale=[0.1],
        malicious_start_round=None,
        aggregation_rule=["GRS_aggregation", "FedAVG", "positives_only", "binary_switch", "plus_one_normalize"], # 3
        data_distribution= ["random_split_42"], # 1
        dirichlet_alpha= None
    ),
    "aggregation_rules_test_model_performance_people_get_kicked_now_cifar": FullPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=False,
        use_nobody_is_kicked=True,
        number_of_inactive_contributors=0,
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=50,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"], #loss_only is the only loss'os
        freerider_noise_scale=[0.01], # 0.0
        freerider_start_round=[1], # 1
        malicious_noise_scale=[0.1],
        malicious_start_round=None,
        aggregation_rule=["binary_switch","positives_only", "FedAVG", "plus_one_normalize"], # 4
        data_distribution= ["random_split_42"], # 1
        dirichlet_alpha= None,
    ),
"data_distribution_mnist": FullPreset(
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
        contribution_score_strategy=["loss_only"], #loss_only is the only loss'os
        freerider_noise_scale=[0.0], # 0.0
        freerider_start_round=[1], # 1
        malicious_noise_scale=[0.1],
        malicious_start_round=None,
        aggregation_rule=["positives_only", "FedAVG", "plus_one_normalize"], # 3
        data_distribution= ["random_split_42", "stratified_split_42", "dirichlet_split_42"], # 3
        dirichlet_alpha= [0.5, 5.0],
    ),
}


# If you want to overwrite a value from the default preset, you need to create a FullPreset class inside PRESETS.