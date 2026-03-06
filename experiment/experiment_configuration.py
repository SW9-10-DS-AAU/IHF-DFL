from experiment_presets import PRESETS, DefaultPreset, ExperimentPreset, FullPreset


class ExperimentConfiguration:

    def __init__(self, preset: str, use_defaults: bool = True):

        if preset not in PRESETS:
            raise ValueError(f"Preset '{preset}' not found")

        p = PRESETS[preset]

        if use_defaults:
            if not isinstance(p, ExperimentPreset):
                raise TypeError(
                    f"Preset '{preset}' must be ExperimentPreset when use_defaults=True"
                )

            d: DefaultPreset = PRESETS["default"]

            # defaults
            self.fork = d.fork
            self.reward = d.reward
            self.standard_buy_in = d.standard_buy_in
            self.min_buy_in = d.min_buy_in
            self.max_buy_in = d.max_buy_in
            self.first_round_fee = d.first_round_fee
            self.punish_factor = d.punish_factor
            self.punish_factor_contrib = d.punish_factor_contrib
            self.force_merge_all = d.force_merge_all
            self.use_nobody_is_kicked = d.use_nobody_is_kicked
            self.number_of_inactive_contributors = d.number_of_inactive_contributors

            # experiment
            self.number_of_good_contributors = p.number_of_good_contributors
            self.number_of_bad_contributors = p.number_of_bad_contributors
            self.number_of_freerider_contributors = p.number_of_freerider_contributors
            self.minimum_rounds = p.minimum_rounds
            self.epochs = p.epochs
            self.batch_size = p.batch_size
            self.use_outlier_detection = p.use_outlier_detection
            self.contribution_score_strategy = p.contribution_score_strategy
            self.freerider_noise_scale = p.freerider_noise_scale
            self.freerider_start_round = p.freerider_start_round
            self.malicious_start_round = p.malicious_start_round
            self.malicious_noise_scale = p.malicious_noise_scale
            self.aggregation_rule = p.aggregation_rule

        else:

            if not isinstance(p, FullPreset):
                raise TypeError(
                    f"Preset '{preset}' must be FullPreset when use_defaults=False"
                )

            for k, v in p.__dict__.items():
                setattr(self, k, v)

    @property
    def number_of_contributors(self):
        return (
            self.number_of_good_contributors
            + self.number_of_bad_contributors
            + self.number_of_freerider_contributors
            + self.number_of_inactive_contributors
        )












 # Apply scaling only if we’re on Sepolia (fork = False)
        #if not fork:
        #    scale = 0.005  # scale down
        #    reward = int(reward * scale)
        #    min_buy_in = int(min_buy_in * scale)
        #    max_buy_in = int(max_buy_in * scale)
        #    standard_buy_in = int(standard_buy_in * scale)