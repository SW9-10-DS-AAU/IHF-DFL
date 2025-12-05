class ExperimentConfiguration:
    def __init__(self,
                 number_of_good_contributors=4,
                 number_of_bad_contributors=1,
                 number_of_freerider_contributors=1,
                 number_of_inactive_contributors=0,
                 reward=int(1e18),
                 minimum_rounds=6,
                 min_buy_in=int(1e18),
                 max_buy_in=int(1e18),
                 standard_buy_in=int(1e18),
                 epochs=1,
                 batch_size=32,
                 punish_factor=2,
                 first_round_fee=50,
                 fork=True,
                 contribution_score_strategy="accuracy"): # Options: mad, legacy, accuracy, None (defaults to MAD)

        # Apply scaling only if we’re on Sepolia (fork = False)
        if not fork:
            scale = 0.005  # scale down
            reward = int(reward * scale)
            min_buy_in = int(min_buy_in * scale)
            max_buy_in = int(max_buy_in * scale)
            standard_buy_in = int(standard_buy_in * scale)

        # Store everything
        self.number_of_good_contributors = number_of_good_contributors
        self.number_of_bad_contributors = number_of_bad_contributors
        self.number_of_freerider_contributors = number_of_freerider_contributors
        self.number_of_inactive_contributors = number_of_inactive_contributors

        self.reward = reward
        self.minimum_rounds = minimum_rounds
        self.min_buy_in = min_buy_in
        self.max_buy_in = max_buy_in
        self.standard_buy_in = standard_buy_in
        self.epochs = epochs
        self.batch_size = batch_size
        self.punish_factor = punish_factor
        self.first_round_fee = first_round_fee
        self.contribution_score_strategy = contribution_score_strategy

    @property
    def number_of_contributors(self):
        return (self.number_of_good_contributors +
                self.number_of_bad_contributors +
                self.number_of_freerider_contributors +
                self.number_of_inactive_contributors)
