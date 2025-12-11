import types
from types import SimpleNamespace

import experiment.experiment_runner as runner
from experiment.experiment_configuration import ExperimentConfiguration


class FakePytorchModel:
    def __init__(self, dataset, good_contributors, total_contributors, epochs, batch_size,
                 standard_buy_in, max_buy_in, freerider_noise_scale, freerider_start_round):
        self.init_args = dict(
            dataset=dataset,
            good_contributors=good_contributors,
            total_contributors=total_contributors,
            epochs=epochs,
            batch_size=batch_size,
            standard_buy_in=standard_buy_in,
            max_buy_in=max_buy_in,
            freerider_noise_scale=freerider_noise_scale,
            freerider_start_round=freerider_start_round,
        )
        self.added_participants = []

    def add_participant(self, role, count):
        self.added_participants.append((role, count))


class FakeManager:
    def __init__(self, pytorch_model, manual_ganache_setup=False):
        self.pytorch_model = pytorch_model
        self.manual_ganache_setup = manual_ganache_setup
        self.init_args = None
        self.built = False
        self.deployed_with = None

    def init(self, *args):
        self.init_args = args
        return self

    def build_contract(self):
        self.built = True

    def deploy_challenge_contract(self, *args):
        self.deployed_with = args
        return ("fake_contract", "0xcontract") + args


class FakeChallenge:
    def __init__(self, manager, configs, pytorch_model, experiment_config):
        self.manager = manager
        self.configs = configs
        self.pytorch_model = pytorch_model
        self.experiment_config = experiment_config
        self.simulated_rounds = None
        self.txHashes = []
        self.gas_register = []
        self.gas_feedback = []
        self.gas_close = []
        self.gas_slot = []
        self.gas_weights = []
        self.gas_exit = []

    def simulate(self, rounds):
        self.simulated_rounds = rounds


def _patch_basics(monkeypatch, fork=True, private_keys=""):
    monkeypatch.setenv("RPC_URL", "http://rpc.example")
    monkeypatch.setenv("PRIVATE_KEYS", private_keys)

    monkeypatch.setattr(runner, "require_env_var", lambda key: runner.os.environ[key])
    monkeypatch.setattr(runner.PM, "PytorchModel", FakePytorchModel)
    monkeypatch.setattr(runner.Manager, "FLManager", FakeManager)
    monkeypatch.setattr(runner.Challenge, "FLChallenge", FakeChallenge)
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(perf_counter=lambda: 0.0))

    if not fork:
        class FakeHTTPProvider:
            def __init__(self, url):
                self.url = url

        class FakeWeb3:
            HTTPProvider = FakeHTTPProvider

            def __init__(self, provider):
                self.provider = provider

        class FakeAccount:
            def __init__(self, key):
                self._private_key = f"raw-{key}"
                self.address = f"addr-{key}"

        monkeypatch.setattr(runner, "Web3", FakeWeb3)
        monkeypatch.setattr(runner, "Account", types.SimpleNamespace(from_key=lambda key: FakeAccount(key)))


def test_run_experiment_fork_flow(monkeypatch):
    # Forked deployments should initialize participants and contracts while skipping account setup.
    _patch_basics(monkeypatch, fork=True)
    config = ExperimentConfiguration(
        number_of_good_contributors=2,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        number_of_inactive_contributors=1,
        minimum_rounds=3,
        standard_buy_in=5,
        max_buy_in=10,
        freerider_noise_scale=0.1,
        freerider_start_round=2,
    )

    experiment = runner.run_experiment("mnist", config)

    fake_model = experiment.manager.pytorch_model
    assert fake_model.init_args["dataset"] == "mnist"
    assert fake_model.init_args["good_contributors"] == 2
    assert fake_model.added_participants == [
        ("bad", 3),
        ("freerider", 1),
        ("inactive", 1),
    ]

    assert experiment.manager.manual_ganache_setup is True
    assert experiment.manager.init_args == (
        2,
        1,
        1,
        1,
        3,
        "http://rpc.example",
        True,
        None,
    )

    assert experiment.manager.built is True
    assert experiment.manager.deployed_with == (
        config.min_buy_in,
        config.max_buy_in,
        config.reward,
        config.minimum_rounds,
        config.punish_factor,
        config.first_round_fee,
    )
    assert experiment.model.simulated_rounds == config.minimum_rounds


def test_run_experiment_non_fork_loads_private_keys(monkeypatch):
    # Non-forked runs should load provided private keys and pass accounts into the manager.
    keys = "key-one\nkey-two\n"
    _patch_basics(monkeypatch, fork=False, private_keys=keys)

    config = ExperimentConfiguration(fork=False, number_of_good_contributors=1, minimum_rounds=2)

    experiment = runner.run_experiment("cifar", config)

    # Accounts are passed into the manager when not forking
    passed_accounts = experiment.manager.init_args[-1]
    assert isinstance(passed_accounts, list)
    assert [acc.address for acc in passed_accounts] == ["addr-key-one", "addr-key-two"]
    assert experiment.manager.init_args[5:7] == ("http://rpc.example", False)
    assert experiment.model.simulated_rounds == config.minimum_rounds