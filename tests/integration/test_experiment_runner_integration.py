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