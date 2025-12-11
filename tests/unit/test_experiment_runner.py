from types import SimpleNamespace

import experiment.experiment_runner as runner

# Maybe delete these tests

class DummyModel:
    def __init__(self):
        self.txHashes = [("fn", "0xabc")]
        self.w3 = SimpleNamespace(
            eth=SimpleNamespace(
                wait_for_transaction_receipt=lambda tx: {"gasUsed": 1, "status": 1}
            )
        )


class DummyPytorchModel:
    def __init__(self):
        self.participants = []
        self.round = 1

    def add_participant(self, *args, **kwargs):
        self.participants.append("user")


class DummyFLManager:
    def __init__(self, model, manual):
        self.model = model
        self.manual = manual
        self.built = False
        self.deployed = False

    def init(self, *args):
        return self

    def build_contract(self):
        self.built = True

    def deploy_challenge_contract(self, *args):
        self.deployed = True
        return ("contract", "0x", 1, 2, 3, 4, 5)


class DummyChallenge:
    def __init__(self, manager, configs, pytorch_model, experiment_config=None):
        self.manager = manager
        self.configs = configs
        self.pytorch_model = pytorch_model
        self.experiment_config = experiment_config
        self.simulated = False
        self.gas_register = [1]
        self.gas_feedback = [2]
        self.gas_close = [3]
        self.gas_slot = [4]
        self.gas_weights = [5]
        self.gas_exit = [6]
        self.txHashes = [("register", b"\x00" * 32)]
        self.model = SimpleNamespace(address="0xmodel")
        self.w3 = SimpleNamespace(
            eth=SimpleNamespace(wait_for_transaction_receipt=lambda tx: {"gasUsed": 10, "status": 1})
        )

    def simulate(self, rounds):
        self.simulated = True

    def visualize_simulation(self, *_):
        return None


class DummyExperiment:
    def __init__(self):
        self.model = DummyChallenge(None, None, None, None)
        self.manager = SimpleNamespace(manager=SimpleNamespace(address="0xmanager"))


def test_run_experiment_with_mocks(monkeypatch):
    monkeypatch.setattr(runner, "require_env_var", lambda name: "rpc")
    monkeypatch.setattr(runner.PM, "PytorchModel", lambda *args, **kwargs: DummyPytorchModel())
    monkeypatch.setattr(runner.Manager, "FLManager", DummyFLManager)
    monkeypatch.setattr(runner.Challenge, "FLChallenge", DummyChallenge)

    config = SimpleNamespace(
        number_of_good_contributors=1,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=0,
        number_of_inactive_contributors=0,
        number_of_contributors=1,
        epochs=1,
        batch_size=1,
        standard_buy_in=1,
        minimum_rounds=1,
        fork=True,
        min_buy_in=1,
        max_buy_in=1,
        reward=1,
        punish_factor=1,
        first_round_fee=1,
        contribution_score_strategy="accuracy",
        freerider_noise_scale=0.01,
        freerider_start_round=3,
    )

    experiment = runner.run_experiment("dataset", config)

    assert isinstance(experiment, runner.Experiment)
    assert experiment.manager.built is True
    assert experiment.manager.deployed is True
    assert experiment.model.simulated is True


def test_print_transactions_outputs_receipts(capsys):
    dummy = DummyExperiment()
    dummy.model.txHashes = [("fn", "hash")]
    dummy.model.w3.eth.receipts = {"hash": {"status": 1, "gasUsed": 42}}

    runner.print_transactions(dummy)
    captured = capsys.readouterr().out
    assert "fn" in captured
    assert "10" in captured


def test_print_latex_writes_addresses(capsys):
    dummy = DummyExperiment()
    dummy.model.pytorch_model = SimpleNamespace(participants=[SimpleNamespace(address="0xP")], disqualified=[])
    runner.print_latex(dummy)
    captured = capsys.readouterr().out
    assert "0xmanager" in captured
    assert "0xmodel" in captured


def test_table_with_gas_and_transactions_latex(capsys):
    dummy = DummyExperiment()
    dummy.manager.gas_deploy = [1]
    dummy.model.gas_register = [1]
    dummy.model.gas_slot = [1]
    dummy.model.gas_weights = [1]
    dummy.model.gas_feedback = [1]
    dummy.model.gas_close = [1]
    runner.table_with_gas_and_transactions_latex(dummy)
    captured = capsys.readouterr().out
    assert "complete round" in captured
