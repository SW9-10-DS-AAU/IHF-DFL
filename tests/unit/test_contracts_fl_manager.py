from types import SimpleNamespace

import openfl.contracts.fl_manager as fl_manager


class DummyHelper:
    def __init__(self):
        self.called_with = None

    def initiate_rpc(self, **kwargs):
        self.called_with = kwargs
        return "web3", 123

    def initialize(self):
        return "manager"


def test_init_delegates_to_connection_helper(monkeypatch):
    helper = DummyHelper()
    monkeypatch.setattr(fl_manager.ConnectionHelper, "initiate_rpc", lambda self, **kwargs: helper.initiate_rpc(**kwargs))
    monkeypatch.setattr(fl_manager.ConnectionHelper, "initialize", lambda self: helper.initialize())

    mgr = fl_manager.FLManager(pytorch_model="model")
    result = mgr.init(1, 0, 0, 0, 5)

    assert result is mgr
    assert mgr.w3 == "web3"
    assert mgr.latestBlock == 123
    assert mgr.manager == "manager"
    assert helper.called_with["NUMBER_OF_GOOD_CONTRIBUTORS"] == 1


def test_get_model_queries_manager(monkeypatch):
    class FakeCall:
        def __init__(self):
            self.called_with = None
        def call(self, params):
            self.called_with = params
            return "result"

    class FakeFunctions:
        def __init__(self):
            self.last_params = None
        def ModelOf(self, address, count):
            self.last_params = (address, count)
            return FakeCall()
        def ModelCountOf(self, address):
            self.last_params = address
            return FakeCall()

    manager_obj = SimpleNamespace(functions=FakeFunctions(), address="0xmanager")
    mgr = fl_manager.FLManager(pytorch_model="model")
    mgr.manager = manager_obj

    participant = SimpleNamespace(address="0xabc")

    assert mgr.get_model_of(participant, 2) == "result"
    assert manager_obj.functions.last_params == ("0xabc", 2)

    assert mgr.get_model_count_of(participant) == "result"
    assert manager_obj.functions.last_params == participant.address
