import types

import openfl.api.connection_helper as connection_helper


class DummyWeb3:
    def __init__(self):
        self.to_checksum_address_calls = []

    def to_checksum_address(self, addr):
        self.to_checksum_address_calls.append(addr)
        return f"checksum_{addr}"

    def toWei(self, value, unit):
        return f"{value}_{unit}"


def test_build_tx_uses_checksum(monkeypatch):
    dummy = DummyWeb3()
    monkeypatch.setattr(connection_helper, "w3", dummy, raising=False)

    tx = connection_helper.ConnectionHelper().build_tx("from", "to", 10)

    assert tx["from"] == "checksum_from"
    assert tx["to"] == "checksum_to"
    assert tx["value"] == 10
    assert dummy.to_checksum_address_calls == ["from", "to"]


def test_build_non_fork_tx_with_data(monkeypatch):
    dummy = DummyWeb3()
    monkeypatch.setattr(connection_helper, "w3", dummy, raising=False)

    tx = connection_helper.ConnectionHelper().build_non_fork_tx(
        addr="0xabc",
        nonce=1,
        to="0xdef",
        value=5,
        data=b"123",
    )

    assert tx["chainId"] == 3
    assert tx["from"] == "0xabc"
    assert tx["to"] == "0xdef"
    assert tx["value"] == 5
    assert tx["data"] == b"123"
    assert tx["maxFeePerGas"] == "12_gwei"
    assert tx["maxPriorityFeePerGas"] == "2_gwei"
