import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import types

# Provide a lightweight yaml stub to avoid external dependency during tests
if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(_stream):
        return {
            "printing": {"ONLY_PRINT_ROUND_SUMMARY": False},
            "contracts": {
                "WAIT_DELAY": 172800,
                "FEEDBACK_ROUND_TIMEOUT": 30,
                "CONTRIBUTION_ROUND_TIMEOUT": 30,
            },
        }

    yaml_stub.safe_load = _safe_load
    sys.modules["yaml"] = yaml_stub

# Add src folder to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from openfl.contracts.fl_challenge import FLChallenge


@pytest.fixture
def mock_w3():
    """Mocks a web3 connection to Ethereum."""
    w3 = MagicMock()
    w3.eth.get_transaction_count.return_value = 10
    w3.eth.get_balance.return_value = 1000000000000000000
    w3.eth.wait_for_transaction_receipt.return_value = {
        "gasUsed": 21000,
        "transactionHash": b"\x00" * 32,
        "logs": [],
    }
    w3.to_checksum_address.side_effect = lambda x: x
    return w3


@pytest.fixture
def mock_contract():
    """Mocks a Solidity Smart Contract."""
    contract = MagicMock()
    contract.functions.register.return_value.transact.return_value = b"\x01" * 32
    contract.functions.feedback.return_value.transact.return_value = b"\x02" * 32
    contract.functions.closeRound.return_value.transact.return_value = b"\x03" * 32
    contract.functions.rewardLeft.return_value.call.return_value = 5000
    return contract


@pytest.fixture
def mock_participants():
    """Create a list of 3 dummy participants."""
    users = []
    for i in range(3):
        user = MagicMock()
        user.address = f"0xAddressUser{i}"
        user.privateKey = f"privateKey{i}"
        user.collateral = 1000
        user.isRegistered = False
        user.attitude = "honest"
        user.cheater = []
        user.id = i
        user.hashedModel = b"hash"
        user.secret = 123
        users.append(user)
    return users


@pytest.fixture
def fl_challenge(mock_w3, mock_contract, mock_participants):
    """Create an instance of FLChallenge with injected mocks."""
    manager = MagicMock()
    manager.w3 = mock_w3
    manager.fork = True

    configs = [
        mock_contract,  # model
        "0xModelAddress",  # modelAddress
        100,  # MIN_BUY_IN
        1000,  # MAX_BUY_IN
        500,  # REWARD
        3,  # MIN_ROUNDS
        0.5,  # PUNISHMENT_FACTOR
        0.1,  # FREERIDER_FACTOR
    ]

    pytorch_model = MagicMock()
    pytorch_model.participants = mock_participants
    pytorch_model.round = 1

    with patch("openfl.contracts.fl_challenge.FLManager.build_tx") as mock_build_tx:
        mock_build_tx.return_value = {"gas": 100000, "gasPrice": 1, "nonce": 1}

        with patch(
            "openfl.contracts.fl_challenge.FLManager.build_non_fork_tx"
        ) as mock_build_nf_tx:
            mock_build_nf_tx.return_value = {"gas": 100000, "nonce": 1}

            challenge = FLChallenge(manager, configs, pytorch_model)
            challenge.get_global_reputation_of_user = MagicMock(return_value=1000)
            yield challenge
