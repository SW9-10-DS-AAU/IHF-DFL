import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import types



from openfl.contracts.fl_challenge import FLChallenge


@pytest.fixture
def mock_w3():
    """Mocks a web3 connection to Ethereum."""
    w3 = MagicMock()

    mock_receipt = MagicMock()
    mock_receipt.gasUsed = 21000
    mock_receipt.transactionHash = b'\x00' * 32
    mock_receipt.logs = []

    mock_receipt.__getitem__.side_effect = lambda x: getattr(mock_receipt, x)

    w3.eth.get_transaction_count.return_value = 10
    w3.eth.get_balance.return_value = 1000000000000000000
    w3.eth.wait_for_transaction_receipt.return_value = mock_receipt

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
        user.hashedModel = b'hash'

        # Give unique secrets to ensure filtering tests work correctly
        user.secret = 100 + i

        users.append(user)
    return users


@pytest.fixture
def fl_challenge(request, mock_w3, mock_contract, mock_participants):
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
        3, # PUNISHMENT_FACTOR_CONTRIB
        0.1,  # FREERIDER_FACTOR
    ]

    pytorch_model = MagicMock()
    pytorch_model.participants = mock_participants
    pytorch_model.round = 1

    experiment_config = getattr(
        request, "param", SimpleNamespace(
            contribution_score_strategy="dotproduct",
            use_outlier_detection=False,
        )
    )

    with patch("openfl.contracts.fl_challenge.FLManager.build_tx") as mock_build_tx:
        mock_build_tx.return_value = {"gas": 100000, "gasPrice": 1, "nonce": 1}

        with patch(
            "openfl.contracts.fl_challenge.FLManager.build_non_fork_tx"
        ) as mock_build_nf_tx:
            mock_build_nf_tx.return_value = {"gas": 100000, "nonce": 1}

            challenge = FLChallenge(manager, configs, pytorch_model, experiment_config)
            challenge.get_global_reputation_of_user = MagicMock(return_value=1000)
            yield challenge
