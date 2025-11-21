import pytest
import torch
import torch.nn as nn
from decimal import Decimal
from unittest.mock import MagicMock

from openfl.contracts.fl_challenge import calc_contribution_score

class DummyModel(nn.Module):
    def __init__(self, weight_value):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[weight_value, weight_value],
                                               [weight_value, weight_value]], dtype=torch.float32))

    def parameters(self):
        return super().parameters()


class TestCalcContributionScore:
    # Basic test case with non-zero global model
    def test_calc_contribution_score_basic(self):
        local_model = DummyModel(1.0)
        global_model = DummyModel(2.0)
        num_mergers = 2

        score = calc_contribution_score(local_model, global_model, num_mergers)

        local_update = torch.cat([p.data.view(-1) for p in local_model.parameters()])
        global_update = torch.cat([p.data.view(-1) for p in global_model.parameters()])
        norm_U_sq = torch.dot(global_update, global_update)
        expected_score = torch.dot(local_update, global_update) / (num_mergers * norm_U_sq)
        expected_score = int(Decimal(expected_score.item()) * Decimal('1e18'))

        assert score == expected_score

    # Edge case where local model is identical to global model
    def test_calc_contribution_score_identical_models(self):
        local_model = DummyModel(2.0)
        global_model = DummyModel(2.0)
        num_mergers = 2

        score = calc_contribution_score(local_model, global_model, num_mergers)

        local_update = torch.cat([p.data.view(-1) for p in local_model.parameters()])
        global_update = torch.cat([p.data.view(-1) for p in global_model.parameters()])
        norm_U_sq = torch.dot(global_update, global_update)
        expected_score = torch.dot(local_update, global_update) / (num_mergers * norm_U_sq)
        expected_score = int(Decimal(expected_score.item()) * Decimal('1e18'))

        assert score == expected_score

    # Edge case where global model has zero weights
    def test_calc_contribution_score_zero_global(self):
        local_model = DummyModel(1.0)
        global_model = DummyModel(0.0)
        num_mergers = 3

        score = calc_contribution_score(local_model, global_model, num_mergers)
        expected_score = int(Decimal(1) / Decimal(num_mergers) * Decimal('1e18'))
        assert score == expected_score


class TestFLChallengeFeatures:

    # Test user registration process
    def test_register_all_users(self, fl_challenge, mock_participants, mock_contract):
        fl_challenge.register_all_users()
        assert mock_contract.functions.register.call_count == 3
        assert mock_participants[0].isRegistered is True

    # Test feedback giving when no cheater is detected
    def test_give_feedback_no_cheater(self, fl_challenge):
        giver = MagicMock()
        giver.address = "0xGiver"

        target = MagicMock()
        target.address = "0xTarget"

        target.roundRep = 5
        giver.cheater = []

        fl_challenge.give_feedback(giver, target, 1)

        fl_challenge.model.functions.feedback.assert_called_with(target.address, 1)

    # Test feedback giving when a cheater is detected
    def test_give_feedback_cheater_detected(self, fl_challenge):
        giver = MagicMock()
        giver.address = "0xGiver"

        target = MagicMock()
        target.address = "0xTarget"

        target.roundRep = 0
        giver.cheater = [target]

        fl_challenge.give_feedback(giver, target, 1)

        fl_challenge.model.functions.feedback.assert_called_with(target.address, -1)

    # Test building feedback bytes
    def test_build_feedback_bytes(self, fl_challenge):
        valid_address = "0xdD870fA1b7C4700F2BD7f44238821C26f7392148"
        votes = [1]

        res = fl_challenge.build_feedback_bytes([valid_address], votes)

        assert isinstance(res, str)
        assert len(res) > 0