import pytest
import torch
import torch.nn as nn
from decimal import Decimal
from unittest.mock import MagicMock, patch

from openfl.contracts.fl_challenge import calc_contribution_score, calc_contribution_scores_mad

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
        """
        Basic test case where local and global models have distinct weights.
        The contribution score should be calculated correctly.
        """
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
        """
        Edge case where the local model is identical to the global model.
        The contribution score should be maximized.
        """
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
        """
        Edge case where the global model has zero weights.
        The contribution score should default to 1/N.
        """
        local_model = DummyModel(1.0)
        global_model = DummyModel(0.0)
        num_mergers = 3

        score = calc_contribution_score(local_model, global_model, num_mergers)
        expected_score = int(Decimal(1) / Decimal(num_mergers) * Decimal('1e18'))
        assert score == expected_score

    # Edge case where both models have zero weights
    def test_calc_contribution_score_both_zero(self):
        """
        Edge case where both local and global models have zero weights.
        The contribution score should default to 1/N.
        """
        local_model = DummyModel(0.0)
        global_model = DummyModel(0.0)
        num_mergers = 4

        score = calc_contribution_score(local_model, global_model, num_mergers)
        expected_score = int(Decimal(1) / Decimal(num_mergers) * Decimal('1e18'))
        assert score == expected_score


class TestCalcContributionScoresMAD:
    # Basic test case with no outliers
    def test_basic_no_outliers(self):
        """
        Scenario: 3 users with updates that are close to each other.
        No values should be filtered out.
        """
        num_mergers = 3

        # Local updates: shape (3, 2)
        # All vectors are reasonably close, no statistical outliers
        local_updates = torch.tensor([
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9]
        ])
        global_update = torch.tensor([1.0, 2.0])

        scores = calc_contribution_scores_mad(local_updates, global_update, mad_thresh=3.5)

        # Manual Calculation for verification
        norm_U_sq = torch.dot(global_update, global_update)  # 1*1 + 2*2 = 5

        expected_scores = []
        for i in range(num_mergers):
            # Standard logic: dot(u, U) / (N * ||U||^2)
            dot = torch.dot(local_updates[i], global_update)
            score_float = dot / (num_mergers * norm_U_sq)
            expected_scores.append(int(Decimal(score_float.item()) * Decimal('1e18')))

        assert scores == expected_scores

    # Test case with an outlier
    def test_with_outliers(self):
        """
        Scenario: User 2 has a massive weight that deviates significantly from the median.
        The MAD logic should zero out that specific weight for User 2.
        """
        # 1 Dimension for simplicity

        # User 0: 1.0
        # User 1: 2.0  <- Median is 2.0
        # User 2: 100.0 <- Outlier!
        local_updates = torch.tensor([
            [1.0],
            [2.0],
            [100.0]
        ])
        global_update = torch.tensor([1.0])

        # Explanation of MAD logic here:
        # Median = 2.0
        # Abs deviations = |1-2|=1, |2-2|=0, |100-2|=98
        # MAD (Median of deviations) = 1.0
        # Z-score User 2 = 0.6745 * (98 / 1) = ~66.1 > 3.5
        # Result: User 2's weight becomes 0.0

        scores = calc_contribution_scores_mad(local_updates, global_update, mad_thresh=3.5)

        norm_U_sq = torch.dot(global_update, global_update)  # 1.0

        # --- Verify User 2 (The Outlier) ---
        # Since the weight 100.0 is filtered to 0.0:
        # dot(0.0, 1.0) = 0
        # Score should be 0
        assert scores[2] == 0

        # --- Verify User 1 (The Median) ---
        # Weight 2.0 is kept.
        # dot(2.0, 1.0) = 2.0
        # score = 2.0 / (3 * 1.0) = 0.666...
        expected_u1 = 2.0 / (3 * norm_U_sq)
        expected_u1_int = int(Decimal(expected_u1.item()) * Decimal('1e18'))
        assert scores[1] == expected_u1_int

    # Edge case: Global update is zero vector
    def test_zero_global_update(self):
        """
        Edge case: Global update vector is zero (or very close to zero).
        Everyone should receive equal contribution scores (1/N).
        """
        num_mergers = 4
        local_updates = torch.randn(num_mergers, 10)  # Random values
        global_update = torch.zeros(10)  # Zero vector

        scores = calc_contribution_scores_mad(local_updates, global_update)

        # Expectation: 1/4 * 1e18
        expected_val = int((Decimal(1) / Decimal(num_mergers)) * Decimal('1e18'))

        for score in scores:
            assert score == expected_val

    # Edge case: All users submit identical models
    def test_identical_inputs(self):
        """
        Edge case: All users submit identical models.
        MAD will be 0. Ensure eps prevents division by zero and logic holds.
        """
        num_mergers = 2
        # Both users submit [2.0, 2.0]
        local_updates = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        global_update = torch.tensor([2.0, 2.0])

        scores = calc_contribution_scores_mad(local_updates, global_update)

        # If identical, deviation is 0. Z-score is 0. Mask is True (keep all).
        norm_U_sq = 8.0  # 2*2 + 2*2
        dot = 8.0  # 2*2 + 2*2

        expected_float = dot / (num_mergers * norm_U_sq)
        expected_int = int(Decimal(expected_float) * Decimal('1e18'))

        assert scores[0] == expected_int
        assert scores[1] == expected_int

    # Test case with partial filtering
    def test_partial_filtering(self):
        """
        Converted manual calculation to use PyTorch Tensors to match precision.
        """
        local_updates = torch.tensor([
            [1.0, 1.0],
            [2.0, 1.0],
            [100.0, 1.0]
        ])
        global_update = torch.tensor([1.0, 1.0])

        scores = calc_contribution_scores_mad(local_updates, global_update)

        effective_u2_filtered = torch.tensor([0.0, 1.0])
        norm_U_sq = torch.dot(global_update, global_update)

        effective_dot = torch.dot(effective_u2_filtered, global_update)

        expected_u2_float = effective_dot / (3 * norm_U_sq)

        expected_u2_int = int(Decimal(expected_u2_float.item()) * Decimal('1e18'))

        assert scores[2] == expected_u2_int


class TestFLChallengeFeatures:
    # Test user registration process
    def test_register_all_users(self, fl_challenge, mock_participants, mock_contract):
        """
        Test the register_all_users method to ensure all users are registered correctly.
        3 users should result in 3 calls to the contract's register function.
        1st user's isRegistered flag should be set to True.
        """
        fl_challenge.register_all_users()
        assert mock_contract.functions.register.call_count == 3
        assert mock_participants[0].isRegistered is True

    # Test feedback giving when no cheater is detected
    def test_give_feedback_no_cheater(self, fl_challenge):
        """
        Test the give_feedback method when the target is not marked as a cheater by the giver.
        """
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
        """
        Test the give_feedback method when the target is marked as a cheater by the giver.
        """
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
        """
        Test the build_feedback_bytes method to ensure it returns a non-empty string.
        """
        valid_address = "0xdD870fA1b7C4700F2BD7f44238821C26f7392148"
        votes = [1]

        res = fl_challenge.build_feedback_bytes([valid_address], votes)

        assert isinstance(res, str)
        assert len(res) > 0

    # Test the MAD score calculation wrapper
    def test_calculate_scores_mad_wrapper(self, fl_challenge, mock_participants):
        """
        Test that _calculate_scores_mad correctly flattens parameters from
        user models and the global model, stacks them, and passes them
        to the underlying math function.
        """
        # 1. Setup: Create a global model with known weights (all 10.0)
        merged_model = DummyModel(10.0)

        # 2. Setup: Assign 'previousModel' to each mock participant with distinct weights
        for i, user in enumerate(mock_participants):
            user.previousModel = DummyModel(float(i + 1))

        # 3. Patch the actual math function.
        with patch('openfl.contracts.fl_challenge.calc_contribution_scores_mad') as mock_math:
            # Set a dummy return value
            mock_math.return_value = [1000, 2000, 3000]

            # 4. Action
            scores = fl_challenge._calculate_scores_mad(mock_participants, merged_model)

            # 5. Assertions
            assert scores == [1000, 2000, 3000]

            # Inspect the arguments passed to calc_contribution_scores_mad
            args, _ = mock_math.call_args
            local_updates_arg = args[0]
            global_update_arg = args[1]

            # Check Global Update Vector
            assert global_update_arg.shape == (4,)
            expected_global = torch.tensor([10.0, 10.0, 10.0, 10.0])
            assert torch.equal(global_update_arg, expected_global)

            # Check Local Updates Matrix
            assert local_updates_arg.shape == (3, 4)

            # User 0 (Row 0) should be all 1.0
            assert torch.equal(local_updates_arg[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
            # User 2 (Row 2) should be all 3.0
            assert torch.equal(local_updates_arg[2], torch.tensor([3.0, 3.0, 3.0, 3.0]))