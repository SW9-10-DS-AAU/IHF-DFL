import math
import random

import pytest
import torch
import torch.nn as nn
import numpy as np
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
from web3.exceptions import ContractLogicError

from openfl.contracts.fl_challenge import (
    FLChallenge,
    calc_contribution_scores_dotproduct,
    remove_outliers_mad,
    # calc_contribution_score,
    # calc_contribution_scores_accuracy,
    normalize_contribution_scores_new,
)
from openfl.utils.shapley import check_shapley_compliance



class DummyModel(nn.Module):
    def __init__(self, weight_value):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[weight_value, weight_value],
                                               [weight_value, weight_value]], dtype=torch.float32))

    def parameters(self):
        return super().parameters()


class TensorModel(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.params = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def parameters(self):
        return [self.params]


class TestNewContribCalcforLossOrAcc:
    full_audit_log = []

    def is_close_to_one(self, vals):
        return math.isclose(sum(vals), 1.0, rel_tol=1e-9)

    def add_to_log(self, status, metric, raw_input, result, message):
        metric_tag = "[ACC]" if metric == "accuracy" else "[LOSS]"
        log_entry = (
            f"[{status}] {metric_tag}\n"
            f"    Accuracy/Loss:                        {raw_input}\n"
            f"    Final_Calculated_Contribution:        {[round(v, 4) for v in result]}\n"
            f"    Note:                                 {message}\n"
            f"    {'-' * 45}"
        )
        TestNewContribCalcforLossOrAcc.full_audit_log.append(log_entry)

    @pytest.mark.parametrize("raw_input, baseline, metric, label", [
        ([0.1, 0.2, 0.25], 0, "accuracy", "Normal Contribution"),
        ([1, 0.2, 0.025], 0, "accuracy", "Normal Contribution"),
        ([20, -10, 15, -5], 0, "accuracy", "Mixed contributions"),
        ([-30, -10, -20], 0, "accuracy", "Negative contributions (Lower is better)"),
        ([0.2, 0.0, 0.0, -0.1], 0, "accuracy", "Mixed with null-players"),
        ([-10, -10, -10], 0, "accuracy", "Negative Symmetri"),
        ([10, 10, 10], 0, "accuracy", "Positive Symmetry"),
        ([10, -20, 0.5, 4], 0, "accuracy", "Mixed positive and heavy negative"),
        ([-2, -3, -4, 0], 0, "accuracy", "Negative with 0 as best"),
        ([5, -5, 4, -3, 0], 0, "accuracy", "Mixed with 0"),
        ([5, 0, 0, 0, 0], 0, "accuracy", "Mostly 0s with one contribution"),
        ([-10, -10, 0.00000001], 0, "accuracy", "Tiny positive"),
        ([-10, -15, -20], 0, "loss", "Normal Contribution"),
        ([-1, -0.2, -0.025], 0, "loss", "Normal Contribution"),
        ([20, -10, 15, -5], 0, "loss", "Mixed contributions"),
        ([-15, -10, -5], 0, "loss", "Loss improvements"),
        ([-10, -10, -10], 0, "loss", "Positive Symmetry"),
        ([10, 10, 10], 0, "loss", "Negative Symmetry"),
        ([30, 10, 20], 0, "loss", "Positive contributions (Lower is better)"),
        ([0.2, 0.0, 0.0, -0.1], 0, "loss", "Mixed with null-players"),
        ([0, 0.0, 0.0, 0.1], 0, "loss", "Mixed with null-players"),
        ([2, 3, 4, 0], 0, "loss", "Positive with 0 as best"),
        ([5, -5, 4, -3, 0], 0, "loss", "Mixed with 0"),
        ([0, 0, 0, -3, 0], 0, "loss", "Mostly 0s with one contribution"),
        ([10, 10, -0.00000001], 0, "loss", "Tiny positive"),
    ])

    def test_contribution_scenarios(self, raw_input, baseline, metric, label):
        # 1. Run the function to get contribution scores
        res = normalize_contribution_scores_new(raw_input, baseline, metric)

        # 2. Calculate difference from accuracy/loss compared to global last round
        diffs = [v - baseline for v in raw_input]
        if metric == "loss":
            diffs = [-1 * d for d in diffs]

        # 3. Shapley Axiom Validation
        success, violations = check_shapley_compliance(diffs, res)

        # 4. Status for log (no violations = OK, violations = AXIOM_BREAK)
        status = "OK" if success else "AXIOM_BREAK"

        self.add_to_log(status, metric, raw_input, res,
                        f"{label} | Violations: {violations if violations else 'None'}")

        assert success is True, f"Expected no violations, but got: {violations}"

    def test_sabotage_axioms_acc(self):
        """
        Verify that the Shapley compliance check is actually effective at catching violations.
        Manipulating the results to break the axioms should trigger failures in the compliance check.
        """
        raw_input = [10, 10, 0]  # Have 2 similar contributions and 1 null player
        baseline = 0
        metric = "accuracy"

        # 1. Recieve the correct contribution scores for the given input (before sabotage)
        correct_res = normalize_contribution_scores_new(raw_input, baseline, metric)
        # Expected: [0.5, 0.5, 0.0]

        # 2. SABOTAGE: Changing the scores to break the axioms
        sabotaged_res = [0.6, 0.3, 0.2]
        # Fail here:
        # - Efficiency: Sum is 1.1 (not 1.0)
        # - Symmetry: Index 0 and 1 have same contribution (10), but different score (0.6 vs 0.3)
        # - Null Player: Index 2 has contribution 0, but score 0.2

        diffs = [10, 10, 0]

        # 3. Validate sabotaged results against Shapley Axioms
        sabotaged_values, violations = check_shapley_compliance(diffs, sabotaged_res)

        # Validate correct results against Shapley Axioms
        success_valid, violations_valid = check_shapley_compliance(diffs, correct_res)


        # 4. Logs (Should provide message: "AXIOM BREAK" and list all violations)
        self.add_to_log("SABOTAGE_TEST_ACC (AXIOM BREAK)", metric, raw_input, sabotaged_res,
                        f"Expected failures | Violations: {violations}")

        # FALSE Assertions to confirm that the compliance check is working (We expect failures here)
        assert sabotaged_values is False
        assert len(violations) >= 3  # Expect atleast 3 violations

        assert success_valid is True # The original, correct results should pass the compliance check


    def test_sabotage_axioms_loss(self):
        """
        Verify that the Shapley compliance check catches violations when using 'loss' metric.
        In loss, a contribution is typically the reduction in loss (e.g., baseline - user_loss).
        """
        raw_input_loss = [0.4, 0.4, 1.0] # Have 2 similar contributions and 1 null player
        baseline_loss = 1.0
        metric = "loss"

        # 1. Receive the correct contribution scores for the given input (before sabotage)
        correct_res = normalize_contribution_scores_new(raw_input_loss, baseline_loss, metric)

        # 2. SABOTAGE: Changing the scores to break the axioms
        sabotaged_res = [0.7, 0.4, 0.2]
        # Fail here:
        # - Efficiency: Sum is 1.1 (not 1.0)
        # - Symmetry: Index 0 and 1 have same contribution (0,4), but different score (0.7 vs 0.4)
        # - Null Player: Index 2 has contribution 0, but score 0.2

        diffs = [0.6, 0.6, 0.0]

        # 3. Validate sabotaged results against Shapley Axioms
        sabotaged_values, violations = check_shapley_compliance(diffs, sabotaged_res)

        # Validate correct results against Shapley Axioms
        success_valid, violations_valid = check_shapley_compliance(diffs, correct_res)

        # 4. Logs (Should provide message: "AXIOM BREAK" and list all violations)
        self.add_to_log("SABOTAGE_TEST_LOSS (AXIOM BREAK)", metric, raw_input_loss, sabotaged_res,
                        f"Expected failures | Violations: {violations}")

        # FALSE Assertions to confirm that the compliance check is working (We expect failures here)
        assert sabotaged_values is False
        assert len(violations) >= 3  # Expect atleast 3 violations

        # The original, correct results should pass the compliance check
        assert success_valid is True

    # --- PRINT LOG ---
    def test_print_audit(self):
        print("\n" + "=" * 90)
        print(f"{'FEDERATED LEARNING AUDIT LOG':^90}")
        print("=" * 90)
        for entry in TestNewContribCalcforLossOrAcc.full_audit_log:
            print(entry)

class TestCalcContributionScore:
    def setup_method(self):
        self.aggregator = MagicMock()
        self.aggregator.model = MagicMock()
        self.aggregator.pytorch_model = MagicMock()
        self.aggregator.pytorch_model.round = 1
        self.aggregator._calculate_scores_accuracy_loss = lambda users, mad_threshold=1.1: \
            FLChallenge._calculate_scores_accuracy_loss(self.aggregator, users, mad_threshold)
        self.aggregator._calculate_scores_accuracy_only = lambda users, mad_threshold=1.1: \
            FLChallenge._calculate_scores_accuracy_only(self.aggregator, users, mad_threshold)
        self.aggregator._calculate_scores_loss_only = lambda users, mad_threshold=1.1: \
            FLChallenge._calculate_scores_loss_only(self.aggregator, users, mad_threshold)

    # Basic test case with non-zero global model
    # def test_calc_contribution_score_basic(self):
    #     """
    #     Basic test case where local and global models have distinct weights.
    #     The contribution score should be calculated correctly.
    #     """
    #     local_model = DummyModel(1.0)
    #     global_model = DummyModel(2.0)
    #     num_mergers = 2
    #
    #     score = calc_contribution_scores_dotproduct(local_model, global_model, num_mergers)
    #
    #     local_update = torch.cat([p.data.view(-1) for p in local_model.parameters()])
    #     global_update = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    #     norm_U_sq = torch.dot(global_update, global_update)
    #     expected_score = torch.dot(local_update, global_update) / (num_mergers * norm_U_sq)
    #     expected_score = int(Decimal(expected_score.item()) * Decimal('1e18'))
    #
    #     assert score == expected_score
    #
    # # Edge case where local model is identical to global model
    # def test_calc_contribution_score_identical_models(self):
    #     """
    #     Edge case where the local model is identical to the global model.
    #     The contribution score should be maximized.
    #     """
    #     local_model = DummyModel(2.0)
    #     global_model = DummyModel(2.0)
    #     num_mergers = 2
    #
    #     score = calc_contribution_scores_dotproduct(local_model, global_model, num_mergers)
    #
    #     local_update = torch.cat([p.data.view(-1) for p in local_model.parameters()])
    #     global_update = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    #     norm_U_sq = torch.dot(global_update, global_update)
    #     expected_score = torch.dot(local_update, global_update) / (num_mergers * norm_U_sq)
    #     expected_score = int(Decimal(expected_score.item()) * Decimal('1e18'))
    #
    #     assert score == expected_score
    #
    # # Edge case where global model has zero weights
    # def test_calc_contribution_score_zero_global(self):
    #     """
    #     Edge case where the global model has zero weights.
    #     The contribution score should default to 1/N.
    #     """
    #     local_model = DummyModel(1.0)
    #     global_model = DummyModel(0.0)
    #     num_mergers = 3
    #
    #     score = calc_contribution_scores_dotproduct(local_model, global_model, num_mergers)
    #     expected_score = int(Decimal(1) / Decimal(num_mergers) * Decimal('1e18'))
    #     assert score == expected_score
    #
    # # Edge case where both models have zero weights
    # def test_calc_contribution_score_both_zero(self):
    #     """
    #     Edge case where both local and global models have zero weights.
    #     The contribution score should default to 1/N.
    #     """
    #     local_model = DummyModel(0.0)
    #     global_model = DummyModel(0.0)
    #     num_mergers = 4
    #
    #     score = calc_contribution_scores_dotproduct(local_model, global_model, num_mergers)
    #     expected_score = int(Decimal(1) / Decimal(num_mergers) * Decimal('1e18'))
    #     assert score == expected_score

    @pytest.mark.parametrize("user_accuracies, user_losses, prev_accuracies, prev_losses, expected_scores", [
        (
                [[87], [85], [88]],  # User Accuracies pr user
                [[12], [11], [14]],  # User Losses pr user
                [87, 85, 88],  # Previous Accuracies
                [12, 11, 14],  # Previous Losses
                [0.2, 0.4, 0.4], # Expected scores
        ),
        (
                [[90], [85], [88]],
                [[10], [11], [14]],
                [87, 85, 88],
                [12, 11, 14],
                [1.0, -3.0, 3.0],
        ),
        (
                [[4], [-2], [12]],
                [[2], [3], [4]],
                [0, 0, 0,],
                [0, 0, 0],
                [0.25396825396825395, 0.09523809523809523, 0.6507936507936507],
        ),(
                [[0], [0], [0.1]],
                [[10], [11], [14]],
                [0, 0, 0],
                [10, 11, 14],
                [-0.07142857142857142, 0.07142857142857142, 1.0],
        ),
    ])
    def test_calculate_scores_accuracy_loss(
            self,
            user_accuracies,
            user_losses,
            prev_accuracies,
            prev_losses,
            expected_scores,
    ):
        print(f"\n--- Test: Accuracy & Loss---")

        users = []
        for i, (accs, losses) in enumerate(zip(user_accuracies, user_losses)):
            user = MagicMock()
            user.address = f"0xAddressUser{i}"
            user._accuracies = accs
            user._losses = losses
            users.append(user)

        self.aggregator.model.functions \
            .getAllPreviousAccuraciesAndLosses.return_value \
            .call.return_value = (prev_accuracies, prev_losses)

        def mock_get_accuracies_losses(address):
            user = next(u for u in users if u.address == address)
            m = MagicMock()
            m.call.return_value = (None, user._accuracies, user._losses)
            return m

        self.aggregator.model.functions \
            .getAllAccuraciesLossesAbout.side_effect = mock_get_accuracies_losses

        scores = FLChallenge._calculate_scores_accuracy_loss(
            self.aggregator, users, mad_threshold=1.1
        )
        scores_normalized = [s / 1e18 for s in scores]
        print(f"scores_normalized = {scores_normalized}")

        assert isinstance(scores, list)
        assert len(scores) == len(users)
        assert all(isinstance(s, int) for s in scores)
        assert scores != []

        if expected_scores is not None:
            assert scores_normalized == pytest.approx(expected_scores, rel=1e-9), \
                f"Expected {expected_scores}, got {scores_normalized}"


    @pytest.mark.parametrize("user_accuracies, prev_accuracies, expected_scores", [
        (
                [[87], [85], [88]],
                [87, 85, 88],
                [-0.2, -1.0, 2.2],
        ),
        (
                [[90], [85], [88]],
                [87, 85, 88],
                [1.6666666666666665, -1.0, 0.33333333333333337],
        ),
        (
                [[4], [2], [-12]],
                [0, 0, 0],
                [1.3333333333333335, 0.6666666666666667, -1.0],
        ),
        (
                [[12], [-2], [-4]],
                [0, 0, 0],
                [2.5, -0.5, -1.0]
        ),
        (
                [[90], [85], [88]],
                [0, 0, 0],
                [0.34220532319391633, 0.3231939163498099, 0.33460076045627374],
        ),
        (
                [[1.5], [1.5], [-2]],
                [0, 0, 0],
                [1.0, 1.0, -1.0],
        ),
        (
                [[0], [0], [0.1]],
                [0, 0, 0],
                [0.0, 0.0, 1.0],
        ),
        (
                [[6], [6], [6]],
                [0, 0, 0],
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        ),
        (
                [[-6], [-6], [-6]],
                [0, 0, 0],
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        ),
        (
                [[6], [5], [4]],
                [0, 0, 0],
                [0.4, 0.3333333333333333, 0.26666666666666666],
        ),
        (
                [[-6], [-4], [-2]],
                [0, 0, 0],
                [0.18181818181818182, 0.2727272727272727, 0.5454545454545454],
        ),
        (
                [[0], [0], [-0.1]],
                [0, 0, 0],
                [0.55, 0.55, -0.1],
        ),
        (
                [[1], [1], [-1], [-1], [0.1]],
                [0, 0, 0],
                [1.4285714285714286, 1.4285714285714286, -1, -1, 0.14285714285714285],
        ),
        (
                [[0], [2], [-1]],
                [0, 0, 0],
                [0.0, 2.0, -1.0],
        ),
        (
                [[0], [0], [0]],
                [0, 0, 0],
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        ),
    ])
    def test_calculate_scores_accuracy_only(self, user_accuracies, prev_accuracies, expected_scores):
        print(f"\n--- Test: Accuracy ---")

        users = []
        for i, accs in enumerate(user_accuracies):
            user = MagicMock()
            user.address = f"0xAddressUser{i}"
            user._accuracies = accs
            users.append(user)

        self.aggregator.model.functions \
            .getAllPreviousAccuraciesAndLosses.return_value \
            .call.return_value = (prev_accuracies, [])

        def mock_get_accuracies(address):
            user = next(u for u in users if u.address == address)
            m = MagicMock()
            m.call.return_value = (None, user._accuracies)
            return m

        self.aggregator.model.functions \
            .getAllAccuraciesAbout.side_effect = mock_get_accuracies

        scores = FLChallenge._calculate_scores_accuracy_only(
            self.aggregator, users, mad_threshold=1.1
        )
        scores_normalized = [s / 1e18 for s in scores]
        print(f"scores_normalized = {scores_normalized}")

        assert isinstance(scores, list)
        assert len(scores) == len(users)
        assert all(isinstance(s, int) for s in scores)
        assert scores != []

        if expected_scores is not None:
            assert scores_normalized == pytest.approx(expected_scores, rel=1e-9), \
                f"Expected {expected_scores}, got {scores_normalized}"


    @pytest.mark.parametrize("user_losses, prev_losses, expected_scores", [
        (
                [[87], [85], [88]],
                [87, 85, 88],
                [0.25, 1.25, -0.5],
        ),
        (
                [[90], [85], [88]],
                [87, 85, 88],
                [-1.0, 2.2, -0.2],
        ),
        (
                [[4], [2], [-12]],
                [0, 0, 0],
                [-1.0, -0.5, 2.5],
        ),
        (
                [[12], [-2], [-4]],
                [0, 0, 0],
                [-1.0, 0.6666666666666667, 1.3333333333333335]
        ),
        (
                [[90], [85], [88]],
                [0, 0, 0],
                [0.3245119305856833, 0.3436008676789588, 0.33188720173535796],
        ),
        (
                [[1.5], [1.5], [-2]],
                [0, 0, 0],
                [-1.0, -1.0, 3.0],
        ),
        (
                [[0], [0], [0], [0.1]],
                [0, 0, 0],
                [0.3666666666666667, 0.3666666666666667, 0.3666666666666667, -0.1],
        ),
        (
                [[6], [6], [6]],
                [0, 0, 0],
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        ),
        (
                [[-6], [-6], [-6]],
                [0, 0, 0],
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        ),
        (
                [[6], [5], [4]],
                [0, 0, 0],
                [0.2702702702702703, 0.32432432432432434, 0.40540540540540543],
        ),
        (
                [[-6], [-4], [-2]],
                [0, 0, 0],
                [0.5, 0.3333333333333333, 0.16666666666666666],
        ),
        (
                [[0], [0], [-0.1]],
                [0, 0, 0],
                [0.0, 0.0, 1.0],
        ),
        (
                [[-2], [0], [-1]],
                [0, 0, 0],
                [0.6666666666666666, 0.0, 0.3333333333333333],
        ),
        (
                [[0], [2], [-1]],
                [0, 0, 0],
                [0.0, -1.0, 2.0],
        ),
        (
                [[0], [0], [0]],
                [0, 0, 0],
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        ),
    ])
    def test_calculate_scores_loss_only(self, user_losses, prev_losses, expected_scores):
        print(f"\n--- Test: Loss ---")

        users = []
        for i, losses in enumerate(user_losses):
            user = MagicMock()
            user.address = f"0xAddressUser{i}"
            user._losses = losses
            users.append(user)

        self.aggregator.model.functions \
            .getAllPreviousAccuraciesAndLosses.return_value \
            .call.return_value = ([], prev_losses)

        def mock_get_losses(address):
            user = next(u for u in users if u.address == address)
            m = MagicMock()
            m.call.return_value = (None, user._losses)
            return m

        self.aggregator.model.functions \
            .getAllLossesAbout.side_effect = mock_get_losses

        scores = FLChallenge._calculate_scores_loss_only(
            self.aggregator, users, mad_threshold=1.1
        )
        scores_normalized = [s / 1e18 for s in scores]
        print(f"scores_normalized = {scores_normalized}")

        assert isinstance(scores, list)
        assert len(scores) == len(users)
        assert all(isinstance(s, int) for s in scores)
        assert scores != []

        if expected_scores is not None:
            assert scores_normalized == pytest.approx(expected_scores, rel=1e-9), \
                f"Expected {expected_scores}, got {scores_normalized}"


class TestCalcContributionScoresMAD:
    @pytest.mark.parametrize(
        "fl_challenge",
        [SimpleNamespace(contribution_score_strategy="dotproduct", use_outlier_detection=True)],
        indirect=True,
    )
    def test_basic_no_outliers(self, fl_challenge):
        """
        When outlier detection is enabled, _calculate_scores_dotproduct should
        still match the plain dot-product results if nothing is trimmed.
        """
        local_updates = torch.tensor([
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9]
        ])
        global_update = torch.tensor([1.0, 2.0])

        merged_model = TensorModel(global_update)

        participants = []
        for update in local_updates:
            user = MagicMock()
            user.previousModel = TensorModel(update)
            user.model = merged_model
            participants.append(user)

        scores = fl_challenge._calculate_scores_dotproduct(participants)

        expected_scores = calc_contribution_scores_dotproduct(local_updates, global_update)

        assert scores == expected_scores

    @pytest.mark.parametrize(
        "fl_challenge",
        [SimpleNamespace(contribution_score_strategy="dotproduct", use_outlier_detection=True)],
        indirect=True,
    )
    def test_trimmed_global_update_used(self, fl_challenge):
        """
        Ensure MAD-enabled scoring feeds the filtered global update into the
        dot-product calculation.
        """
        local_updates = torch.tensor([
            [1.0, 1.0],
            [2.0, 1.0],
            [100.0, 1.0]
        ])
        merged_model = TensorModel([1.0, 1.0])

        participants = []
        for update in local_updates:
            user = MagicMock()
            user.previousModel = TensorModel(update)
            user.model = merged_model
            participants.append(user)

        filtered_global_update = torch.tensor([0.0, 1.0])

        with patch.object(fl_challenge, 'trim_global_update_using_mad', return_value=filtered_global_update) as mock_trim:
            with patch('openfl.contracts.fl_challenge.calc_contribution_scores_dotproduct', return_value=[10, 20, 30]) as mock_math:
                scores = fl_challenge._calculate_scores_dotproduct(participants)

        assert scores == [10, 20, 30]

        mock_trim.assert_called_once()
        args, _ = mock_math.call_args
        local_updates_arg, global_update_arg = args

        assert torch.equal(local_updates_arg, local_updates)
        assert torch.equal(global_update_arg, filtered_global_update)

    @pytest.mark.parametrize(
        "fl_challenge",
        [SimpleNamespace(contribution_score_strategy="dotproduct", use_outlier_detection=False)],
        indirect=True,
    )
    def test_outlier_detection_disabled(self, fl_challenge):
        """
        When the flag is false, the original global update should be used and
        MAD trimming should not run.
        """
        local_updates = torch.tensor([
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0]
        ])
        merged_model = TensorModel([1.0, 1.0])

        participants = []
        for update in local_updates:
            user = MagicMock()
            user.previousModel = TensorModel(update)
            user.model = merged_model
            participants.append(user)

        with patch.object(fl_challenge, 'trim_global_update_using_mad') as mock_trim:
            with patch('openfl.contracts.fl_challenge.calc_contribution_scores_dotproduct', return_value=[1, 2, 3]) as mock_math:
                scores = fl_challenge._calculate_scores_dotproduct(participants)

        assert scores == [1, 2, 3]
        mock_trim.assert_not_called()

        args, _ = mock_math.call_args
        local_updates_arg, global_update_arg = args

        assert torch.equal(local_updates_arg, local_updates)
        assert torch.equal(global_update_arg, torch.tensor([1.0, 1.0]))

# class TestCalcContributionScoresAccuracy:
#     @pytest.mark.parametrize(
#         "fl_challenge",
#         [SimpleNamespace(contribution_score_strategy="accuracy", use_outlier_detection=True)],
#         indirect=True,
#     )
    # def test_accuracy_scores_combines_accuracy_and_loss(self, fl_challenge, mock_participants):
    #     """
    #     Accuracy scoring should normalize both accuracy and loss inputs and
    #     combine them into integer scores.
    #     """
    #     fl_challenge.model.functions.getAllPreviousAccuraciesAndLosses.call.return_value = (
    #         [50, 60], # Global Accuracy
    #         [5, 4], # Global loss
    #     )
    #
    #     user_metrics = [
    #         ([], [70, 80], [3, 4]), # Individual user accuracy and loss, over two rounds.
    #         ([], [60, 70], [5, 6]),
    #         ([], [90, 90], [1, 2]),
    #     ]
    #
    #     fl_challenge.model.functions.getAllAccuraciesAbout.return_value.call.side_effect = user_metrics
    #
    #     with patch( # Temorarily replace the following functions from fl_challenge module
    #         "openfl.contracts.fl_challenge.calc_contribution_scores_accuracy",
    #         side_effect=[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], # Mock normalized accuracy and loss returned from function, per user
    #     ) as mock_normalize, patch(
    #         "openfl.contracts.fl_challenge.remove_outliers_mad", # Outlier removal is mocked to do nothing
    #         side_effect=lambda arr, z_threshold, return_mask=False: ( # Lambda function to return the original array unchanged. If return mask is true, return mask of all true elements, so all data is kept
    #                 (arr, np.ones_like(arr, dtype=bool)) if return_mask else arr
    #         ),
    #     ):
    #         scores = fl_challenge._calculate_scores_accuracy(mock_participants) # Call _calculate_scores_accuracy, which will then use the mocked values instead of calling the actual functions which we mocked just before
    #
    #     expected = [
    #         int(Decimal(x) * Decimal("1e18"))
    #         for x in (
    #             (0.6 + 0.4) / 2, # Average normalized accuracy and inverted loss
    #             (0.3 + 0.25) / 2,
    #             (0.1 + 0.35) / 2,
    #         )
    #     ]
    #
    #     assert scores == expected
    #
    #     # Both accuracies and losses should be normalized independently
    #     calls = mock_normalize.call_args_list
    #
    #     assert len(calls) == 2
    #     acc_args, acc_mean = calls[0].args
    #     loss_args, loss_mean = calls[1].args
    #
    #     # Each user contributes their averaged accuracy across rounds
    #     # (e.g., mean([70, 80]) == 75). We normalize those averages against the
    #     # averaged historical accuracy.
    #     assert acc_args == pytest.approx([75.0, 65.0, 90.0])
    #     assert acc_mean == pytest.approx(55.0)
    #     assert loss_args == pytest.approx([3.5, 5.5, 1.5])
    #     assert loss_mean == pytest.approx(4.5)
    #
    # def test_accuracy_scores_uniform_when_no_change(self, fl_challenge, mock_participants):
    #     """
    #     When users match the historical averages, scores should split evenly
    #     after normalization and loss inversion.
    #     Check that there is no relative improvement or decline for any user
    #     """
    #
    #     fl_challenge.model.functions.getAllPreviousAccuraciesAndLosses.call.return_value = (
    #         [10, 10],
    #         [2, 2],
    #     )
    #
    #     fl_challenge.model.functions.getAllAccuraciesAbout.return_value.call.return_value = (
    #         [],
    #         [10, 10, 10],
    #         [2, 2, 2],
    #     )
    #
    #     with patch(
    #         "openfl.contracts.fl_challenge.remove_outliers_mad",
    #         side_effect=lambda arr, _: arr,
    #     ):
    #         scores = fl_challenge._calculate_scores_accuracy(mock_participants[:2])
    #
    #     expected_val = int(Decimal("0.5") * Decimal("1e18"))
    #     assert scores == [expected_val, expected_val]
    #
    # def test_accuracy_scores_handles_negative_trend(self, fl_challenge, mock_participants):
    #     """
    #     Declining accuracies relative to history should still normalize
    #     correctly and reflect higher losses.
    #     """
    #
    #     fl_challenge.model.functions.getAllPreviousAccuraciesAndLosses.call.return_value = (
    #         [60, 60],
    #         [1, 1],
    #     )
    #
    #     user_metrics = [ # Both users have worse accuracy than global
    #         ([], [55, 50], [3, 4]),
    #         ([], [45, 40], [4, 5]),
    #     ]
    #     fl_challenge.model.functions.getAllAccuraciesAbout.return_value.call.side_effect = user_metrics
    #
    #     with patch(
    #         "openfl.contracts.fl_challenge.remove_outliers_mad",
    #         side_effect=lambda arr, _: arr,
    #     ):
    #         scores = fl_challenge._calculate_scores_accuracy(mock_participants[:2])
    #
    #     # Calculate expected normalization manually
    #     norm_acc = [(52.5 - 60) / ((52.5 - 60) + (42.5 - 60)), (42.5 - 60) / ((52.5 - 60) + (42.5 - 60))]
    #     norm_loss = [(3.5 - 1) / ((3.5 - 1) + (4.5 - 1)), (4.5 - 1) / ((3.5 - 1) + (4.5 - 1))]
    #     inv_loss = [1 - x for x in norm_loss]
    #     score0 = (norm_acc[0] + inv_loss[0]) / (sum(norm_acc) + sum(inv_loss))
    #     score1 = (norm_acc[1] + inv_loss[1]) / (sum(norm_acc) + sum(inv_loss))
    #
    #     expected = [
    #         int(Decimal(score0) * Decimal("1e18")),
    #         int(Decimal(score1) * Decimal("1e18")),
    #     ]
    #
    #     assert scores == expected


class TestNormalizeContributionScores:

    def test_accuracy_function_normalizes_positive_deltas(self):
        scores = normalize_contribution_scores_new([6, 8], 4, "accuracy")
        assert scores == [pytest.approx(2 / 6), pytest.approx(4 / 6)]

    def test_accuracy_function_returns_uniform_when_sum_zero(self):
        scores = normalize_contribution_scores_new([5, 5, 5], 5, "accuracy")
        assert scores == [pytest.approx(1 / 3)] * 3

    def test_accuracy_function_raises_on_empty(self):
        with pytest.raises(Exception):
            normalize_contribution_scores_new([], 0, "accuracy")


class TestRemoveOutliersMAD:
    threshold = 1.1
    def test_returns_original_when_zero_or_small_std(self):
        arr = [5,5,5]

        result = remove_outliers_mad(arr, self.threshold)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.asarray(arr))

        arr = [4.5, 5, 5.5]

        result = remove_outliers_mad(arr, self.threshold)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.asarray(arr))

    def test_filters_values_outside_threshold(self):
        arr = [1, 1, 1, 10]

        result = remove_outliers_mad(arr, self.threshold)

        np.testing.assert_array_equal(result, np.asarray([1, 1, 1]))


    def test_filters_values_outside_threshold_prev_accuracies(self):
        arr = [50, 52, 48, 10]
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([50, 52, 48]))

        arr = [40,41,50,59,60]
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([40, 41,50,59,60]))

        arr = [40, 41, 50, 59, 60, 90] # high value
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([40, 41, 50, 59, 60]))

        arr = [5, 40, 41, 50, 59, 60, 90] # low value
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([40, 41, 50, 59, 60]))

    def test_filters_values_outside_threshold_prev_losses(self):
        arr = [43.76, 34.9, 34.9, 48.16, 40.57, 37.24]
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([43.76, 34.9, 34.9, 40.57, 37.24]))

        arr = [60, 61, 61, 61, 61, 61, 61]
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([60, 61, 61, 61, 61, 61, 61]))

        arr = [60, 61, 61, 61, 61, 61, 61]
        result = remove_outliers_mad(arr, self.threshold)
        np.testing.assert_array_equal(result, np.asarray([60, 61, 61, 61, 61, 61, 61]))


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
    @pytest.mark.parametrize(
        "fl_challenge",
        [SimpleNamespace(contribution_score_strategy="dotproduct", use_outlier_detection=True)],
        indirect=True,
    )
    def test_calculate_scores_mad_wrapper(self, fl_challenge, mock_participants):
        """
        Ensure _calculate_scores_dotproduct flattens models and feeds them to
        the dot-product scorer when MAD mode is enabled.
        """
        merged_model = DummyModel(10.0)

        for i, user in enumerate(mock_participants):
            user.previousModel = DummyModel(float(i + 1))
            user.model = merged_model

        with patch('openfl.contracts.fl_challenge.calc_contribution_scores_dotproduct') as mock_math:
            mock_math.return_value = [1000, 2000, 3000]

            scores = fl_challenge._calculate_scores_dotproduct(mock_participants)

            assert scores == [1000, 2000, 3000]

            args, _ = mock_math.call_args
            local_updates_arg = args[0]
            global_update_arg = args[1]

            assert global_update_arg.shape == (4,)
            expected_global = torch.tensor([10.0, 10.0, 10.0, 10.0])
            assert torch.equal(global_update_arg, expected_global)

            assert local_updates_arg.shape == (3, 4)

            assert torch.equal(local_updates_arg[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
            assert torch.equal(local_updates_arg[2], torch.tensor([3.0, 3.0, 3.0, 3.0]))


class TestFLChallengeWorkflow:
    # Test strategy selection for dot-product with MAD enabled
    def test_strategy_selection_dotproduct(self, mock_w3, mock_contract):
        manager = MagicMock(w3=mock_w3, fork=True)
        experiment_config = SimpleNamespace(
            contribution_score_strategy='dotproduct',
            use_outlier_detection=True,
        )
        configs = [mock_contract, "0xModel", 100, 1000, 500, 3, 0.5, 3, 0.1]
        pytorch_model = MagicMock()

        challenge = FLChallenge(manager, configs, pytorch_model, experiment_config)

        assert challenge._contribution_score_strategy == 'dotproduct'
        assert challenge.experiment_config.use_outlier_detection is True
        assert challenge._get_contribution_score_calculator() == challenge._calculate_scores_dotproduct

    # Test strategy selection for naive mode
    def test_strategy_selection_naive(self, mock_w3, mock_contract):
        manager = MagicMock(w3=mock_w3)
        experiment_config = SimpleNamespace(
            contribution_score_strategy='naive',
            use_outlier_detection=False,
        )
        configs = [mock_contract, "0xModel", 100, 1000, 500, 3, 0.5, 3, 0.1]
        pytorch_model = MagicMock()

        challenge = FLChallenge(manager, configs, pytorch_model, experiment_config)

        assert challenge._contribution_score_strategy == 'naive'
        assert challenge._get_contribution_score_calculator() == challenge._calculate_scores_naive

    # Test invalid strategy selection
    def test_strategy_selection_invalid(self, mock_w3, mock_contract):
        manager = MagicMock(w3=mock_w3)
        experiment_config = SimpleNamespace(
            contribution_score_strategy='invalid_strategy',
            use_outlier_detection=False,
        )
        configs = [mock_contract, "0xModel", 100, 1000, 500, 3, 0.5, 3, 0.1]
        pytorch_model = MagicMock()

        challenge = FLChallenge(manager, configs, pytorch_model, experiment_config)

        with pytest.raises(ValueError) as excinfo:
            challenge._get_contribution_score_calculator()
        assert "Unknown contribution score strategy" in str(excinfo.value)

    # Test hashed weights provision filtering inactive users
    def test_users_provide_hashed_weights_filters_inactive(self, fl_challenge, mock_participants):
        """
        Test that inactive users are skipped when providing hashed weights.
        Relying on conftest.py unique secrets (100, 101, 102).
        """
        mock_participants[1].attitude = "inactive"

        fl_challenge.users_provide_hashed_weights()

        assert fl_challenge.model.functions.provideHashedWeights.call_count == 2

        calls = fl_challenge.model.functions.provideHashedWeights.call_args_list
        secrets_sent = [c[0][1] for c in calls]  # args[1] is the secret

        assert mock_participants[0].secret in secrets_sent
        assert mock_participants[2].secret in secrets_sent
        assert mock_participants[1].secret not in secrets_sent

    # Test slot registration logic
    def test_user_register_slot(self, fl_challenge, mock_participants):
        """
        Test that slot registration calls keccak and the contract.
        """
        with patch('web3.Web3.solidity_keccak') as mock_keccak:
            mock_keccak.return_value = b'\x09' * 32

            fl_challenge.user_register_slot()

            assert mock_keccak.call_count == 3
            assert fl_challenge.model.functions.registerSlot.call_count == 3
            fl_challenge.model.functions.registerSlot.assert_called_with(b'\x09' * 32)

    # Test contribution score submission logic
    def test_contribution_score_submission(self, fl_challenge, mock_participants):
        """
        Test the contribution_score method (submission logic).
        """
        mock_strategy_fn = MagicMock(return_value=[100, 200, 300])

        fl_challenge._get_contribution_score_calculator = MagicMock(return_value=mock_strategy_fn)

        # Mock model / contract calls:
        # getAllAccuraciesAbout(address).call() -> (voters, accuracies, losses)
        fl_challenge.model.functions.getAllAccuraciesAbout.return_value.call.return_value = (
            ["0xvoter1", "0xvoter2"],  # voters
            [90, 95],  # accuracies
            [1, 2],  # losses
        )

        # getAllPreviousAccuraciesAndLosses().call() -> (prev_accs, prev_losses)
        fl_challenge.model.functions.getAllPreviousAccuraciesAndLosses.call.return_value = (
            [],  # prev_accs
            [],  # prev_losses
        )

        for u in mock_participants:
            u.model = DummyModel(1.0)
            u.previousModel = DummyModel(1.0)

        fl_challenge.contribution_score(mock_participants)

        assert fl_challenge.model.functions.submitContributionScore.call_count == 3

        fl_challenge.model.functions.submitContributionScore.assert_any_call(300)

        assert mock_participants[0].contribution_score == 100
        assert mock_participants[2].contribution_score == 300

    # Test naive score calculation wrapper
    def test_calculate_scores_naive_helper(self, fl_challenge, mock_participants):
        scores = fl_challenge._calculate_scores_naive(mock_participants)
        expected_val = int((Decimal(1) / Decimal(len(mock_participants))) * Decimal('1e18'))
        assert scores == [expected_val] * len(mock_participants)

    @patch('time.sleep', return_value=None)
    # Test close_round where everything goes exactly as planned
    def test_close_round_happy_path(self, mock_sleep, fl_challenge):
        """
        Test close_round where feedback and contribution rounds finish immediately.
        Relies on conftest.py handling the hybrid receipt mock.
        """
        fl_challenge.model.functions.isFeedBackRoundDone.return_value.call.return_value = True
        fl_challenge.model.functions.isContributionRoundDone.return_value.call.return_value = True

        fl_challenge.close_round()

        assert fl_challenge.model.functions.settle.call_count == 1
        assert fl_challenge.pytorch_model.round == 2

    @patch('time.sleep', return_value=None)
    # Test close_round with wait loops
    def test_close_round_wait_loops(self, mock_sleep, fl_challenge):
        """
        Test close_round logic when it has to wait (loop).
        """
        fl_challenge.model.functions.isFeedBackRoundDone.return_value.call.side_effect = [False, True]
        fl_challenge.model.functions.isContributionRoundDone.return_value.call.return_value = True

        fl_challenge.close_round()

        assert fl_challenge.model.functions.settle.call_count == 1
        assert mock_sleep.called


class TestNonForkInteractions:
    # Test registration flow when fork=False (Production mode)
    def test_register_all_users_non_fork(self, mock_w3, mock_contract, mock_participants):
        """
        Test registration flow when fork=False (Production mode).
        Should sign transactions locally and send raw transaction.
        """
        manager = MagicMock()
        manager.w3 = mock_w3
        manager.fork = False  # Switch to non-fork mode

        configs = [mock_contract, "0xModel", 100, 1000, 500, 3, 0.5, 3, 0.1]
        pytorch_model = MagicMock()
        pytorch_model.participants = mock_participants
        experiment_config = SimpleNamespace(
            contribution_score_strategy='dotproduct',
            use_outlier_detection=False,
        )

        mock_signed_tx = MagicMock()
        mock_signed_tx.rawTransaction = b'raw_tx_bytes'
        mock_w3.eth.account.sign_transaction.return_value = mock_signed_tx
        mock_w3.eth.send_raw_transaction.return_value = b'\x09' * 32

        with patch('openfl.contracts.fl_challenge.FLManager.build_non_fork_tx') as mock_build_nf:
            mock_build_nf.return_value = {'gas': 100000, 'nonce': 1}

            challenge = FLChallenge(manager, configs, pytorch_model, experiment_config)
            challenge.register_all_users()

            assert mock_w3.eth.account.sign_transaction.call_count == 3
            assert mock_w3.eth.send_raw_transaction.call_count == 3


class TestErrorHandling:
    @patch('builtins.input', return_value='')
    # Test that FRC error during feedback triggers time jump and retry
    def test_give_feedback_handles_frc_error(self, mock_input, fl_challenge, mock_w3):
        """
        Test that encountering a 'FRC' error triggers a time jump and a retry.
        """
        giver = MagicMock()
        giver.address = "0xGiver"
        giver.cheater = []
        target = MagicMock()
        target.address = "0xTarget"

        contract_func = fl_challenge.model.functions.feedback.return_value
        contract_func.transact.side_effect = [
            ContractLogicError("Execution reverted: FRC"),
            b'\x01' * 32
        ]

        fl_challenge.give_feedback(giver, target, 1)

        mock_w3.provider.make_request.assert_called_with("evm_increaseTime", [fl_challenge.config.WAIT_DELAY])

        assert contract_func.transact.call_count == 2


class TestSystemExit:
    # Test exit_system loops through participants
    def test_exit_system(self, fl_challenge, mock_participants):
        """Test the exit_system loops through participants."""
        for i in mock_participants:
            fl_challenge.w3.eth.get_balance.return_value = 0

        fl_challenge.exit_system()

        assert fl_challenge.model.functions.exitModel.call_count == 3
        assert fl_challenge.w3.eth.wait_for_transaction_receipt.call_count == 3


class TestReporting:
    # Test log parsing and round summary printing
    def test_print_round_summary(self, fl_challenge):
        """
        Test that log parsing works correctly given a mock receipt with specific events.
        """
        mock_receipt = MagicMock()

        expected_events = {
            "EndRound": [{
                "args": {
                    "round": 1,
                    "validVotes": 10,
                    # Change 'sumOfWeights' to 'sumOfWeightedContribScore'
                    "sumOfWeightedContribScore": 500,
                    "totalPunishment": 0
                }
            }],
            "Reward": [{"args": {"user": "0xUser", "roundScore": 100, "win": 50, "newReputation": 1050}}],
            "Punishment": [],
            "Disqualification": []
        }

        with patch.object(fl_challenge, 'get_events', return_value=expected_events):
            fl_challenge.print_round_summary(mock_receipt)