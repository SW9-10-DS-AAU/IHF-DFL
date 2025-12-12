from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import numpy as np
import torch
import torch.nn as nn

# All scoring helpers come from the challenge contract. These tests
# validate how the contract normalizes/weights participant updates across the
# three scoring strategies exposed in production (dotproduct, naive, accuracy).
from openfl.contracts.fl_challenge import (
    FLChallenge,
    calc_contribution_score_naive,
    calc_contribution_scores_accuracy,
    calc_contribution_scores_dotproduct,
)


class TinyModel(nn.Module):
    # Minimal model used to simulate local updates. The parameters are small
    # tensors so we can reason about vector math in the tests without large
    # fixtures or real training runs.
    def __init__(self, base_weight: float, noise: float = 0.0):
        super().__init__()
        values = torch.tensor([base_weight + noise, base_weight + noise], dtype=torch.float32)
        self.params = nn.Parameter(values)

    def parameters(self):
        return [self.params]


def build_challenge(strategy: str, *, use_outlier_detection: bool = False, contract=None):
    """Construct an FLChallenge with the scoring strategy under test.

    The manager, w3 connection, and contract are mocked because we only care
    about the pure Python scoring routines. Pytorch model metadata is also
    stubbed to keep test setup lightweight.
    """
    manager = MagicMock()
    manager.w3 = MagicMock()
    manager.fork = True

    configs = [contract or MagicMock(), "0xModel", 1, 2, 3, 4, 0.5, 0.1]
    pytorch_model = MagicMock()
    pytorch_model.participants = []

    experiment_config = SimpleNamespace(
        contribution_score_strategy=strategy,
        use_outlier_detection=use_outlier_detection,
    )
    return FLChallenge(manager, configs, pytorch_model, experiment_config)


def make_participant(idx: int, previous_model: nn.Module, merged_model: nn.Module):
    """Create a minimal participant stub used by scoring methods."""
    return SimpleNamespace(
        id=idx,
        address=f"0xparticipant{idx}",
        privateKey=f"priv{idx}",
        previousModel=previous_model,
        model=merged_model,
    )


def make_accuracy_contract(prev_accs, prev_losses, user_metrics):
    """Fake accuracy contract that returns historical metrics for normalization."""
    functions = SimpleNamespace()
    functions.getAllPreviousAccuraciesAndLosses = SimpleNamespace(
        call=lambda: (prev_accs, prev_losses)
    )

    def about(addr):
        accs, losses = user_metrics[addr]
        return SimpleNamespace(call=lambda: (None, accs, losses))

    functions.getAllAccuraciesAbout = about
    return SimpleNamespace(functions=functions)


class TestDotProductScoring:
    def test_low_noise_freerider_has_small_penalty(self):
        '''
        Slightly noisy freerider should score below honest peers while keeping ordering deterministic.
        In this setup, participant[1] drifts 0.01 below the merged model, so
        the dot product between their update and the global update shrinks.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        freerider = make_participant(1, TinyModel(0.99), merged)
        backup = make_participant(2, TinyModel(1.05), merged)

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct([honest, freerider, backup])

        local_updates = torch.stack([
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.99, 0.99]),
            torch.tensor([1.05, 1.05]),
        ])
        expected = calc_contribution_scores_dotproduct(
            local_updates, torch.tensor([1.0, 1.0])
        )

        assert scores == expected
        assert scores[1] < scores[0]

    def test_medium_noise_shifts_rankings(self):
        '''
        Moderate noise should demote the freerider below both honest contributors.
        Noise magnitude of 0.1 makes participant[1] diverge further from the
        merged model, so their alignment (dot product) drops compared to the others.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        freerider = make_participant(1, TinyModel(0.9), merged)
        backup = make_participant(2, TinyModel(1.1), merged)

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct([honest, freerider, backup])

        assert scores[2] > scores[0] > scores[1]

    def test_high_noise_filtered_by_mad(self):
        '''
        Outlier filtering should reduce extreme penalties while keeping honest scores positive.
        Here the freerider update is nearly opposite the merged model.
        The median absolute deviation (MAD) filter used in the dot-product path
        should clamp the negative impact so we don't over-penalize.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        freerider = make_participant(1, TinyModel(-1.0), merged)
        backup = make_participant(2, TinyModel(1.0), merged)

        challenge_no_filter = build_challenge("dotproduct", use_outlier_detection=False)
        challenge_filter = build_challenge("dotproduct", use_outlier_detection=True)

        raw_scores = challenge_no_filter._calculate_scores_dotproduct([honest, freerider, backup])
        filtered_scores = challenge_filter._calculate_scores_dotproduct([honest, freerider, backup])

        assert abs(filtered_scores[1]) <= abs(raw_scores[1])
        assert filtered_scores[0] > 0
        assert filtered_scores[2] > 0

    def test_zero_global_update_distributes_evenly(self):
        '''
        When the merged model is zero, all local updates should contribute equally.
        With a zero global baseline the dot-product denominator is constant, so
        relative ordering only depends on local update magnitudes which are
        identical here.
        '''
        merged = TinyModel(0.0)
        participants = [
            make_participant(i, TinyModel(val), merged)
            for i, val in enumerate([0.0, 0.5, 1.0])
        ]

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct(participants)

        assert scores[0] == scores[1] == scores[2]

    def test_negative_alignment_produces_negative_score(self):
        '''
        Anti-aligned updates should receive negative scores relative to honest contributors.
        participant[1] pushes in the opposite direction of the merged model,
        so its dot product is negative while honest participants remain positive.
        '''
        merged = TinyModel(1.0)
        honest = make_participant(0, TinyModel(1.0), merged)
        anti_aligned = make_participant(1, TinyModel(-1.0), merged)
        freerider = make_participant(2, TinyModel(0.9), merged)

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        scores = challenge._calculate_scores_dotproduct([honest, anti_aligned, freerider])

        assert scores[1] < 0
        assert scores[0] > scores[2] > scores[1]

    def test_scores_stable_when_participants_reordered(self):
        '''
        The dot-product calculation should be deterministic regardless of participant ordering.
        This guards against position-dependent behavior when stacking updates
        and ensures addresses map to the same scores even if the list is shuffled.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(1.0), merged),
            make_participant(1, TinyModel(0.95), merged),
            make_participant(2, TinyModel(1.05), merged),
        ]

        challenge = build_challenge("dotproduct", use_outlier_detection=False)
        baseline = challenge._calculate_scores_dotproduct(participants)

        reversed_participants = list(reversed(participants))
        reversed_scores = challenge._calculate_scores_dotproduct(reversed_participants)

        baseline_by_addr = {p.address: score for p, score in zip(participants, baseline)}
        reversed_by_addr = {p.address: score for p, score in zip(reversed_participants, reversed_scores)}

        assert baseline_by_addr == reversed_by_addr


class TestNaiveScoring:
    def test_low_noise_participants_share_equally(self):
        '''
        All contributors split rewards evenly regardless of minor noise.
        The naive strategy ignores update alignment entirely, so any small
        perturbations should still yield the same pro-rata payout.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(1.0), merged),
            make_participant(1, TinyModel(1.0, noise=0.01), merged),
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        expected = [calc_contribution_score_naive(len(participants))] * len(participants)
        assert scores == expected

    def test_medium_noise_does_not_change_share(self):
        '''
        Moderate deviations still yield equal naive scores for all participants.
        With three contributors, calc_contribution_score_naive should divide
        the 1e18 reward pool by 3 even though the input weights differ.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(0.8), merged),
            make_participant(1, TinyModel(1.0, noise=0.1), merged),
            make_participant(2, TinyModel(1.2), merged),
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        assert len(set(scores)) == 1

    def test_high_noise_freerider_still_equal(self):
        '''
        Even large noise should not alter uniform naive contributions.
        participant[1] deviates by +1.0 but naive scoring ensures all four
        receive identical integer rewards.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(0, TinyModel(1.0), merged),
            make_participant(1, TinyModel(1.0, noise=1.0), merged),
            make_participant(2, TinyModel(1.0), merged),
            make_participant(3, TinyModel(1.0), merged),
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        assert all(score == scores[0] for score in scores)

    def test_single_participant_gets_full_share(self):
        '''
        Single contributor receives the full reward pool.
        The naive helper returns 1e18 for one participant, mirroring contract
        behavior of distributing the entire reward when there is no
        competition.
        '''
        merged = TinyModel(1.0)
        participants = [make_participant(0, TinyModel(1.5), merged)]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        assert scores == [int(1e18)]

    def test_large_group_equal_distribution(self):
        '''
        Naive scoring should divide rewards uniformly across many participants.
        This guards against rounding issues when distributing to larger
        cohorts; every participant should still get the same integer amount.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(i, TinyModel(1.0 + 0.01 * i), merged)
            for i in range(10)
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        expected_score = calc_contribution_score_naive(len(participants))
        assert scores == [expected_score] * len(participants)

    def test_reward_pool_preserved_after_distribution(self):
        '''
        Naive scoring should conserve the reward pool aside from integer rounding.
        Summing the per-participant payout should equal the helper's output
        multiplied by the participant count, which mirrors on-chain behavior.
        '''
        merged = TinyModel(1.0)
        participants = [
            make_participant(i, TinyModel(0.9 + 0.02 * i), merged)
            for i in range(6)
        ]

        challenge = build_challenge("naive")
        scores = challenge._calculate_scores_naive(participants)

        per_user = calc_contribution_score_naive(len(participants))
        assert sum(scores) == per_user * len(participants)


class TestAccuracyScoring:
    def test_low_noise_freerider_scores_lower(self):
        '''
        Small accuracy/loss differences should rank the freerider last despite similar baselines.
        Each participant reports two accuracy/loss entries. The freerider has
        slightly worse metrics, so after normalizing against previous
        experiment averages they should get the smallest share of the pool.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.7, 0.7, 0.7]
        prev_losses = [0.1, 0.1, 0.1]
        metrics = {
            users[0].address: ([0.9, 0.92], [0.2, 0.22]),
            users[1].address: ([0.85, 0.86], [0.25, 0.26]),
            users[2].address: ([0.8, 0.81], [0.3, 0.31]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy(users)

        avg_prev_acc = np.mean(prev_accs)
        avg_prev_loss = np.mean(prev_losses)
        avg_accuracies = [np.mean(v[0]) for v in metrics.values()]
        avg_losses = [np.mean(v[1]) for v in metrics.values()]
        norm_acc = calc_contribution_scores_accuracy(avg_accuracies, avg_prev_acc)
        norm_loss = calc_contribution_scores_accuracy(avg_losses, avg_prev_loss)
        inverted_losses = [1 - x for x in norm_loss]
        total = sum(norm_acc) + sum(inverted_losses)
        expected = [int(((a + l) / total) * 1e18) for a, l in zip(norm_acc, inverted_losses)]

        assert sum(scores) == pytest.approx(1e18, rel=0, abs=5)
        assert scores[0] > scores[1] > scores[2]

    def test_medium_noise_freerider_penalized_by_accuracy(self):
        '''
        Slightly worse accuracy should push the noisier participant below honest peers.
        participant[1] records lower accuracies and higher losses than the
        others; normalization should reduce their final payout even though
        noise is only 0.1.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0, noise=0.1), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.6, 0.6, 0.6]
        prev_losses = [0.1, 0.1, 0.1]
        metrics = {
            users[0].address: ([0.85, 0.86], [0.26, 0.25]),
            users[1].address: ([0.55, 0.56], [0.55, 0.56]),
            users[2].address: ([0.83, 0.84], [0.28, 0.27]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy(users)

        assert scores[1] < min(scores[0], scores[2])

    def test_high_noise_freerider_losses_dominate(self):
        '''
        Poor accuracy and high loss should heavily penalize the noisy participant.
        Accuracy/loss pairs for participant[1] are intentionally extreme to
        ensure loss normalization drives their score to the minimum.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0, noise=1.0), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.6, 0.6, 0.6]
        prev_losses = [0.1, 0.1, 0.1]
        metrics = {
            users[0].address: ([0.9], [0.2]),
            users[1].address: ([0.3], [0.8]),
            users[2].address: ([0.78], [0.35]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy(users)

        assert scores[1] == min(scores)
        assert scores[0] > scores[2] > scores[1]

    def test_handles_zero_sum_differences(self):
        '''
        When all participants tie on accuracy and loss, they should each receive equal scores.
        All metrics exactly match previous experiment averages, so normalized
        accuracy/loss values should be identical for every user.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0, noise=0.1), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.8, 0.8, 0.8]
        prev_losses = [0.2, 0.2, 0.2]
        metrics = {
            users[0].address: ([0.8], [0.2]),
            users[1].address: ([0.8], [0.2]),
            users[2].address: ([0.8], [0.2]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy(users)

        assert scores[0] == scores[1] == scores[2]

    def test_accuracy_scores_return_integers_and_sum_to_pool(self):
        '''
        Accuracy-based scoring should emit integer values that collectively use the reward pool.
        Guards against regression where floating point division or rounding
        errors leak value from the 1e18 total supply allocated for contributions.
        '''
        users = [
            make_participant(0, TinyModel(1.0), TinyModel(1.0)),
            make_participant(1, TinyModel(1.0), TinyModel(1.0)),
            make_participant(2, TinyModel(1.0), TinyModel(1.0)),
        ]
        prev_accs = [0.4, 0.5, 0.6]
        prev_losses = [0.5, 0.5, 0.5]
        metrics = {
            users[0].address: ([0.7, 0.71], [0.3, 0.31]),
            users[1].address: ([0.68, 0.69], [0.32, 0.33]),
            users[2].address: ([0.65, 0.66], [0.34, 0.35]),
        }
        contract = make_accuracy_contract(prev_accs, prev_losses, metrics)
        challenge = build_challenge("accuracy", contract=contract)

        scores = challenge._calculate_scores_accuracy(users)

        assert all(isinstance(score, int) for score in scores)
        assert sum(scores) == pytest.approx(1e18, rel=0, abs=5)
