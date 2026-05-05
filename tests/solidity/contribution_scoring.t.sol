// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

// Inline harness that exposes settleContributionScores() directly so tests can
// set up precise state without going through the full settle() pipeline.
// Runs in foundry evm, does not using ganache.
// This harness only exposes internal state/setup hooks needed for deterministic
// contribution-scoring tests.

// Reward logic is partially checked
// If a user is punished is checked
// If a user is disqualified is not checked...

contract ContributionScoringHarness is OpenFLModel {
    constructor(
        bytes32 _modelHash,
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty
    )
        OpenFLModel(
            _modelHash,
            _min_collateral,
            _max_collateral,
            _reward,
            _min_rounds,
            _punishfactor,
            _punishfactorContrib,
            _freeriderPenalty
        )
    {
        testing = true;
    }

    function _initEligibleUser(address user, uint grs, uint8 votesCount) external {
        participants.push(user);
        User storage u = users[user];
        u.isRegistered = true;
        u.isPunished = false;
        u.isDisqualified = false;
        u.whitelistedForRewards = true;
        u.globalReputationScore = grs;
        u.roundReputation = 1;
        u.nrOfRoundsParticipated = 1;
        u.nrOfVotesFromUser = votesCount;
        u.addr = user;
        nrOfActiveParticipants += 1;
    }

    function _setContribution(address user, int256 score) external {
        contributionScore[round][user] = score;
    }

    function _setVotesPerRound(uint8 votes) external {
        votesPerRound = votes;
    }

    function _settleContributionScores(
        uint totalPunishment,
        uint evaluationDisqualificationPool
    ) external returns (uint256) {
        return settleContributionScores(totalPunishment, evaluationDisqualificationPool);
    }

    function _getGRS(address user) external view returns (uint) {
        return users[user].globalReputationScore;
    }

    function _isPunished(address user) external view returns (bool) {
        return users[user].isPunished;
    }

    function _isDisqualified(address user) external view returns (bool) {
        return users[user].isDisqualified;
    }

    function _rewardPerRound() external view returns (uint) {
        return rewardPerRound;
    }
}

contract ContributionScoringTest is Test {
    ContributionScoringHarness model;

    address a = makeAddr("a");
    address b = makeAddr("b");

    bytes32 constant MODEL_HASH = bytes32(0);
    uint constant COLLATERAL = 1 ether;
    uint constant REWARD = 3 ether;
    uint8 constant MIN_ROUNDS = 3;
    uint8 constant PUNISH_FACTOR = 3;
    uint8 constant PUNISH_CONTRIB = 3;
    uint8 constant FREERIDER_PENALTY = 50;

    function setUp() public {
        model = new ContributionScoringHarness(
            MODEL_HASH,
            COLLATERAL,
            COLLATERAL,
            REWARD,
            MIN_ROUNDS,
            PUNISH_FACTOR,
            PUNISH_CONTRIB,
            FREERIDER_PENALTY
        );
    }

    // Verifies positive contributors are rewarded proportionally to their
    // weighted contribution score: weight = nrOfVotesFromUser * contributionScore.
    function testRewardsAreProportionalToWeightedContribution() public {
        model._initEligibleUser(a, COLLATERAL, 1);
        model._initEligibleUser(b, COLLATERAL, 1);
        model._setVotesPerRound(2);
        model._setContribution(a, 1e18);
        model._setContribution(b, 3e18);

        uint beforeA = model._getGRS(a);
        uint beforeB = model._getGRS(b);

        uint256 sum = model._settleContributionScores(0, 0);
        assertEq(sum, 4e18);

        uint afterA = model._getGRS(a);
        uint afterB = model._getGRS(b);

        assertEq(afterA - beforeA, 0.25 ether);
        assertEq(afterB - beforeB, 0.75 ether);
    }

    // Verifies negative contribution triggers punishment and that slashed value
    // is redistributed through the reward pool to eligible positive contributors.
    function testNegativeContributionIsPunishedAndRedistributed() public {
        model._initEligibleUser(a, COLLATERAL, 1);
        model._initEligibleUser(b, COLLATERAL, 1);
        model._setVotesPerRound(2);
        model._setContribution(a, -1e18);
        model._setContribution(b, 1e18);

        uint beforeA = model._getGRS(a);
        uint beforeB = model._getGRS(b);

        model._settleContributionScores(0, 0);

        uint expectedPunishment = COLLATERAL / PUNISH_CONTRIB;
        uint expectedReward = model._rewardPerRound() + expectedPunishment;

        uint afterA = model._getGRS(a);
        uint afterB = model._getGRS(b);

        assertEq(afterA, beforeA - expectedPunishment);
        assertEq(afterB, beforeB + expectedReward);
        assertTrue(model._isPunished(a));
    }

    // Verifies settleContributionScores() reverts when no strictly positive
    // weighted contribution exists (sumOfWeightedContribScore <= 0).
    // Revert: Fail and fallback
    function testRevertWhenNoPositiveWeightedContributionExists() public {
        model._initEligibleUser(a, COLLATERAL, 1);
        model._setVotesPerRound(1);
        model._setContribution(a, 0);

        vm.expectRevert("sumOfWeightedContribScore is <= 0 in settle!");
        model._settleContributionScores(0, 0);
    }

    // Verifies a sufficiently low-GRS user with negative contribution is
    // disqualified (not just punished), and their full GRS is moved to the pool.
    function testNegativeContributionCanDisqualifyUser() public {
        uint lowGrs = 0.4 ether;
        model._initEligibleUser(a, lowGrs, 1);
        model._initEligibleUser(b, COLLATERAL, 1);
        model._setVotesPerRound(2);
        model._setContribution(a, -1e18);
        model._setContribution(b, 1e18);

        uint beforeB = model._getGRS(b);
        model._settleContributionScores(0, 0);
        uint afterB = model._getGRS(b);

        assertTrue(model._isDisqualified(a));
        assertEq(model._getGRS(a), 0);
        assertEq(afterB, beforeB + model._rewardPerRound() + lowGrs);
    }
}

