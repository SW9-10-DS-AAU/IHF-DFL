// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

// Inline harness that exposes settleEvaluationScores() directly so tests can
// set up precise state without going through the full settle() pipeline.

// Runs in foundry evm, does not using ganache.

contract EvalVotingHarness is OpenFLModel {
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

    // Add a pre-whitelisted, eligible user directly (bypasses register flow)
    function _initEligibleUser(address user, uint grs) external {
        participants.push(user);
        User storage u = users[user];
        u.isRegistered = true;
        u.isPunished = false;
        u.isDisqualified = false;
        u.isPassivePunished = false;
        u.whitelistedForRewards = true;
        u.globalReputationScore = grs;
        u.roundReputation = 1;
        u.nrOfRoundsParticipated = 1;
        u.addr = user;
        nrOfActiveParticipants += 1;
    }

    function _setEvalScore(address user, uint256 score) external {
        evaluationScore[round][user] = score;
        hasSubmittedEvaluationScore[round][user] = true;
    }

    function _setPassivePunished(address user, bool val) external {
        users[user].isPassivePunished = val;
    }

    function _settleEvalScores() external returns (uint) {
        return settleEvaluationScores();
    }

    function _getGRS(address user) external view returns (uint) {
        return users[user].globalReputationScore;
    }

    function _isDisqualified(address user) external view returns (bool) {
        return users[user].isDisqualified;
    }
}

contract EvalVotingTest is Test {
    EvalVotingHarness model;

    address a = makeAddr("a");
    address b = makeAddr("b");

    bytes32 constant MODEL_HASH = bytes32(0);
    uint constant COLLATERAL = 1 ether;
    uint constant REWARD = 1 ether;
    uint8 constant MIN_ROUNDS = 1;
    uint8 constant PUNISH_FACTOR = 3;
    uint8 constant PUNISH_CONTRIB = 3;
    uint8 constant FREERIDER_PENALTY = 50;

    // staking_min_grs = COLLATERAL / PUNISH_CONTRIB = 1e18 / 3
    // disq_threshold  = COLLATERAL / PUNISH_FACTOR  = 1e18 / 3
    uint constant INIT_GRS = 1 ether;

    function setUp() public {
        model = new EvalVotingHarness(
            MODEL_HASH, COLLATERAL, COLLATERAL, REWARD,
            MIN_ROUNDS, PUNISH_FACTOR, PUNISH_CONTRIB, FREERIDER_PENALTY
        );
    }

    // evalScore > 1e18 → evaluation_reward > staking_min_grs → GRS increases
    function testNormalRewardIncreasesGRS() public {
        model._initEligibleUser(a, INIT_GRS);
        model._setEvalScore(a, 2e18);

        uint before = model._getGRS(a);
        model._settleEvalScores();

        uint _after = model._getGRS(a);
        assertGt(_after, before);
    }

    // evalScore = 1e18 → evaluation_reward == staking_min_grs → net GRS change is zero
    function testBreakEvenLeavesGRSUnchanged() public {
        model._initEligibleUser(a, INIT_GRS);
        model._setEvalScore(a, 1e18);

        uint before = model._getGRS(a);
        model._settleEvalScores();
        uint _after = model._getGRS(a);

        assertEq(_after, before);
    }

    // isPassivePunished + surplus reward → surplus is confiscated into disqualification pool
    function testPassivePunishmentCapSurplusGoesToPool() public {
        model._initEligibleUser(a, INIT_GRS);
        model._setEvalScore(a, 2e18);
        model._setPassivePunished(a, true);

        uint pool = model._settleEvalScores();

        assertGt(pool, 0);
    }

    // evalScore = 0 and low GRS → new_global_rep < disq_threshold → user is disqualified
    // GRS = 0.5e18; staking_min_grs = 1e18/3 ≈ 0.333e18
    // new_global_rep = 0.5e18 + 0 - 0.333e18 ≈ 0.167e18 < disq_threshold → disqualified
    function testLowEvalScoreDisqualifiesUser() public {
        model._initEligibleUser(a, 0.5 ether);
        model._setEvalScore(a, 0);

        model._settleEvalScores();

        assertTrue(model._isDisqualified(a));
    }

    // Eligible user who never submitted an evaluation score causes a revert
    function testRevertIfEvalScoreNotSubmitted() public {
        model._initEligibleUser(a, INIT_GRS);

        vm.expectRevert("Evaluation score not submitted for user");
        model._settleEvalScores();
    }

    // Submitting an evaluation score twice reverts on the second call
    function testRevertOnDoubleEvalScoreSubmit() public {
        vm.deal(a, 1 ether);
        vm.prank(a);
        model.register{value: 1 ether}();

        // contribScore = 0 so the contribution-score guard doesn't fire on retry
        vm.prank(a);
        model.submitContributionScoreAndVotingEvaluation(0, 1e18);

        vm.expectRevert("Evaluation score already submitted");
        vm.prank(a);
        model.submitContributionScoreAndVotingEvaluation(0, 1e18);
    }
}
