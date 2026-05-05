// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

// Minimal harness that adds read-only helpers for inspecting the invariant.
contract NoEtherLostHarness is OpenFLModel {
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
        payable
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

    // Sum of globalReputationScore across every tracked participant slot.
    function totalGRS() external view returns (uint256 total) {
        for (uint i = 0; i < participants.length; i++) {
            total += users[participants[i]].globalReputationScore;
        }
    }

    function participantCount() external view returns (uint) {
        return participants.length;
    }
}

contract NoEtherLost is Test {
    NoEtherLostHarness model;

    address user1 = makeAddr("user1");
    address user2 = makeAddr("user2");
    address user3 = makeAddr("user3");

    bytes32 constant MODEL_HASH        = bytes32(0);
    uint constant    COLLATERAL        = 1 ether;
    uint constant    REWARD            = 3 ether;
    uint8 constant   MIN_ROUNDS        = 3;
    uint8 constant   PUNISH_FACTOR     = 3;
    uint8 constant   PUNISH_CONTRIB    = 3;
    uint8 constant   FREERIDER_PENALTY = 0; // 0 % keeps balance arithmetic exact

    // setUp() is run before each test, ensuring there are 3 registered users available for each test
    function setUp() public {
        model = new NoEtherLostHarness{value: REWARD}(
            MODEL_HASH, COLLATERAL, COLLATERAL, REWARD,
            MIN_ROUNDS, PUNISH_FACTOR, PUNISH_CONTRIB, FREERIDER_PENALTY
        );

        vm.deal(user1, COLLATERAL);
        vm.deal(user2, COLLATERAL);
        vm.deal(user3, COLLATERAL);
        vm.prank(user1); model.register{value: COLLATERAL}();
        vm.prank(user2); model.register{value: COLLATERAL}();
        vm.prank(user3); model.register{value: COLLATERAL}();
    }

    // All-positive feedback + scores for one round
    function playRound() internal {
        vm.warp(block.timestamp + 86401);
        vm.roll(block.number + 1);

        vm.prank(user1); model.feedback(user2, 1);
        vm.prank(user1); model.feedback(user3, 1);
        vm.prank(user2); model.feedback(user1, 1);
        vm.prank(user2); model.feedback(user3, 1);
        vm.prank(user3); model.feedback(user1, 1);
        vm.prank(user3); model.feedback(user2, 1);

        vm.prank(user1); model.submitContributionScoreAndVotingEvaluation(1e18, 1e18);
        vm.prank(user2); model.submitContributionScoreAndVotingEvaluation(1e18, 1e18);
        vm.prank(user3); model.submitContributionScoreAndVotingEvaluation(1e18, 1e18);
    }

    // Feedback only
    function playRoundWithoutScores() internal {
        vm.warp(block.timestamp + 86401);
        vm.roll(block.number + 1);

        vm.prank(user1); model.feedback(user2, 1);
        vm.prank(user1); model.feedback(user3, 1);
        vm.prank(user2); model.feedback(user1, 1);
        vm.prank(user2); model.feedback(user3, 1);
        vm.prank(user3); model.feedback(user1, 1);
        vm.prank(user3); model.feedback(user2, 1);
    }

    function assertExactInvariant(string memory label) internal view {
        assertEq(
            address(model).balance,
            model.totalGRS() + model.rewardLeft(),
            label
        );
    }

    // After settle(), up to 1 wei of rounding dust per participant is acceptable
    function assertPostSettleInvariant(string memory label) internal view {
        uint balance            = address(model).balance;
        uint sumGRS             = model.totalGRS();
        uint rewardL            = model.rewardLeft();
        uint numberParticipants = model.participantCount(); // THRESHOLD: 1 wei per participant
        assertGe(balance, sumGRS + rewardL, string(abi.encodePacked(label, ": ether destroyed")));
        assertLe(balance - sumGRS - rewardL, numberParticipants, string(abi.encodePacked(label, ": dust exceeds 1 wei per participant")));
    }


    // Invariant holds after registration
    function testInvariantAfterRegistration() public {
        // balance = REWARD + 3*COLLATERAL
        // totalGRS = 3*COLLATERAL
        // rewardLeft = REWARD
        assertExactInvariant("balance != totalGRS + rewardLeft after registration");
    }

    // exitModel returns collateral exactly; reward pool stays untouched
    function testRegisterAndExitReturnsExactCollateral() public {
        vm.prank(user1); model.exitModel();
        vm.prank(user2); model.exitModel();
        vm.prank(user3); model.exitModel();

        assertEq(user1.balance, COLLATERAL, "user1 did not recover full collateral");
        assertEq(user2.balance, COLLATERAL, "user2 did not recover full collateral");
        assertEq(user3.balance, COLLATERAL, "user3 did not recover full collateral");
        assertEq(address(model).balance, REWARD, "Contract should only hold undistributed reward");
    }

    // settle() reverts when evaluation scores are missing
    function testSettleRevertsWithoutEvaluationScore() public {
        playRoundWithoutScores();

        vm.expectRevert("Evaluation score not submitted for user");
        model.settle();
    }

    // settle() does not destroy ether
    function testSettleDoesNotDestroyEther() public {
        assertExactInvariant("Pre-settle: balance != totalGRS + rewardLeft");
        playRound();
        model.settle();
        assertPostSettleInvariant("Post-settle");
    }

    // Punishment redistributes slashed ether to good users — none is destroyed
    // user2 receives two negative votes (score == -1) and ends up punished
    // user2 does not need to submit evaluation scores because _isEligibleForRewards() == false
    function testGriefingPunishmentPreservesInvariant() public {
        vm.warp(block.timestamp + 86401);
        vm.roll(block.number + 1);

        // user2 gets two negative votes → roundReputation = -2e18 → punished
        vm.prank(user1); model.feedback(user2, -1);
        vm.prank(user1); model.feedback(user3,  1);
        vm.prank(user2); model.feedback(user1,  1);
        vm.prank(user2); model.feedback(user3,  1);
        vm.prank(user3); model.feedback(user1,  1);
        vm.prank(user3); model.feedback(user2, -1);

        vm.prank(user1); model.submitContributionScoreAndVotingEvaluation(1e18, 1e18);
        vm.prank(user3); model.submitContributionScoreAndVotingEvaluation(1e18, 1e18);

        model.settle();

        assertPostSettleInvariant("griefing: after punishment of user2");
    }

    // Invariant holds across three consecutive rounds without re-registering
    function testInvariantHoldsAcrossMultipleRounds() public {
        playRound();
        model.settle();
        assertPostSettleInvariant("runde 1");

        playRound();
        model.settle();
        assertPostSettleInvariant("runde 2");

        playRound();
        model.settle();
        assertPostSettleInvariant("runde 3");
    }

    // Total ether is conserved through deploy → settle → all exit
    // Every wei that entered (REWARD + 3*COLLATERAL) must end up either in a
    // user wallet or still in the contract
    function testTotalEtherConservedAfterSettleAndExit() public {
        uint totalEtherInPlay = address(model).balance;

        playRound();
        model.settle();

        vm.prank(user1); model.exitModel();
        vm.prank(user2); model.exitModel();
        vm.prank(user3); model.exitModel();

        uint returnedToUsers     = user1.balance + user2.balance + user3.balance;
        uint remainingInContract = address(model).balance;

        assertEq(
            returnedToUsers + remainingInContract,
            totalEtherInPlay,
            "Total ether not conserved: ether was created or destroyed"
        );
    }
}