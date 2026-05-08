// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

// Inline harness mirrors contracts/harnesses/OpenFLModelHarness.sol. Kept local
// so Foundry doesn't have to resolve the harness file's "./OpenFLModel.sol"
// relative import, which is only valid under the flat-source layout that
// scripts/compile_contracts.py uses.
contract OpenFLModelHarness is OpenFLModel {
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

    struct InitParams {
        address[] participants;
        uint256[] reputations;
        int256[] roundReps;
        uint8[] nrOfVotesOfUser;
        uint8 round;
        int256[] contributionScores;
    }

    function __testInitSettleState(InitParams calldata p) external {
        delete participants;
        round = p.round;
        uint len = p.participants.length;
        for (uint i = 0; i < len; i++) {
            address voter = p.participants[i];
            _initUser(voter, p.reputations[i], p.roundReps[i], p.nrOfVotesOfUser[i]);
            contributionScore[p.round][voter] = p.contributionScores[i];
        }
        votesPerRound = uint8(len);
        nrOfActiveParticipants = len;
    }

    function _initUser(address u, uint reputation, int roundRep, uint8 votesCount) internal {
        participants.push(u);
        User storage user = users[u];
        user.isRegistered = true;
        user.isPunished = false;
        user.whitelistedForRewards = false;
        user.globalReputationScore = reputation;
        user.roundReputation = roundRep;
        user.nrOfVotesFromUser = votesCount;
        user.nrOfRoundsParticipated = 1;
        user.addr = u;
    }

//    function _setEvalScore(uint8 _round, address user, uint256 score) public {
//        evaluationScore[_round][user] = score;
//        hasSubmittedEvaluationScore[_round][user] = true;
//    }

    function _setUserGRSAtAddress(address userAddr, uint value) public {
        users[userAddr].globalReputationScore = value;
    }

    function _setUserGRSAtAddressStorage(address userAddr, uint value) public {
        User storage user = users[userAddr];
        user.globalReputationScore = value;
    }

    function _getUserGRSAtAddress(address userAddr) public view returns (uint) {
        return users[userAddr].globalReputationScore;
    }
}

contract OpenFLModelHarnessTest is Test {
    OpenFLModelHarness model;

    address a = makeAddr("a");
    address b = makeAddr("b");
    address c = makeAddr("c");

    bytes32 constant MODEL_HASH = bytes32(0);
    uint constant COLLATERAL = 1 ether;
    uint constant REWARD = 1 ether;
    uint8 constant MIN_ROUNDS = 8;
    uint8 constant PUNISH_FACTOR = 3;
    uint8 constant PUNISH_CONTRIB = 3;
    uint8 constant FREERIDER_PENALTY = 50;

    function _deploy() internal returns (OpenFLModelHarness m) {
        m = new OpenFLModelHarness(
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

    function testSettle() public {
        model = _deploy();

        address[] memory parts = new address[](3);
        parts[0] = a;
        parts[1] = b;
        parts[2] = c;

        uint256[] memory reps = new uint256[](3);
        reps[0] = 1e18;
        reps[1] = 1e18;
        reps[2] = 1e18;

        int256[] memory roundReps = new int256[](3);
        roundReps[0] = 1;
        roundReps[1] = 1;
        roundReps[2] = 1;

        uint8[] memory votes = new uint8[](3);
        votes[0] = 3;
        votes[1] = 3;
        votes[2] = 3;

        int256[] memory scores = new int256[](3);
        scores[0] = -1e18;
        scores[1] = -1e18;
        scores[2] = 3e18;

        model.__testInitSettleState(
            OpenFLModelHarness.InitParams({
                participants: parts,
                reputations: reps,
                roundReps: roundReps,
                nrOfVotesOfUser: votes,
                round: 2,
                contributionScores: scores
            })
        );

//        model._setEvalScore(2, a, 1e18);
//        model._setEvalScore(2, b, 1e18);
//        model._setEvalScore(2, c, 1e18);

        model.settle();

        emit log_named_uint("a_rep", model._getUserGRSAtAddress(a));
        emit log_named_uint("b_rep", model._getUserGRSAtAddress(b));
        emit log_named_uint("c_rep", model._getUserGRSAtAddress(c));
    }

    function testSettersAndGetters() public {
        model = _deploy();

        model._setUserGRSAtAddress(a, 20);
        assertEq(model._getUserGRSAtAddress(a), 20);

        model._setUserGRSAtAddress(a, 30);
        assertEq(model._getUserGRSAtAddress(a), 30);
    }

    function testSettersAndGettersViaStorageRef() public {
        model = _deploy();

        model._setUserGRSAtAddressStorage(a, 20);
        assertEq(model._getUserGRSAtAddress(a), 20);

        model._setUserGRSAtAddressStorage(a, 30);
        assertEq(model._getUserGRSAtAddress(a), 30);
    }

    function testFeedbackUpdatesRoundReputation() public {
        model = _deploy();

        vm.deal(a, 1 ether);
        vm.prank(a);
        model.register{value: 1 ether}();

        vm.warp(block.timestamp + 86401);
        vm.roll(block.number + 1);

        vm.prank(a);
        model.feedback(b, 1);

        // getUser tuple index 5 is nrOfVotesFromUser (uint8)
        (, , , , , uint8 votesAfter1, , , , ) = model.getUser(a);
        assertEq(uint(votesAfter1), 1);

        vm.prank(a);
        model.feedback(c, 0);

        (, , , , , uint8 votesAfter2, , , , ) = model.getUser(a);
        assertEq(uint(votesAfter2), 2);
    }
}
