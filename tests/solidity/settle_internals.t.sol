// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";


// - collectFreeriderFees
//  - deducts fee for first-round users and resets punish flags
// - punishMaliciousUsers
//  - punishment branch and disqualification branch
// - punishHelpers
//  - helper gets passive-punished, vote count zeroed, votedPositiveFor cleared
// - paybackFreeriders
//  - whitelisted first-round users get payback, non-whitelisted add to additionalPunishment
// - _disqualifyUser
//  - flags + GRS zeroing + active participant decrement
// - settleContributionScores (Eq. 11)
//  - negative weight costs reward * |w| / sum of positive weights
//  - loss >= GRS clamps GRS to 0 and removes the user, threshold removal stays
//  - penalties and forfeited stakes are carried into the next round's reward

contract SettleInternalsHarness is OpenFLModel {
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

    function _addUser(
        address user,
        uint256 grs,
        int256 roundRep,
        uint8 roundsParticipated,
        uint8 votesFromUser,
        bool whitelisted
    ) external {
        participants.push(user);
        User storage u = users[user];
        u.addr = user;
        u.isRegistered = true;
        u.isPunished = false;
        u.isDisqualified = false;
        u.isPassivePunished = false;
        u.whitelistedForRewards = whitelisted;
        u.globalReputationScore = grs;
        u.roundReputation = roundRep;
        u.nrOfRoundsParticipated = roundsParticipated;
        u.nrOfVotesFromUser = votesFromUser;
        nrOfActiveParticipants += 1;
    }

    function _setVotesPerRound(uint8 votes) external {
        votesPerRound = votes;
    }

    function _setUserPunished(address user, bool value) external {
        users[user].isPunished = value;
    }

    function _setGRS(address user, uint256 value) external {
        users[user].globalReputationScore = value;
    }

    function _setUserPassivePunished(address user, bool value) external {
        users[user].isPassivePunished = value;
    }

    function _setUserWhitelisted(address user, bool value) external {
        users[user].whitelistedForRewards = value;
    }

    function _setVotedPositiveFor(address voter, address target, bool value) external {
        votedPositiveFor[voter][target] = value;
    }

    function _setRound(uint8 value) external {
        round = value;
    }

    function _setContributionScore(uint8 forRound, address user, int256 score) external {
        contributionScore[forRound][user] = score;
        hasSubmittedContributionScore[forRound][user] = true;
    }

    function _setVotesFromUser(address user, uint8 votes) external {
        users[user].nrOfVotesFromUser = votes;
    }

    function _settleContributionScores(uint256 totalPunishment) external returns (uint256) {
        return settleContributionScores(totalPunishment);
    }

    function _getCarriedReward() external view returns (uint256) {
        return carriedReward;
    }

    function _collectFreeriderFees() external returns (uint256) {
        return collectFreeriderFees();
    }

    function _punishMaliciousUsers() external returns (uint256) {
        return punishMaliciousUsers();
    }

    function _punishHelpers() external {
        punishHelpers();
    }

    function _paybackFreeriders(uint256 freeriderLock) external returns (uint256) {
        return paybackFreeriders(freeriderLock);
    }

    function _disqualifyUser(address user) external {
        _disqualifyUser(users[user]);
    }

    function _getGRS(address user) external view returns (uint256) {
        return users[user].globalReputationScore;
    }

    function _isDisqualified(address user) external view returns (bool) {
        return users[user].isDisqualified;
    }

    function _isRegistered(address user) external view returns (bool) {
        return users[user].isRegistered;
    }

    function _isPunished(address user) external view returns (bool) {
        return users[user].isPunished;
    }

    function _isPassivePunished(address user) external view returns (bool) {
        return users[user].isPassivePunished;
    }

    function _isWhitelisted(address user) external view returns (bool) {
        return users[user].whitelistedForRewards;
    }

    function _getVotesFromUser(address user) external view returns (uint8) {
        return users[user].nrOfVotesFromUser;
    }

    function _getVotesPerRound() external view returns (uint8) {
        return votesPerRound;
    }

    function _getActiveParticipants() external view returns (uint256) {
        return nrOfActiveParticipants;
    }

    function _getFreeriderPenalty() external view returns (uint256) {
        return freeriderPenalty;
    }

    function _isVotedPositiveFor(address voter, address target) external view returns (bool) {
        return votedPositiveFor[voter][target];
    }

    function _punishedCount() external view returns (uint256) {
        return punishedAddresses.length;
    }
}

contract SettleInternalsTest is Test {
    SettleInternalsHarness model;

    address alice = makeAddr("alice");
    address bob = makeAddr("bob");
    address carol = makeAddr("carol");

    bytes32 constant MODEL_HASH = bytes32(0);
    uint256 constant COLLATERAL = 1 ether;
    uint256 constant REWARD = 3 ether;
    uint8 constant MIN_ROUNDS = 3;
    uint8 constant PUNISH_FACTOR = 3;
    uint8 constant PUNISH_CONTRIB = 3;
    uint8 constant FREERIDER_PENALTY_PCT = 50;

    function setUp() public {
        model = new SettleInternalsHarness(
            MODEL_HASH,
            COLLATERAL,
            COLLATERAL,
            REWARD,
            MIN_ROUNDS,
            PUNISH_FACTOR,
            PUNISH_CONTRIB,
            FREERIDER_PENALTY_PCT
        );
    }

    function testCollectFreeriderFees_DeductsAndResetsFlags() public {
        model._addUser(alice, COLLATERAL, 0, 1, 0, false);
        model._setUserPunished(alice, true);
        model._setUserPassivePunished(alice, true);

        uint256 penalty = model._getFreeriderPenalty();
        uint256 lockAmount = model._collectFreeriderFees();

        assertEq(lockAmount, penalty);
        assertEq(model._getGRS(alice), COLLATERAL - penalty);
        assertFalse(model._isPunished(alice));
        assertFalse(model._isPassivePunished(alice));
    }

    function testPunishMaliciousUsers_PunishesNegativeRoundRep() public {
        model._addUser(alice, 3 ether, -1, 2, 2, false);
        model._setVotesPerRound(5);

        uint256 totalPunishment = model._punishMaliciousUsers();

        assertEq(totalPunishment, 1 ether);
        assertEq(model._getGRS(alice), 2 ether);
        assertTrue(model._isPunished(alice));
        assertEq(model._getVotesFromUser(alice), 0);
        assertEq(model._getVotesPerRound(), 3);
        assertEq(model._punishedCount(), 1);
    }

    function testPunishMaliciousUsers_DisqualifiesZeroGRSUser() public {
        model._addUser(alice, COLLATERAL, -1, 2, 1, false);
        model._setVotesPerRound(1);

        assertFalse(model._isDisqualified(alice));
        assertTrue(model._isRegistered(alice));

        model._setGRS(alice, 0);

        uint256 beforeActive = model._getActiveParticipants();
        uint256 totalPunishment = model._punishMaliciousUsers();

        assertEq(totalPunishment, 0);
        assertTrue(model._isDisqualified(alice));
        assertFalse(model._isRegistered(alice));
        assertTrue(model._isPunished(alice));
        assertEq(model._getGRS(alice), 0);
        assertEq(model._getActiveParticipants(), beforeActive - 1);
    }

    function testPunishHelpers_MarksSupportersAndClearsVotes() public {
        model._addUser(alice, 2 ether, -1, 2, 0, false);
        model._addUser(bob, COLLATERAL, 1, 2, 2, false);
        model._setVotesPerRound(2);
        model._setVotedPositiveFor(bob, alice, true);

        model._punishMaliciousUsers();
        model._punishHelpers();

        assertTrue(model._isPassivePunished(bob));
        assertEq(model._getVotesFromUser(bob), 0);
        assertEq(model._getVotesPerRound(), 0);
        assertFalse(model._isVotedPositiveFor(bob, alice));
    }

    function testPaybackFreeriders_RewardsWhitelistedAndPunishesOthers() public {
        model._addUser(alice, COLLATERAL, 1, 1, 0, true);
        model._addUser(bob, COLLATERAL, 1, 1, 0, false);
        model._addUser(carol, COLLATERAL, 1, 2, 0, true);

        uint256 penalty = model._getFreeriderPenalty();
        uint256 beforeAlice = model._getGRS(alice);
        uint256 beforeBob = model._getGRS(bob);
        uint256 beforeCarol = model._getGRS(carol);

        uint256 additionalPunishment = model._paybackFreeriders(2 * penalty);

        assertEq(model._getGRS(alice), beforeAlice + penalty);
        assertEq(model._getGRS(bob), beforeBob);
        assertEq(model._getGRS(carol), beforeCarol);
        assertEq(additionalPunishment, penalty);
    }

    // alice and bob share the positive weights: 1e18 * 2 votes each, so the sum is 4e18.
    // carol is the user under test. Reward per round is 1 ether.
    function _setUpSettlement(uint256 carolGRS, uint8 carolVotes, int256 carolScore) internal {
        model._addUser(alice, 2 ether, 0, 2, 2, true);
        model._addUser(bob, 2 ether, 0, 2, 2, true);
        model._addUser(carol, carolGRS, 0, 2, carolVotes, true);
        model._setVotesPerRound(6);
        model._setContributionScore(0, alice, 1e18);
        model._setContributionScore(0, bob, 1e18);
        model._setContributionScore(0, carol, carolScore);
    }

    function testSettle_NegativeScoreCostsProportionalToRewardAndIsNotRemoved() public {
        // w = 1 * -5e18; loss = 1e18 * 5e18 / 4e18 = 1.25 ether. The old rule (GRS * |c| / 3) would have removed carol.
        _setUpSettlement(10 ether, 1, -5e18);

        model._settleContributionScores(0);

        assertEq(model._getGRS(carol), 10 ether - 1.25 ether);
        assertFalse(model._isDisqualified(carol));
        assertTrue(model._isRegistered(carol));
    }

    function testSettle_LossAtLeastGRSClampsToZeroAndRemoves() public {
        // loss 1.25 ether exceeds GRS 1 ether: no underflow revert
        _setUpSettlement(1 ether, 1, -5e18);

        model._settleContributionScores(0);

        assertEq(model._getGRS(carol), 0);
        assertTrue(model._isDisqualified(carol));
        assertFalse(model._isRegistered(carol));
    }

    function testSettle_GRSAtOrBelowThresholdIsRemoved() public {
        // threshold = min_collateral / 3 = 0.333.. ether; loss = 1e18 * 0.1e18 / 4e18 = 0.025 ether
        _setUpSettlement(0.3 ether, 1, -0.1e18);

        model._settleContributionScores(0);

        assertTrue(model._isDisqualified(carol));
        assertEq(model._getGRS(carol), 0);
    }

    function testSettle_GRSAboveThresholdAfterLossIsKept() public {
        // loss 0.025 ether, GRS 0.5 -> 0.475 ether, above the 0.333 threshold
        _setUpSettlement(0.5 ether, 1, -0.1e18);

        model._settleContributionScores(0);

        assertEq(model._getGRS(carol), 0.475 ether);
        assertFalse(model._isDisqualified(carol));
    }

    function testSettle_PositivePayoutsAreUnchangedByPenalties() public {
        _setUpSettlement(10 ether, 1, -5e18);

        model._settleContributionScores(0);

        // R = 1 ether, weights 2e18 of 4e18 each: 0.5 ether, penalty is not part of R
        assertEq(model._getGRS(alice), 2 ether + 0.5 ether);
        assertEq(model._getGRS(bob), 2 ether + 0.5 ether);
    }

    function testSettle_PenaltyIsCarriedToNextRound() public {
        _setUpSettlement(10 ether, 1, -5e18);

        model._settleContributionScores(0);

        assertEq(model._getCarriedReward(), 1.25 ether);
    }

    function testSettle_ForfeitedStakeIsCarriedWhenLossExceedsGRS() public {
        _setUpSettlement(1 ether, 1, -5e18);

        model._settleContributionScores(0);

        // the whole 1 ether stake is forfeited, not the 1.25 ether loss
        assertEq(model._getCarriedReward(), 1 ether);
    }

    function testSettle_ThresholdRemovalCarriesLossAndRemainingStake() public {
        _setUpSettlement(0.3 ether, 1, -0.1e18);

        model._settleContributionScores(0);

        assertEq(model._getCarriedReward(), 0.3 ether);
    }

    function testSettle_CarriedRewardIsPaidInNextRoundAndReset() public {
        _setUpSettlement(10 ether, 1, -5e18);
        model._settleContributionScores(0);
        model._setRound(1);
        model._setUserWhitelisted(alice, true);
        model._setUserWhitelisted(bob, true);
        model._setContributionScore(1, alice, 1e18);
        model._setContributionScore(1, bob, 1e18);
        uint256 aliceBefore = model._getGRS(alice);

        model._settleContributionScores(0);

        // R = 1 ether + 1.25 ether carried; alice weight 2e18 of 4e18
        assertEq(model._getGRS(alice) - aliceBefore, uint256(2.25 ether) / 2);
        assertEq(model._getCarriedReward(), 0);
    }

    function testDisqualifyUser_SetsFlagsAndDecrementsActiveCount() public {
        model._addUser(alice, COLLATERAL, -1, 1, 0, false);
        uint256 beforeActive = model._getActiveParticipants();

        model._disqualifyUser(alice);

        assertTrue(model._isDisqualified(alice));
        assertFalse(model._isRegistered(alice));
        assertTrue(model._isPunished(alice));
        assertEq(model._getGRS(alice), 0);
        assertEq(model._getActiveParticipants(), beforeActive - 1);
    }
}
