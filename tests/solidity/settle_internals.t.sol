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

    function _setUserPassivePunished(address user, bool value) external {
        users[user].isPassivePunished = value;
    }

    function _setUserWhitelisted(address user, bool value) external {
        users[user].whitelistedForRewards = value;
    }

    function _setVotedPositiveFor(address voter, address target, bool value) external {
        votedPositiveFor[voter][target] = value;
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

    function testPunishMaliciousUsers_DisqualifiesLowGRSUser() public {
        uint256 lowGrs = 0.4 ether;
        model._addUser(alice, lowGrs, -1, 2, 1, false);
        model._setVotesPerRound(1);

        uint256 beforeActive = model._getActiveParticipants();
        uint256 totalPunishment = model._punishMaliciousUsers();

        assertEq(totalPunishment, lowGrs);
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
