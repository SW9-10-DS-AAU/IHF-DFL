// contracts/test/OpenFLModelHarness.sol
pragma solidity =0.8.9;

import "../OpenFLModel.sol";

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
    }

    function __testInitSettleState(InitParams calldata p) external {
        delete participants;

        round = p.round;
        uint len = p.participants.length;

        for (uint i = 0; i < len; i++) {
            address voter = p.participants[i];
            //int8[] calldata voterVotes = p.votes[i];

            _initUser(
                voter,
                p.reputations[i],
                p.roundReps[i],
                p.nrOfVotesOfUser[i]
            );

            //_initVotes(p.participants, voterVotes, voter, p.round);
        }

        votesPerRound = uint8(len);

        nrOfParticipants = len;
    }

    function _initUser(
        address u,
        uint reputation,
        int roundRep,
        uint8 votesCount
    ) internal {
        participants.push(u);

        User storage user = users[u];
        user.isRegistered = true;
        user.isPunished = false;
        user.whitelistedForRewards = false;

        user.globalReputationScore = reputation;
        user.roundReputation = roundRep;
        user.nrOfVotesFromUser = votesCount;
        user.nrOfRoundsParticipated = 1;
        user.weightedContribScore = 0;
        user.addr = u;
    }

    function _initVotes(
        address[] calldata participants_,
        int8[] calldata voterVotes,
        address voter,
        uint8 round_
    ) internal {
        uint len = participants_.length;

        for (uint j = 0; j < len; j++) {
            //feedbackOf[round_][voter][participants_[j]] = voterVotes[j];
        }
    }

    function __isPunished(address userAddr) external view returns (bool) {
        if (punishedAddresses.length > 0) {
            return true;
        }

        for (uint i = 0; i < punishedAddresses.length; i++) {
            if (punishedAddresses[i] == userAddr) {
                return true;
            }
        }
        return false;
    }

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
