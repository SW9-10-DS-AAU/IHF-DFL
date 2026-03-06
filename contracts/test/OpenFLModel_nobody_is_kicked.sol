// SPDX-License-Identifier: Apache-2.0
//  ___ _   _ ____       ____  _____ _
// |_ _| | | |  _ \     |  _ \|  ___| |
//  | || |_| | |_) |____| | | | |_  | |
//  | ||  _  |  __/_____| |_| |  _| | |___
// |___|_| |_|_|        |____/|_|   |_____|
// OpenFL is a Ethereum-based reputation system to facilitate federated learning.
// This contract is part of the OpenFL research paper by Anton Wahrstätter. The contracts do only
// represent Proof-of-Concepts and have not been developed to be used in productive
// environments. Do not use them, except for testing purpose.

pragma solidity =0.8.9;

import "./OpenFLModel.sol";

contract OpenFLModel_nobody_is_kicked is OpenFLModel {

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
    {}


    function settle() public override {
        emit EndRound(
            round,
            0,
            0,
            0
        );

        // Reset variables
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered && !user.isDisqualified) {
                user.nrOfVotesFromUser = 0;
                user.roundReputation = 0;
                user.nrOfRoundsParticipated += 1;
                user.isPunished = false;
                for (uint j = 0; j < participants.length; j++) {
                    delete hasVoted[user.addr][participants[j]];
                }
            }
        }

        round += 1;
        votesPerRound = 0;
        nrOfProvidedHashedWeights = 0;
        delete punishedAddresses;
    }
}