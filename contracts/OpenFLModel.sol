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

contract OpenFLModel {
    bytes32 public modelHash;

    uint8 public round = 0;
    uint8 public votesPerRound;
    uint8 public punishfactor;
    uint8 public min_rounds;
    uint8 public punishfactorContrib;

    uint public nrOfActiveParticipants;
    uint public nrOfProvidedHashedWeights;
    uint public initTS;
    uint public min_collateral;
    uint public max_collateral;
    uint public totalReward;
    uint public rewardPerRound;
    uint public rewardLeft;
    uint public roundStart;
    uint public contributionStart;
    uint public freeriderPenalty;
    uint constant ONE_DAY = 864e2;

    address[] public participants;
    address[] punishedAddresses;

    bool public testing = false;

    // Dont change order, fl_challenge.py relies on order. Maybe use getters if bytecode size allows later
    struct User {
        int256 weightedContribScore; // 32
        uint globalReputationScore; // 32
        int256 roundReputation; // 32
        address addr; // 20
        uint8 nrOfRoundsParticipated; // 1
        uint8 nrOfVotesFromUser; // 1
        bool isPunished; // 1
        bool isRegistered; // 1
        bool whitelistedForRewards; // 1
        bool isDisqualified; // 1
    }

    mapping(address => User) public users;

    mapping(address => mapping(address => bool)) public hasVoted;
    mapping(address => mapping(address => bool)) public votedPositiveFor;
    mapping(address => mapping(uint8 => bytes32)) public secretOf;
    mapping(address => mapping(uint8 => bytes32)) public weightsOf;
    mapping(uint8 => mapping(address => int256)) public contributionScore; // round => user => score
    mapping(uint8 => uint256) public nrOfContributionScores; // round => number of submissions

    struct AccuracyLossSubmission {
        address[] adrs;
        uint16[] acc;
        uint16[] loss;
    }

    struct AccuracySubmission {
        address[] adrs;
        uint16[] acc;
    }

    struct LossSubmission {
        address[] adrs;
        uint16[] loss;
    }


    mapping(uint8 => mapping(address => uint16)) public prev_accs;
    mapping(uint8 => mapping(address => uint16)) public prev_losses;

    // Mapping from sender to all their submissions
    mapping(uint16 => mapping(address => AccuracyLossSubmission[]))
        private accuracyLossSubmissions;

    mapping(uint16 => mapping(address => AccuracySubmission[]))
        private accuracySubmissions;

    mapping(uint16 => mapping(address => LossSubmission[]))
        private lossSubmissions;


    modifier onlyRegisteredUsers() {
        require(users[msg.sender].isRegistered, "SNR");
        _;
    }

    modifier feedbackRoundOpened() {
        require(
            nrOfProvidedHashedWeights == nrOfActiveParticipants ||
                roundStart + ONE_DAY < block.timestamp,
            "FRC"
        );
        _;
    }

    modifier feedbackRoundClosed() {
        require(
            nrOfProvidedHashedWeights != nrOfActiveParticipants &&
                roundStart + ONE_DAY > block.timestamp,
            "NA"
        );
        require(weightsOf[msg.sender][round] == bytes32(0), "WFE");
        _;
    }

    modifier onlyValidTargets(address target) {
        require(target != msg.sender, "SET");
        require(!hasVoted[msg.sender][target], "VAC");
        _;
    }

    modifier onlyNotYetRegisteredUsers() {
        require(!users[msg.sender].isRegistered, "SAR");
        _;
    }

    modifier hasNotYetProvidedWeights() {
        require(weightsOf[msg.sender][round] == bytes32(0), "SAP");
        _;
    }

    event FederatedLerningModelDeployed(
        uint initTS,
        uint max_collateral,
        uint min_collateral,
        uint total_reward,
        uint8 min_rounds,
        uint freerider_fee
    );

    event Registered(
        address user,
        uint reputationValue,
        uint totalCollateral,
        uint numberOfContributers
    );

    event Feedback(
        address target,
        address user,
        uint globalReputation,
        int256 newRoundReputation
    );

    event ContributionScoreSubmitted(
        address indexed user,
        int256 contributionScore
    );

    event EndRound(
        uint8 round,
        uint8 validVotes,
        uint sumOfWeightedContribScore,
        uint totalPunishment
    );

    event Punishment(
        address victim,
        int256 roundScore,
        uint loss,
        uint newReputation
    );

    event PassivPunishment(
        address victim,
        int256 roundScore,
        uint loss,
        uint newReputation
    );

    event Disqualification(
        address victim,
        int256 roundScore,
        uint loss,
        uint newReputation
    );

    event Reward(address user, int256 roundScore, uint win, uint newReputation);

    constructor(
        bytes32 _modelHash,
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty
    ) payable {
        // Initialize Contract
        initTS = block.timestamp;
        roundStart = block.timestamp;
        modelHash = _modelHash;
        min_collateral = _min_collateral;
        max_collateral = _max_collateral;
        totalReward = _reward;
        min_rounds = _min_rounds;
        punishfactor = _punishfactor;
        punishfactorContrib = _punishfactorContrib;
        freeriderPenalty = (min_collateral * _freeriderPenalty) / 100;
        rewardPerRound = totalReward / min_rounds;
        rewardLeft = totalReward;

        emit FederatedLerningModelDeployed(
            initTS,
            min_collateral,
            max_collateral,
            totalReward,
            min_rounds,
            freeriderPenalty
        );
    }

    function setTesting(bool _testing) external {
        testing = _testing;
    }

    // Register participants
    function register() public payable onlyNotYetRegisteredUsers {
        require(
            msg.value >= min_collateral && msg.value <= max_collateral,
            "NWR"
        );
        registrationProcess(msg.sender);
    }

    // Register initiator of model
    function register(
        address initiator
    ) public payable onlyNotYetRegisteredUsers {
        require(
            msg.value >= min_collateral && msg.value <= max_collateral,
            "NWR"
        );
        // Require Staking here
        registrationProcess(initiator);
    }

    // Registration helper
    function registrationProcess(address userAddr) internal {
        User storage user = users[userAddr];
        user.isRegistered = true;
        user.globalReputationScore = msg.value;
        user.nrOfRoundsParticipated = 1;
        user.addr = userAddr;
        nrOfActiveParticipants += 1;
        participants.push(userAddr);
        emit Registered(
            user.addr,
            msg.value,
            address(this).balance,
            nrOfActiveParticipants
        );
    }

    // Register Slot
    function registerSlot(
        bytes32 _secret
    ) public onlyRegisteredUsers hasNotYetProvidedWeights {
        secretOf[msg.sender][round] = _secret;
    }

    // Timestamp weights to the chain
    function provideHashedWeights(
        bytes32 hashedWeights,
        uint salt
    ) public onlyRegisteredUsers hasNotYetProvidedWeights {
        require(
            secretOf[msg.sender][round] ==
                keccak256(abi.encodePacked(hashedWeights, salt, msg.sender)),
            "NKS"
        );
        weightsOf[msg.sender][round] = hashedWeights;
        nrOfProvidedHashedWeights += 1;
    }

    function feedback(
        address target,
        int256 score
    )
        public
        virtual
        onlyRegisteredUsers
        onlyValidTargets(target)
        feedbackRoundOpened
    {
        //(address target, int score) = abi.decode(data, (address, int));
        hasVoted[msg.sender][target] = true;
        users[msg.sender].nrOfVotesFromUser += 1;
        votesPerRound += 1;
        if (score == 1) {
            votedPositiveFor[msg.sender][target] = true;
            users[target].roundReputation +=
                1 *
                int(users[msg.sender].globalReputationScore);
        }
        if (score == -1) {
            votedPositiveFor[msg.sender][target] = false;
            users[target].roundReputation -=
                1 *
                int(users[msg.sender].globalReputationScore);
        }
        if (score == 0) {
            votedPositiveFor[msg.sender][target] = false;
        }
        emit Feedback(
            target,
            msg.sender,
            users[msg.sender].globalReputationScore,
            users[target].roundReputation
        );
    }

    function submitContributionScore(int256 score) external {
        require(users[msg.sender].isRegistered, "User not registered");
        require(
            contributionScore[round][msg.sender] == 0,
            "Score already submitted"
        );

        contributionScore[round][msg.sender] = score;
        nrOfContributionScores[round] += 1;

        emit ContributionScoreSubmitted(msg.sender, score);
    }

    function isFeedBackRoundDone() public view returns (bool roundClosed) {
        if (nrOfActiveParticipants == 0) {
            return false; // no participants => not done
        }

        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            // If a particaipant hasnt voted for everyone else wait
            if (user.isRegistered && !user.isDisqualified) {
                if (user.nrOfVotesFromUser < nrOfActiveParticipants - 1) {
                    return false;
                }
            }
        }
        return true;
    }

    function isContributionRoundDone() public returns (bool roundClosed) {
        uint mergedUsers = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (
                users[participants[i]].roundReputation < 0 &&
                !users[participants[i]].isDisqualified
            ) {
                mergedUsers++;
            }
        }
        if (nrOfContributionScores[round] < mergedUsers) {
            return false;
        }

        return true;
    }

    function settle() public {
        uint totalPunishment;
        uint freeriderLock; // A global total of sum of freerider penalties

        // First round users pay their anti-freerider fee
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.nrOfRoundsParticipated == 1) {
                user.globalReputationScore =
                    user.globalReputationScore -
                    freeriderPenalty;
                freeriderLock += freeriderPenalty;
            }
        }

        // Punish malicious users
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered && !user.isDisqualified) {
                if (user.roundReputation < 0) {
                    votesPerRound -= user.nrOfVotesFromUser;

                    uint punishment = uint(
                        user.globalReputationScore / punishfactor
                    );

                    if (
                        user.globalReputationScore >
                        min_collateral / punishfactor
                    ) {
                        user.isPunished = true;
                        punishedAddresses.push(participants[i]);
                        user.whitelistedForRewards = false;

                        user.globalReputationScore =
                            user.globalReputationScore -
                            punishment;
                        user.roundReputation =
                            user.roundReputation -
                            int(punishment);
                        totalPunishment += punishment;
                        emit Punishment(
                            participants[i],
                            user.roundReputation,
                            punishment,
                            user.globalReputationScore
                        );
                    } else {
                        user.isRegistered = false;
                        user.isPunished = true;
                        punishedAddresses.push(participants[i]);
                        user.whitelistedForRewards = false;

                        totalPunishment += user.globalReputationScore;

                        emit Disqualification(
                            user.addr,
                            user.roundReputation,
                            user.globalReputationScore,
                            0
                        );
                        user.globalReputationScore = 0;
                        nrOfActiveParticipants -= 1;
                        user.isDisqualified = true;
                    }
                } else {
                    user.whitelistedForRewards = true;
                }
            }
        }

        // Punish helpers of malicious users
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered && !user.isDisqualified) {
                for (uint j = 0; j < punishedAddresses.length; j++) {
                    if (
                        votedPositiveFor[participants[i]][punishedAddresses[j]]
                    ) {
                        votedPositiveFor[participants[i]][
                            punishedAddresses[j]
                        ] = false;
                        votesPerRound -= user.nrOfVotesFromUser;
                        user.whitelistedForRewards = false;
                        emit PassivPunishment(
                            participants[i],
                            user.roundReputation,
                            0,
                            user.globalReputationScore
                        );
                    }
                }
            }
        }

        // Pay back freerider 1st round stake to good users
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered && !user.isDisqualified) {
                if (user.nrOfRoundsParticipated == 1) {
                    if (user.whitelistedForRewards) {
                        user.globalReputationScore =
                            user.globalReputationScore +
                            freeriderPenalty;
                        freeriderLock -= freeriderPenalty;
                    } else {
                        totalPunishment += freeriderPenalty;
                        freeriderLock -= freeriderPenalty;
                    }
                }
            }
        }

        // Devide reward between every user who provided (non-malicious) feedback
        // Pay back freeriderLock funds to good users
        // First round users pay their anti-freerider fee
        int256 sumOfWeightedContribScore = 0;
        uint BoundedSumOfWeightedContribScore = 0;
        if (votesPerRound > 0 && rewardLeft >= rewardPerRound) {
            rewardLeft -= rewardPerRound;

            uint reward = rewardPerRound;
            if (totalPunishment > 0) {
                reward += totalPunishment;
            }

            // Compute weights
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (_isEligibleForRewards(user)) {
                    int256 weight = int256(uint(user.nrOfVotesFromUser)) *
                        contributionScore[round][user.addr];
                    user.weightedContribScore = weight;

                    sumOfWeightedContribScore += weight;
                }
            }

            // check if a user should be disqualified
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (_isEligibleForRewards(user)) {
                    BoundedSumOfWeightedContribScore = sumOfWeightedContribScore <=
                        0
                        ? 1
                        : uint(sumOfWeightedContribScore);
                    uint personalReward = (reward *
                        absUint(user.weightedContribScore)) /
                        BoundedSumOfWeightedContribScore;
                    if (
                        contributionScore[round][user.addr] < 0 &&
                        (user.globalReputationScore <=
                            personalReward * punishfactorContrib)
                    ) {
                        reward += user.globalReputationScore;

                        emit Disqualification(
                            participants[i],
                            user.roundReputation,
                            user.globalReputationScore,
                            0
                        );
                        sumOfWeightedContribScore +=
                            int(uint256(user.nrOfVotesFromUser)) *
                            contributionScore[round][user.addr];

                        user.globalReputationScore = 0;
                        nrOfActiveParticipants -= 1;
                        user.isDisqualified = true;
                    }
                }
            }
            BoundedSumOfWeightedContribScore = sumOfWeightedContribScore <= 0
                ? 1
                : uint(sumOfWeightedContribScore);
            uint redistributedPenalty = 0;
            uint positiveSumOfWeights = 0;
            // Give punishmentts (negative rewards) based on contribution score
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (_isEligibleForRewards(user) && contributionScore[round][user.addr] < 0
                ) {
                    uint personalPunishment = (reward *
                        absUint(user.weightedContribScore)) /
                        BoundedSumOfWeightedContribScore;

                    uint penalty = personalPunishment * punishfactorContrib;

                    // Todo: Penalty can become huge. This sets it to only ever empty GRS, but this is probably problematic
                    // Should probably be clamped to buyIn(not a variable)/punishfactor or an alternative penalty
                    // calculation that does not exceed buyIn
                    if (penalty >= user.globalReputationScore) {
                        penalty = user.globalReputationScore;
                    }

                    user.globalReputationScore -= penalty;
                    redistributedPenalty += penalty;

                    emit Reward(
                        user.addr,
                        user.roundReputation,
                        0,
                        user.globalReputationScore
                    );
                }
            }
            reward += redistributedPenalty;

            // Compute total weight of positive contributors
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (_isEligibleForRewards(user) && contributionScore[round][user.addr] >= 0) {
                    positiveSumOfWeights += uint(user.weightedContribScore);
                }
            }
            positiveSumOfWeights = positiveSumOfWeights == 0
                ? 1
                : positiveSumOfWeights;

            // Give rewards based on positive contribution score
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (_isEligibleForRewards(user) && contributionScore[round][user.addr] >= 0) { // NOTE: This refactor adds the case of !user.Disqualified, in contrast to before)
                    uint personalReward = (reward *
                        uint(user.weightedContribScore)) / positiveSumOfWeights;

                    user.globalReputationScore += personalReward;

                    emit Reward(
                        user.addr,
                        user.roundReputation,
                        personalReward,
                        user.globalReputationScore
                    );
                }

                delete user.whitelistedForRewards;
                delete user.weightedContribScore;
            }
        }
        emit EndRound(
            round,
            votesPerRound,
            BoundedSumOfWeightedContribScore,
            totalPunishment
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

    // Exit contract - Not safe, gaurds exists but will crash the contract if not met, exits should be queued?
    function exitModel() public onlyRegisteredUsers feedbackRoundClosed {
        require(users[msg.sender].globalReputationScore > 0, "NEF");
        uint val = users[msg.sender].globalReputationScore;
        users[msg.sender].globalReputationScore = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (participants[i] == msg.sender) {
                delete participants[i];
            }
        }
        users[msg.sender].isRegistered = false;
        payable(address(msg.sender)).transfer(val);
    }

    function submitFeedbackBytes(bytes calldata raw) external {
        address[] memory ads;
        int16[] memory ints;

        (ads, ints) = parseRaw(raw);

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function submitFeedbackBytesAndAccuraciesLosses(
        bytes calldata raw,
        uint16[] calldata accuracies,
        uint16[] calldata losses,
        uint16 prev_acc,
        uint16 prev_loss
    ) external {
        address[] memory ads;
        int16[] memory ints;

        (ads, ints) = parseRaw(raw);

        require(
            accuracies.length == ads.length,
            "INVALID_LENGTH OF ACCURACY ARRAY"
        );
        require(
            losses.length == ads.length, "INVALID_LENGTH OF LOSS ARRAY");
            accuracyLossSubmissions[round][msg.sender].push(
            AccuracyLossSubmission({adrs: ads, acc: accuracies, loss: losses})
        );
        require(
            prev_acc >= 0 && prev_acc <= 10000,
            "PREVIOUS ACCURACY NOT BETWEEN 0 AND 10000 in submitFeedbackBytesAndAccuraciesLosses"
        );
        require (
            prev_loss >= 0 && prev_loss <= 10000,
            "PREVIOUS LOSS NOT BETWEEN 0 AND 10000 in submitFeedbackBytesAndAccuraciesLosses"
        );
        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }


     function submitFeedbackBytesAndAccuracies(
        bytes calldata raw,
        uint16[] calldata accuracies,
        uint16 prev_acc
    ) external {
        address[] memory ads;
        int16[] memory ints;

         (ads, ints) = parseRaw(raw);

        require(
            accuracies.length == ads.length,
            "INVALID_LENGTH OF ACCURACY ARRAY"
        );

         accuracySubmissions[round][msg.sender].push(
            AccuracySubmission({adrs: ads, acc: accuracies})
        );
        require(
            prev_acc >= 0 && prev_acc <= 10000,
            "PREVIOUS ACCURACY NOT BETWEEN 0 AND 10000 submitFeedbackBytesAndAccuracies"
        );
        prev_accs[round][msg.sender] = prev_acc;

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }


    function submitFeedbackBytesAndLosses(
        bytes calldata raw,
        uint16[] calldata losses,
        uint16 prev_loss
    ) external {
        address[] memory ads;
        int16[] memory ints;

        (ads, ints) = parseRaw(raw);

        require(
            losses.length == ads.length, "INVALID_LENGTH OF LOSS ARRAY");
            lossSubmissions[round][msg.sender].push(
            LossSubmission({adrs: ads, loss: losses})
        );

        prev_losses[round][msg.sender] = prev_loss;

        require(
            prev_loss >= 0 && prev_loss <= 10000,
            "PREVIOUS LOSS NOT BETWEEN 0 AND 10000 in submitFeedbackBytesAndLosses"
        );

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function parseRaw(bytes calldata raw)
        internal
        pure
        returns (address[] memory ads, int16[] memory ints)
    {

        assembly {
            let tmp := 0
            let tmp2 := 0

            // offset inside `raw` starts at raw.offset
            let offset := raw.offset
            // adsCount = calldatasize / 0x34
            let adsCount := div(raw.length, 0x34)

            // allocate memory for addresses array
            ads := mload(0x40)
            mstore(0x40, add(ads, add(0x20, mul(adsCount, 0x20))))
            mstore(ads, adsCount)

            // load addresses (20 bytes each)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp := calldataload(offset)
                tmp := shr(96, tmp)
                mstore(add(add(ads, 0x20), mul(i, 0x20)), tmp)
                offset := add(offset, 0x14)
            }

            // allocate memory for ints array
            ints := mload(0x40)
            mstore(0x40, add(ints, add(0x20, mul(adsCount, 0x20))))
            mstore(ints, adsCount)

            // load int256 values (32 bytes each)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp2 := calldataload(offset)
                mstore(add(add(ints, 0x20), mul(i, 0x20)), tmp2)
                offset := add(offset, 0x20)
            }
        }
    }


    function getAllPreviousAccuraciesAndLosses()
        external
        view
        returns (
            uint16[] memory previous_accuracies,
            uint16[] memory previous_losses
        )
    {
        uint8 count_merged_participants = 0;
        for (uint i = 0; i < participants.length; i++) {
            User storage u = users[participants[i]];
            if (u.isRegistered && !u.isDisqualified && u.roundReputation >= 0) {
                count_merged_participants += 1;
            }
        }

        previous_accuracies = new uint16[](count_merged_participants);
        previous_losses = new uint16[](count_merged_participants);
        uint8 j = 0;
        for (uint i = 0; i < participants.length; i++) {
            User storage u = users[participants[i]];
            if (u.isRegistered && !u.isDisqualified && u.roundReputation >= 0) {
                previous_accuracies[j] = prev_accs[round][participants[i]];
                previous_losses[j] = prev_losses[round][participants[i]];
                j++;
            }
        }
    }

    function getAllAccuraciesLossesAbout(
        address target
    )
        external
        view
        returns (
            address[] memory voters,
            uint16[] memory accuracies,
            uint16[] memory losses
        )
    {
        uint totalCount = 0;

        // 1️. First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracyLossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracyLossSubmission storage sub = accuracyLossSubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) { // TODO: GØR whitelisted eller lign. ACCESSIBLE OG CLEAR DEN EFTER ROUND END!
                        totalCount++;
                    }
                }
            }
        }

        // 2. Allocate arrays
        voters = new address[](totalCount);
        accuracies = new uint16[](totalCount);
        losses = new uint16[](totalCount);

        uint idx = 0;

        // 3. Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracyLossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracyLossSubmission storage sub = accuracyLossSubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                            voters[idx] = sender.addr;
                            accuracies[idx] = sub.acc[k];
                            losses[idx] = sub.loss[k];
                            idx++;
                    }
                }
            }
        }
    }



    function getAllAccuraciesAbout(
        address target
    )
        external
        view
        returns (
            address[] memory voters,
            uint16[] memory accuracies
        )
    {
        uint totalCount = 0;

        // 1️. First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracySubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                            totalCount++;
                    }
                }
            }
        }

        // 2. Allocate arrays
        voters = new address[](totalCount);
        accuracies = new uint16[](totalCount);

        uint idx = 0;

        // 3. Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracySubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                            voters[idx] = sender.addr;
                            accuracies[idx] = sub.acc[k];
                            idx++;
                    }
                }
            }
        }
    }


    function getAllLossesAbout(
        address target
    )
        external
        view
        returns (
            address[] memory voters,
            uint16[] memory losses
        )
    {
        uint totalCount = 0;

        // 1️. First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = lossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                LossSubmission storage sub = lossSubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                            totalCount++;
                    }
                }
            }
        }

        // 2. Allocate arrays
        voters = new address[](totalCount);
        losses = new uint16[](totalCount);

        uint idx = 0;

        // 3. Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = lossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                LossSubmission storage sub = lossSubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                            voters[idx] = sender.addr;
                            losses[idx] = sub.loss[k];
                            idx++;
                    }
                }
            }
        }
    }

    function _isEligibleVoter(User storage sender) internal view returns (bool) {
        return sender.isRegistered && !sender.isDisqualified && sender.roundReputation >= 0;
    }

    function _isEligibleForRewards(User storage user)
        internal
        view
        returns (bool)
    {
        return (
            user.isRegistered &&
            user.whitelistedForRewards &&
            !user.isPunished &&
            !user.isDisqualified
        );
    }


    // Fallback function parses dynamic size feedback arrays
    // @dev This allows the contract to have an arbitrary number of participants
    fallback() external {
        address[] memory ads;
        int16[] memory ints;

        assembly {
            let tmp := 0
            let tmp2 := 0

            // Skip : function selector    : 0x4 bytes
            let offset := 0x00

            // Compute the number of addresses :
            // ((array length - 0x04) - 0x20) / 0x14
            // ((array length - sizeof(function Selector)) - sizeof(uint256)) / sizeof(address)
            let adsCount := div(calldatasize(), 0x34)

            // Allocate memory for the address array
            ads := mload(0x40)
            mstore(0x40, add(ads, add(0x20, mul(adsCount, 0x20))))

            // Set the size of the array
            mstore(ads, adsCount)

            // Get an address from calldata on each iteration :
            // loads 0x20 bytes from calldata starting at offset : calldata[offset: offset + 0x20)
            // shift value by 96 bits (12 bytes) to the right to keep only the relevant portion (first 20 bytes)
            // store that value at ads[i]
            // increments calldata offset by 0x14 (20 bytes)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp := calldataload(offset)
                tmp := shr(96, tmp)
                mstore(add(add(ads, 0x20), mul(i, 0x20)), tmp)
                offset := add(offset, 0x14)
            }

            // Allocate memory for the address array
            ints := mload(0x40)
            mstore(0x40, add(ints, add(0x20, mul(adsCount, 0x20))))

            // Set the size of the array
            mstore(ints, adsCount)

            // Get an address from calldata on each iteration :
            // loads 0x20 bytes from calldata starting at offset : calldata[offset: offset + 0x20)
            // store that value at ads[i]
            // increments calldata offset by 0x20 (32 bytes)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp2 := calldataload(offset)
                mstore(add(add(ints, 0x20), mul(i, 0x20)), tmp2)
                offset := add(offset, 0x20)
            }
        }

        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function getUser(
        address u
    )
        external
        view
        returns (
            address,
            int256,
            uint,
            int,
            uint8,
            uint8,
            bool,
            bool,
            bool,
            bool
        )
    {
        User storage user = users[u];
        return (
            user.addr,
            user.weightedContribScore,
            user.globalReputationScore,
            user.roundReputation,
            user.nrOfRoundsParticipated,
            user.nrOfVotesFromUser,
            user.isPunished,
            user.isRegistered,
            user.whitelistedForRewards,
            user.isDisqualified
        );
    }

    function absUint(int x) public pure returns (uint) {
        return x >= 0 ? uint(x) : uint(-x);
    }
}
