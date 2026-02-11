// SPDX-License-Identifier: Apache-2.0
//  _______  _______  _______  _        _______  _          _______     _______
// (  ___  )(  ____ )(  ____ \( (    /|(  ____ \( \        / ___   )   (  __   )
// | (   ) || (    )|| (    \/|  \  ( || (    \/| (        \/   )  |   | (  )  |
// | |   | || (____)|| (__    |   \ | || (__    | |            /   )   | | /   |
// | |   | ||  _____)|  __)   | (\ \) ||  __)   | |          _/   /    | (/ /) |
// | |   | || (      | (      | | \   || (      | |         /   _/     |   / | |
// | (___) || )      | (____/\| )  \  || )      | (____/\  (   (__/\ _ |  (__) |
// (_______)|/       (_______/|/    )_)|/       (_______/  \_______/(_)(_______)
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

    uint public nrOfParticipants;
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

    bool public testing = true;

    // Dont change order, fl_challenge.py relies on order. Maybe use getters if bytecode size allows later
    struct User {
        int256 weightedContribScore; // 32
        uint globalReputationScore; // 32
        int roundReputation; // 32
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

    struct AccuracySubmission {
        address[] adrs;
        uint8[] acc;
        uint256[] loss;
    }

    mapping(uint8 => mapping(address => uint8)) public prev_accs;
    mapping(uint8 => mapping(address => uint256)) public prev_losses;

    // Mapping from sender to all their submissions
    mapping(uint8 => mapping(address => AccuracySubmission[]))
        private accuracySubmissions;

    modifier onlyRegisteredUsers() {
        require(users[msg.sender].isRegistered, "SNR");
        _;
    }

    modifier feedbackRoundOpened() {
        require(
            nrOfProvidedHashedWeights == nrOfParticipants ||
                roundStart + ONE_DAY < block.timestamp,
            "FRC"
        );
        _;
    }

    modifier feedbackRoundClosed() {
        require(
            nrOfProvidedHashedWeights != nrOfParticipants &&
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
        int newroundReputation
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
        int roundScore,
        uint loss,
        uint newReputation
    );

    event PassivPunishment(
        address victim,
        int roundScore,
        uint loss,
        uint newReputation
    );

    event Disqualification(
        address victim,
        int roundScore,
        uint loss,
        uint newReputation
    );

    event Reward(address user, int roundScore, uint win, uint newReputation);

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

    function clamp0Uint(int x) internal pure returns (uint) {
        return x > 0 ? uint(x) : 0;
    }

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
        users[userAddr].addr = userAddr;
        users[userAddr].isRegistered = true;
        users[userAddr].isDisqualified = true;
        users[userAddr].globalReputationScore = msg.value;
        nrOfParticipants += 1;
        participants.push(userAddr);
        users[userAddr].nrOfRoundsParticipated = 1;
        emit Registered(
            userAddr,
            msg.value,
            address(this).balance,
            nrOfParticipants
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
        int score
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
        if (nrOfParticipants == 0) {
            return false; // no participants => not done
        }

        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            // If a particaipant hasnt voted for everyone else wait
            if (user.isRegistered) {
                if (user.nrOfVotesFromUser < nrOfParticipants - 1) {
                    return false;
                }
            }
        }
        return true;
    }

    function isContributionRoundDone() public view returns (bool roundClosed) {
        uint mergedUsers = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (users[participants[i]].roundReputation < 0) {
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
        delete punishedAddresses;

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
            if (user.isRegistered) {
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
                        punishedAddresses.push(user.addr);
                        user.whitelistedForRewards = false;

                        user.globalReputationScore =
                            user.globalReputationScore -
                            punishment;

                        user.roundReputation =
                            user.roundReputation -
                            int(punishment);
                        totalPunishment += punishment;
                        emit Punishment(
                            user.addr,
                            user.roundReputation,
                            punishment,
                            user.globalReputationScore
                        );
                    } else {
                        user.isRegistered = false;
                        user.isPunished = true;
                        punishedAddresses.push(user.addr);
                        user.whitelistedForRewards = false;

                        totalPunishment += user.globalReputationScore;

                        emit Disqualification(
                            user.addr,
                            user.roundReputation,
                            user.globalReputationScore,
                            0
                        );
                        user.globalReputationScore = 0;
                        nrOfParticipants -= 1;
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
            if (user.isRegistered) {
                for (uint j = 0; j < punishedAddresses.length; j++) {
                    if (votedPositiveFor[user.addr][punishedAddresses[j]]) {
                        votedPositiveFor[user.addr][
                            punishedAddresses[j]
                        ] = false;
                        votesPerRound -= user.nrOfVotesFromUser;
                        user.whitelistedForRewards = false;
                        emit PassivPunishment(
                            user.addr,
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
            if (user.isRegistered) {
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
        int sumOfWeightedContribScore = 0;
        uint boundedSumOfWeightedContribScore = 0;
        if (votesPerRound > 0 && rewardLeft >= rewardPerRound) {
            rewardLeft -= rewardPerRound;

            uint reward = rewardPerRound;
            if (totalPunishment > 0) {
                reward += totalPunishment;
            }

            // Compute weights
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    user.isRegistered &&
                    user.whitelistedForRewards &&
                    !user.isPunished
                ) {
                    uint weight = user.nrOfVotesFromUser *
                        absUint(contributionScore[round][user.addr]);

                    user.weightedContribScore += contributionScore[round][
                        user.addr
                    ] < 0
                        ? int(weight)
                        : -int(weight);

                    sumOfWeightedContribScore += user.weightedContribScore;
                }
            }

            // check if a user should be disqualified
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    user.isRegistered &&
                    user.whitelistedForRewards &&
                    !user.isPunished
                ) {
                    boundedSumOfWeightedContribScore = sumOfWeightedContribScore <=
                        0
                        ? 1
                        : uint(sumOfWeightedContribScore);
                    //Clamping is fine here as it is used as a punishment
                    uint personalReward = (reward *
                        clamp0Uint(user.weightedContribScore)) /
                        boundedSumOfWeightedContribScore;
                    if (
                        contributionScore[round][user.addr] < 0 &&
                        (user.globalReputationScore <=
                            personalReward * punishfactorContrib)
                    ) {
                        reward += user.globalReputationScore;

                        emit Disqualification(
                            user.addr,
                            user.roundReputation,
                            user.globalReputationScore,
                            0
                        );
                        sumOfWeightedContribScore += int(
                            user.nrOfVotesFromUser *
                                absUint(contributionScore[round][user.addr])
                        );

                        user.globalReputationScore = 0;
                        nrOfParticipants -= 1;
                        user.isDisqualified = true;
                    }
                }
            }
            boundedSumOfWeightedContribScore = sumOfWeightedContribScore <= 0
                ? 1
                : uint(sumOfWeightedContribScore);
            uint redistributedPenalty = 0;
            uint positivesumOfWeightedContribScore = 0;
            // Give punishments (negative rewards) based on contribution score
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    user.isRegistered &&
                    user.whitelistedForRewards &&
                    !user.isPunished &&
                    contributionScore[round][user.addr] < 0
                ) {
                    uint personalReward = (reward *
                        clamp0Uint(user.weightedContribScore)) /
                        boundedSumOfWeightedContribScore;

                    uint penalty = personalReward * punishfactorContrib;

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

                if (
                    user.isRegistered &&
                    user.whitelistedForRewards &&
                    !user.isPunished &&
                    contributionScore[round][user.addr] > 0
                ) {
                    positivesumOfWeightedContribScore += uint(
                        user.weightedContribScore
                    );
                }
            }
            positivesumOfWeightedContribScore = positivesumOfWeightedContribScore ==
                0
                ? 1
                : positivesumOfWeightedContribScore;

            // Give rewards (or negative rewards) based on contribution score
            // Todo: No negative rewards???? What it talking about
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    user.isRegistered &&
                    user.whitelistedForRewards &&
                    !user.isPunished &&
                    contributionScore[round][user.addr] > 0
                ) {
                    uint personalReward = (reward *
                        clamp0Uint(user.weightedContribScore)) /
                        positivesumOfWeightedContribScore;

                    user.globalReputationScore += personalReward;

                    emit Reward(
                        user.addr,
                        user.roundReputation,
                        personalReward,
                        user.globalReputationScore
                    );
                }

                user.whitelistedForRewards = false;
                user.weightedContribScore = 0;
            }
        }
        emit EndRound(
            round,
            votesPerRound,
            boundedSumOfWeightedContribScore,
            totalPunishment
        );

        // Reset variables
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered) {
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
    }

    // Exit contract -- Does not work
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
        int256[] memory ints;

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

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function submitFeedbackBytesAndAccuracies(
        bytes calldata raw,
        uint8[] calldata accuracies,
        uint256[] calldata losses,
        uint8 prev_acc,
        uint256 prev_loss
    ) external {
        address[] memory ads;
        int256[] memory ints;

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

        require(
            accuracies.length == ads.length,
            "INVALID_LENGTH OF ACCURACY ARRAY"
        );
        require(losses.length == ads.length, "INVALID_LENGTH OF LOSS ARRAY");
        accuracySubmissions[round][msg.sender].push(
            AccuracySubmission({adrs: ads, acc: accuracies, loss: losses})
        );
        require(
            prev_acc >= 0 && prev_acc <= 100,
            "PREVIOUS ACCURACY NOT BETWEEN 0 AND 100"
        );
        prev_accs[round][msg.sender] = prev_acc;
        prev_losses[round][msg.sender] = prev_loss;

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }
    function getAllPreviousAccuraciesAndLosses()
        external
        view
        returns (
            uint8[] memory previous_accuracies,
            uint256[] memory previous_losses
        )
    {
        uint8 count_merged_participants = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (users[participants[i]].roundReputation >= 0) {
                count_merged_participants += 1;
            }
        }

        previous_accuracies = new uint8[](count_merged_participants);
        previous_losses = new uint256[](count_merged_participants);
        uint8 j = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (users[participants[i]].roundReputation >= 0) {
                previous_accuracies[j] = prev_accs[round][participants[i]];
                previous_losses[j] = prev_losses[round][participants[i]];
                j++;
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
            uint8[] memory accuracies,
            uint256[] memory losses
        )
    {
        uint totalCount = 0;

        // 1️⃣ First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            address senderAddr = participants[i];
            uint subCount = accuracySubmissions[round][senderAddr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    senderAddr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target) {
                        // TODO: GØR whitelisted eller lign. ACCESSIBLE OG CLEAR DEN EFTER ROUND END!
                        if (users[senderAddr].roundReputation >= 0) {
                            totalCount++;
                        }
                    }
                }
            }
        }

        // 2️⃣ Allocate arrays
        voters = new address[](totalCount);
        accuracies = new uint8[](totalCount);
        losses = new uint256[](totalCount);

        uint idx = 0;

        // 3️⃣ Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracySubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target) {
                        if (sender.roundReputation >= 0) {
                            voters[idx] = sender.addr;
                            accuracies[idx] = sub.acc[k];
                            losses[idx] = sub.loss[k];
                            idx++;
                        }
                    }
                }
            }
        }
    }

    function absUint(int256 x) internal pure returns (uint256) {
        if (x >= 0) return uint256(x);

        if (x == type(int256).min) {
            return uint256(type(int256).max) + 1;
        }

        return uint256(-x);
    }
}
