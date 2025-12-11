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
    uint public ONE_DAY = 864e2;
    bool public testing;

    address[] public participants;
    address[] punishedAddresses;

    mapping(address => bool) public isPunished;
    mapping(address => bool) public isRegistered;
    mapping(address => uint) public GlobalReputationOf;
    mapping(address => int) public RoundReputationOf;
    mapping(address => uint8) public nrOfVotesOfUser;
    mapping(address => bool) public whitelistedForRewards;
    mapping(address => uint8) roundOfUser;
    mapping(address => mapping(address => bool)) public hasVoted;
    mapping(address => mapping(address => bool)) public votedPositiveFor;
    mapping(address => mapping(uint8 => bytes32)) public secretOf;
    mapping(address => mapping(uint8 => bytes32)) public weightsOf;
    mapping(address => uint256) personalWeight;
    mapping(uint8 => mapping(address => uint256)) public contributionScore; // round => user => score
    mapping(uint8 => mapping(address => bool)) public isContribScoreNegative;
    mapping(uint8 => uint256) public nrOfContributionScores; // round => number of submissions
    mapping(uint8 => mapping(address => mapping(address => int8)))
        public feedbackOf; // round → voter → target → score
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
        require(isRegistered[msg.sender], "SNR");
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
        require(!isRegistered[msg.sender], "SAR");
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
        int newRoundReputation
    );

    event ContributionScoreSubmitted(
        address indexed user,
        uint256 contributionScore,
        bool isContribScoreNegative
    );

    event EndRound(
        uint8 round,
        uint8 validVotes,
        uint sumOfWeights,
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
    function registrationProcess(address user) internal {
        isRegistered[user] = true;
        GlobalReputationOf[user] = msg.value;
        nrOfParticipants += 1;
        participants.push(user);
        roundOfUser[user] = 1;
        emit Registered(
            user,
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
        nrOfVotesOfUser[msg.sender] += 1;
        votesPerRound += 1;
        if (score == 1) {
            votedPositiveFor[msg.sender][target] = true;
            RoundReputationOf[target] +=
                1 *
                int(GlobalReputationOf[msg.sender]);
        }
        if (score == -1) {
            votedPositiveFor[msg.sender][target] = false;
            RoundReputationOf[target] -=
                1 *
                int(GlobalReputationOf[msg.sender]);
        }
        if (score == 0) {
            votedPositiveFor[msg.sender][target] = false;
        }
        emit Feedback(
            target,
            msg.sender,
            GlobalReputationOf[msg.sender],
            RoundReputationOf[target]
        );
    }

    function submitContributionScore(uint256 score, bool is_negative) external {
        require(isRegistered[msg.sender], "User not registered");
        require(
            contributionScore[round][msg.sender] == 0,
            "Score already submitted"
        );

        contributionScore[round][msg.sender] = score;
        isContribScoreNegative[round][msg.sender] = is_negative;
        nrOfContributionScores[round] += 1;

        emit ContributionScoreSubmitted(msg.sender, score, is_negative);
    }

    function isFeedBackRoundDone() public view returns (bool roundClosed) {
        if (nrOfParticipants == 0) {
            return false; // no participants => not done
        }

        for (uint i = 0; i < participants.length; i++) {
            address user = participants[i];
            // If a particaipant hasnt voted for everyone else wait
            if (isRegistered[user]) {
                if (nrOfVotesOfUser[participants[i]] < nrOfParticipants - 1) {
                    return false;
                }
            }
        }
        return true;
    }

    function isContributionRoundDone() public returns (bool roundClosed) {
        uint mergedUsers = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (RoundReputationOf[participants[i]] < 0) {
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
            if (roundOfUser[participants[i]] == 1) {
                GlobalReputationOf[participants[i]] =
                    GlobalReputationOf[participants[i]] -
                    freeriderPenalty;
                freeriderLock += freeriderPenalty;
            }
        }

        // Punish malicious users
        for (uint i = 0; i < participants.length; i++) {
            if (isRegistered[participants[i]]) {
                if (RoundReputationOf[participants[i]] < 0) {
                    votesPerRound -= nrOfVotesOfUser[participants[i]];

                    uint punishment = uint(
                            GlobalReputationOf[participants[i]] / punishfactor
                        );

                    if (
                        GlobalReputationOf[participants[i]] >
                        min_collateral / punishfactor
                    ) {
                        isPunished[participants[i]] = true;
                        punishedAddresses.push(participants[i]);
                        whitelistedForRewards[participants[i]] = false;

                        GlobalReputationOf[participants[i]] =
                            GlobalReputationOf[participants[i]] -
                            punishment;
                        RoundReputationOf[participants[i]] =
                            RoundReputationOf[participants[i]] -
                            int(punishment);
                            totalPunishment += punishment;
                        emit Punishment(
                            participants[i],
                            RoundReputationOf[participants[i]],
                            punishment,
                            GlobalReputationOf[participants[i]]
                        );
                    } else {
                        isRegistered[participants[i]] = false;
                        isPunished[participants[i]] = true;
                        punishedAddresses.push(participants[i]);
                        whitelistedForRewards[participants[i]] = false;

                        totalPunishment += GlobalReputationOf[participants[i]];

                        emit Disqualification(
                            participants[i],
                            RoundReputationOf[participants[i]],
                            GlobalReputationOf[participants[i]],
                            0
                        );
                        GlobalReputationOf[participants[i]] = 0;
                        nrOfParticipants -= 1;
                        delete participants[i];
                    }
                } else {
                    whitelistedForRewards[participants[i]] = true;
                }
            }
        }

        // Punish helpers of malicious users
        for (uint i = 0; i < participants.length; i++) {
            if (isRegistered[participants[i]]) {
                for (uint j = 0; j < punishedAddresses.length; j++) {
                    if (
                        votedPositiveFor[participants[i]][punishedAddresses[j]]
                    ) {
                        votedPositiveFor[participants[i]][
                            punishedAddresses[j]
                        ] = false;
                        votesPerRound -= nrOfVotesOfUser[participants[i]];
                        whitelistedForRewards[participants[i]] = false;
                        emit PassivPunishment(
                            participants[i],
                            RoundReputationOf[participants[i]],
                            0,
                            GlobalReputationOf[participants[i]]
                        );
                    }
                }
            }
        }

        // Pay back freerider 1st round stake to good users
        for (uint i = 0; i < participants.length; i++) {
            if (isRegistered[participants[i]]) {
                if (roundOfUser[participants[i]] == 1) {
                    if (whitelistedForRewards[participants[i]]) {
                        GlobalReputationOf[participants[i]] =
                            GlobalReputationOf[participants[i]] +
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
        int sumOfWeights = 0;
        uint boundedSumOfWeights = 0;
        if (votesPerRound > 0 && rewardLeft >= rewardPerRound) {
            rewardLeft -= rewardPerRound;

            uint reward = rewardPerRound;
            if (totalPunishment > 0) {
                reward += totalPunishment;
            }

            // Compute weights
            for (uint i = 0; i < participants.length; i++) {
                address user = participants[i];

                if (isRegistered[user] && whitelistedForRewards[user] && !isPunished[user]) {
                    uint weight = nrOfVotesOfUser[user] *
                        contributionScore[round][user];
                    personalWeight[user] = weight;

                    if (isContribScoreNegative[round][user]) {
                        sumOfWeights -= int(weight);
                    } else {
                        sumOfWeights += int(weight);
                    }
                }
            }
            boundedSumOfWeights = sumOfWeights < 0 ? 0 : uint(sumOfWeights);

            // Give rewards
            for (uint i = 0; i < participants.length; i++) {
                address user = participants[i];

                if (isRegistered[user] && whitelistedForRewards[user] && !isPunished[user]) {
                    uint personalReward = (reward * personalWeight[user]) /
                        boundedSumOfWeights;

                    delete whitelistedForRewards[user];
                    delete personalWeight[user];

                    if (isContribScoreNegative[round][user]) {
                        GlobalReputationOf[participants[i]] -= personalReward;
                    } else {
                        GlobalReputationOf[participants[i]] += personalReward;
                    }
                    // TODO: maybe give negative reputation
                    emit Reward(
                        user,
                        RoundReputationOf[user],
                        personalReward,
                        GlobalReputationOf[user]
                    ); // TODO: we updated win, but do we need to update roundScore?
                }
            }
        }
        emit EndRound(
            round,
            votesPerRound,
            boundedSumOfWeights,
            totalPunishment
        );

        // Reset variables
        for (uint i = 0; i < participants.length; i++) {
            if (isRegistered[participants[i]]) {
                nrOfVotesOfUser[participants[i]] = 0;
                RoundReputationOf[participants[i]] = 0;
                roundOfUser[participants[i]] += 1;
                for (uint j = 0; j < participants.length; j++) {
                    delete hasVoted[participants[i]][participants[j]];
                }
            }
        }

        round += 1;
        votesPerRound = 0;
        nrOfProvidedHashedWeights = 0;
        delete punishedAddresses;
    }

    // Exit contract
    function exitModel() public onlyRegisteredUsers feedbackRoundClosed {
        require(GlobalReputationOf[msg.sender] > 0, "NEF");
        uint val = GlobalReputationOf[msg.sender];
        GlobalReputationOf[msg.sender] = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (participants[i] == msg.sender) {
                delete participants[i];
            }
        }
        isRegistered[msg.sender] = false;
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
            if (RoundReputationOf[participants[i]] >= 0) {
                count_merged_participants += 1;
            }
        }

        previous_accuracies = new uint8[](count_merged_participants);
        previous_losses = new uint256[](count_merged_participants);
        uint8 j = 0;
        for (uint i = 0; i < participants.length; i++) {
            if (RoundReputationOf[participants[i]] >= 0) {
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
            address sender = participants[i];
            uint subCount = accuracySubmissions[round][sender].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target) {
                        // TODO: GØR whitelisted eller lign. ACCESSIBLE OG CLEAR DEN EFTER ROUND END!
                        if (RoundReputationOf[participants[i]] >= 0) {
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
            address sender = participants[i];
            uint subCount = accuracySubmissions[round][sender].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target) {
                        if (RoundReputationOf[participants[i]] >= 0) {
                            voters[idx] = sender;
                            accuracies[idx] = sub.acc[k];
                            losses[idx] = sub.loss[k];
                            idx++;
                        }
                    }
                }
            }
        }
    }

    // Fallback function parses dynamic size feedback arrays
    // @dev This allows the contract to have an arbitrary number of participants
    fallback() external {
        address[] memory ads;
        int256[] memory ints;

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
}
