// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

/// @notice A mock version of OpenFLModel for gas testing the fallback only
contract OpenFLModelMock is OpenFLModel {
    constructor(
        bytes32 modelHash,
        uint minCollateral,
        uint maxCollateral,
        uint reward,
        uint8 minRounds,
        uint8 punishFactor,
        uint8 freeriderPenalty
    ) OpenFLModel(modelHash, minCollateral, maxCollateral, reward, minRounds, punishFactor, freeriderPenalty) {}

    /// @notice Override feedback to skip all logic/modifiers
    function feedback(address, int) public override {
        // do nothing
    }
}

contract FallbackGasTest is Test {
    OpenFLModel model;

    address[] users;
    int256[] scores;

    uint256 constant N = 6; // change this to test different batch sizes


    function setUp() public {
        // Deploy a dummy model — constructor args do not matter for gas measurement
        model = new OpenFLModel(
            bytes32("testhash"),
            uint(1e18),
            uint(1.8e18),
            uint(1e18),
            3,
            3,
            50
        );

        model.setTesting(true);

        users = new address[](N);
        scores = new int256[](N);

        for (uint256 i = 0; i < N; i++) {
            users[i] = address(uint160(i + 1));
            scores[i] = int256(i);
        }
    }

    function testFallbackGas() public {
        bytes memory data = buildPacked(users, scores);

        // call fallback
        (bool ok,) = address(model).call(data);
        require(ok, "fallback call failed");
    }

    function testSubmitFeedbackBytesGas() public {
    bytes memory data = buildPacked(users, scores);

    // call the new function directly
    model.submitFeedbackBytes(data);
}

    // -------------------------------------------------------------
    // Helper: Encode calldata exactly as your fallback expects
    // -------------------------------------------------------------
    function buildPacked(address[] memory a, int256[] memory v)
        internal pure
        returns (bytes memory out)
    {
        require(a.length == v.length, "length mismatch");

        uint256 n = a.length;
        uint256 total = n * (20 + 32);

        out = new bytes(total);
        uint256 offset = 0;

        // Write 20-byte addresses
        for (uint256 i = 0; i < n; i++) {
            uint256 addr = uint256(uint160(a[i]));

            assembly {
                // store left-padded address → only last 20 bytes matter
                mstore(add(add(out, 0x20), offset), shl(96, addr))
            }
            offset += 20;
        }

        // Write 32-byte ints
        for (uint256 i = 0; i < n; i++) {
            int256 val = v[i];

            assembly {
                mstore(add(add(out, 0x20), offset), val)
            }
            offset += 32;
        }
    }
}
