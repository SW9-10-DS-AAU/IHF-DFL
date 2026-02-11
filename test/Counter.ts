import { expect } from "chai";
import { network } from "hardhat";

const { ethers } = await network.connect();
const [owner, a, b, c] = await ethers.getSigners();

describe("A test i think", function () {
  it("A simple test of Settle", async function () {
    const model = await ethers.deployContract(
      "OpenFLModelHarness",
      [
        ethers.ZeroHash,
        1000000000000000000n,
        1000000000000000000n,
        1000000000000000000n,
        8,
        3,
        3,
        50
      ]
    );

    await model.__testInitSettleState({
      participants: [a.address, b.address, c.address],
      reputations: [1000000000000000000n, 1000000000000000000n, 1000000000000000000n],
      roundReps: [-1, 1, 1],
      nrOfVotesOfUser: [3, 3, 3],
      round: 0,
      votes: [[0, 1, 1], [0, 0, 1], [0, 1, 0]]
    });

    await model.settle();

    expect(await model.__isPunished(a.address)).to.equal(true);
  });
});
