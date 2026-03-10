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
      roundReps: [1, 1, 1],
      nrOfVotesOfUser: [3, 3, 3],
      round: 2,
      contributionScores: [-1000000000000000000n, -1000000000000000000n, 3000000000000000000n]
    });

    await model.settle();
    let a_rep = await model._getUserGRSAtAddress(a.address)
    console.log("a_rep: ", a_rep)
    let b_rep = await model._getUserGRSAtAddress(b.address)
    console.log("b_rep: ", b_rep)
    let c_rep = await model._getUserGRSAtAddress(c.address)
    console.log("c_rep: ", c_rep)

    expect(await model._getUserGRSAtAddress(a.address)).to.equal(a_rep);
  });
});


describe("Struct functionality", function () {
  it("Tests whether setters and getters work as expected", async function () {
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

    await model._setUserGRSAtAddress(a.address, 20);
    expect(await model._getUserGRSAtAddress(a.address)).to.equal(20);

    await model._setUserGRSAtAddress(a.address, 30);
    expect(await model._getUserGRSAtAddress(a.address)).to.equal(30);
  });
});

describe("User storage user test", function () {
  it("Tests whether setters and getters work as expected when using \"User storage user\"", async function () {
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

    await model._setUserGRSAtAddressStorage(a.address, 20);
    expect(await model._getUserGRSAtAddress(a.address)).to.equal(20);

    await model._setUserGRSAtAddressStorage(a.address, 30);
    expect(await model._getUserGRSAtAddress(a.address)).to.equal(30);
  });
});

describe("Feedback functionality test", function () {
  it("Tests whether feedback updates roundReputation correctly", async function () {
    const [owner, a, b] = await ethers.getSigners();

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

    // Register both users
    await model.connect(a)["register()"]({
      value: ethers.parseEther("1")
    });

    await ethers.provider.send("evm_increaseTime", [86400]);
    await ethers.provider.send("evm_mine", []);

    // A gives positive feedback to B
    await model.connect(a).feedback(b.address, 1);

    // roundReputation is index 3 in getUser return tuple
    expect((await model.getUser(a.address))[5]).to.equal(1);

    await model.connect(a).feedback(c.address, 0);

    expect((await model.getUser(a.address))[5]).to.equal(2);
  });
});