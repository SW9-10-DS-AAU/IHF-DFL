import fs from "fs";

const artifact = JSON.parse(
  fs.readFileSync(
    "artifacts/contracts/OpenFLModel.sol/OpenFLModel.json"
  )
);

const bytecode = artifact.deployedBytecode;
const size = (bytecode.length - 2) / 2;

console.log("Deployed size:", size, "bytes");