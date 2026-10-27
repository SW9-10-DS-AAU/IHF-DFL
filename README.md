# OpenFL: Decentralized Federated Learning on Public Blockchain Systems

```
 ___ _   _ ____       ____  _____ _     
|_ _| | | |  _ \     |  _ \|  ___| |    
 | || |_| | |_) |____| | | | |_  | |    
 | ||  _  |  __/_____| |_| |  _| | |___ 
|___|_| |_|_|        |____/|_|   |_____|                      
```

# Getting started
## 1. Ganache
- Download Ganache
- Set up a workspace (Not quickstart)
- Set gas limit much higher than default, same with balance
- Set accounts to 8

## 2. Environment Variables
The project contains a .env file located in the .env folder, but supports easy replacement of this environmnent.
The project runs with the .env.ganache .env file by default. If another .env is preferred run the program with the 
``ENV=<env_file_identifier>`` prefix. Providing no ENV prefix and providing ``ENV=ganache`` is therefore equivalent.

In your Environment, you must have the following variables set:
```
RPC_URL="<RPC_URL from ganache or sepolia, including port>"
PRIVATE_KEYS="<Private keys from your accounts colon separated (for non-locally forked blockchain). If you have fork=true (using Ganache), there is no need to set private keys. Then just keep this variable empty>"
```

## 3. Requirements
- Only tested with Python3.10
Create a virtual environment and activate it.
- Run ``python3 -m venv .venv``
- Then Run ``source .venv/bin/activate``

Install requirements:
Only tested with Python3.10
- Run ``pip install -e ".[dev]"``

Build the abi and bytecode files from the smart contracts
``python3 scripts/compile_contracts.py``

Strip notebook outputs before committing (run once per clone):
``nbstripout --install``

Install ROCM(AMD) or CUDA(NVIDIA):
ROCM:
``pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1``
CUDA:
``pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130``


## 4. Running an Experiment

Two modes are availble for running an experiment: sample or experiment.
### Sample
Just want to see a quick demo of the code? Run the sample.py file with:
`python ./experiment/sample.py`

### Experiment
Want to run a systematic experiment with different parameters?

The Experiment folder contains files for running experiments on different datasets.
The presets in experiment_presets.py define the parameter grid of each experiment.

The file experiments.py runs every configuration of one preset on one dataset. Without flags it runs the `test` preset on MNIST:

`python ./experiment/experiments.py`

Choose the preset and dataset with `--preset` and `--dataset` (`mnist` or `cifar.10`):

`python ./experiment/experiments.py --preset cifar_openfl_low_noise --dataset cifar.10`

Each run records the commit it ran on (`git_commit`) and whether tracked files had uncommitted changes (`git_dirty`) in its `.pkl` metadata.
The script refuses to start when tracked files have uncommitted changes, or when git is unavailable, so every run can be tied to a commit.
Untracked files do not count. Pass `--allow-dirty` to run anyway; the runs are then recorded with `git_dirty=True`, or with `git_commit="unknown"` when git is unavailable.

Or if you want to specify a different .env file, run with the ENV prefix as described in the Environment Variables section:
``ENV=ganache python ./experiment/experiments.py``

### Paper figures
Rebuild the paper's Fig. 3-8 from stored runs, from the repo root:

`python -m analysis.paper_figures [RUNS_DIR] [OUT_DIR]`

`RUNS_DIR` defaults to `data/runs` and `OUT_DIR` to `data/output_graphs/paper`, both relative to the repo root. The script writes `fig3.png` to `fig8.png` into `OUT_DIR`.
It expects the four paper presets (480 runs).


## 5. Solidity testing
- All forge commands must be run in the foundry directory.
- Download Foundry in a Unix-like shell (WSL or Linux): 
  - `curl -L https://foundry.paradigm.xyz | bash && source ~/.bashrc`
  - `foundryup`
  - `cd foundry && forge soldeer install`
  
- To Test:
  - `cd foundry`
  - `forge build` 
  - `forge test`

# 6. Running Tests

Tests are run with pytest from the repo root:
```
pytest
```

By default, tests marked as `slow` are skipped. These are tests that load full datasets (e.g. CIFAR-10)
and can take significant time. To include them:
```
pytest -m slow
```

To run only the slow tests:
```
pytest -m slow --no-header -q
```

The slow acceptance test for the paper figures (`tests/integration/test_paper_figures_acceptance.py`) checks the paper's numbers against its 480 original runs, kept in `data/runs_paper_v1` (gitignored).
They are kept apart from `data/runs`, where new runs land, because `analysis.paper_figures` loads every run under the folder it is given.
Point it at another folder with the `IHF_RUNS_DIR` environment variable; the test is skipped when the folder does not exist:
```
IHF_RUNS_DIR=/path/to/runs pytest -m slow tests/integration/test_paper_figures_acceptance.py
```

Solidity tests (forge) run automatically as part of the pytest suite on Linux/WSL.
On Windows they are skipped — run them manually in WSL:
```
cd foundry && forge test
```

# 7. Test Coverage

To get test coverage for python code, run the following command: \
`pytest --cov tests/`

To get test coverage for solidity code, run the following command: \
`forge coverage`

(Optional) Output to a file
`forge coverage --report lcov`


# 8. Solidity Compiler (Arm)
If running on arm cpu, you need to download a recompiled solidity (solc) compiler for arm 
and place it in the directory ```~/.local/bin/solc```
The files can be found here: https://github.com/0xidm/solc-bin