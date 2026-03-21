# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenFL 2.0 is a federated learning research platform that integrates PyTorch-based distributed ML training with Ethereum smart contracts. It simulates Byzantine-resilient federated learning with on-chain reputation and incentive mechanisms.

## Commands

### Setup
```bash
pip install -e ".[dev]"
python3 scripts/compile_contracts.py   # Build ABI + bytecode from Solidity contracts

# GPU (choose one):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1  # AMD
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130    # NVIDIA
```

### Running Experiments
```bash
ENV=ganache python ./experiment/experiment_runner.py
```

### Python Tests
```bash
pytest --cov=openfl tests/
```

### Solidity Tests (requires Foundry in WSL/Linux)
```bash
forge build
forge test
```

## Architecture

### Layers

**Experiment Layer** (`experiment/`)
- `experiment_configuration.py` — central config: participant counts, reward/collateral/punishment params, training hyperparams, contribution score strategy
- `experiments.py` — dataset-specific configs (CIFAR-10, MNIST)
- `experiment_runner.py` — orchestrates a full experiment end-to-end

**ML Layer** (`src/openfl/ml/pytorch_model.py`)
- `PytorchModel` — orchestrates federated learning simulation; manages participants, runs training rounds, evaluates contributions
- `Participant` — represents one FL participant; tracks collateral, reputation, attitude (good/bad/freerider/inactive), and submitted model hashes

**Contract Interaction Layer** (`src/openfl/contracts/`)
- `fl_manager.py` (`FLManager`) — deploys and manages `OpenFLManager`/`OpenFLModel` contracts; bridges Python ↔ blockchain
- `fl_challenge.py` (`FLChallenge`) — drives the FL round lifecycle: user registration, hashed weight submission, feedback exchange, reward/punishment dispatch; implements contribution scoring strategies

**Contribution Scoring Strategies** (selected via `ExperimentConfiguration`):
- `dotproduct` — matrix multiplication of weight vectors
- `naive` — accuracy-based
- `accuracy_loss` — combined accuracy + loss
- `accuracy_only` / `loss_only` — single-metric variants

**Blockchain / Web3 Layer** (`src/openfl/api/`, `contracts/`)
- `connection_helper.py` — RPC connection, ABI/bytecode loading, account init
- `OpenFLManager.sol` — deploys new FL model contracts per user
- `OpenFLModel.sol` — on-chain reputation system: registration, hashed weight submission, voting, punishments, rewards

### Environment Configuration

Environment files live in `.env/`. The active env is selected via `ENV=<identifier>` prefix (defaults to `ganache`).

Required variables:
- `RPC_URL` — blockchain RPC endpoint including port
- `PRIVATE_KEYS` — colon-separated private keys (only needed when `fork=false`, i.e., Sepolia; leave empty for Ganache fork mode)

### Ganache Setup

Ganache requires a workspace (not quickstart) with: gas limit set significantly above default, high balance, and exactly **8 accounts**.

### Python Version

Only Python 3.10 is supported/tested.
