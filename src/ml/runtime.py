import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.paths import repo_root
from utils.colors import green, red, yellow

logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
PIN_MEMORY = USE_CUDA
NON_BLOCKING = USE_CUDA
NUM_WORKERS = min(4, os.cpu_count() // 2) if torch.cuda.is_available() else 0
PERSISTENT_WORKERS = USE_CUDA and NUM_WORKERS > 0
AMP = USE_CUDA  # Optional: mixed precision on CUDA
DATASET_ROOT = repo_root(Path(__file__)) / "data" / "datasets"
# cuDNN autotune for fixed-size inputs (both MNIST 28x28 and CIFAR-10 32x32)
torch._dynamo.config.cache_size_limit = 512
torch.backends.cudnn.benchmark = USE_CUDA

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


def model_to_device(net: nn.Module) -> nn.Module:
    # Move model once; keep it on the chosen device
    return net.to(DEVICE, non_blocking=NON_BLOCKING)


def cuda_safe_dataloader(ds, batch_size, shuffle=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )


def print_training_mode(num_gpus: int, num_processes: int):
    """Prints a clean status message describing how training will run."""
    if num_gpus >= 2:
        print(green(f"Detected {num_gpus} GPU(s) → Parallel multi-GPU training"))

    elif num_gpus == 1:
        if num_processes > 1:
            print(yellow(
                f"Detected 1 GPU → Parallel training on one GPU (shared across {num_processes} workers)"
            ))
        else:
            print(green("Detected 1 GPU → Sequential GPU training"))

    else:  # CPU-only
        if num_processes > 1:
            print(yellow(
                f"Detected 0 GPU(s) → Parallel CPU training ({num_processes} workers)"
            ))
        else:
            print(red("Detected 0 GPU(s) → Sequential CPU mode"))
