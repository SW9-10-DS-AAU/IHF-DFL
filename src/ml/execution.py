from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import os

import psutil
import torch
from utils.colors import yellow


@dataclass(frozen=True)
class TrainingPlan:
    # reason values: debug, multi_gpu, cpu_parallel, cpu_single_worker, single_gpu
    reason: str
    parallel: bool
    num_gpus: int
    workers: int


def pin_cuda_worker(device_id: int):
    torch.cuda.set_device(device_id)


def resolve_cpu_pool_size(participants: int):
    logical_cores = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    available_gb = psutil.virtual_memory().available / (1024 ** 3)

    core_cap = max(1, physical_cores - 1)
    ram_cap = max(1, int((available_gb - 2) // 2))
    resolved = max(1, min(participants, core_cap, ram_cap))
    decision = {
        "participants": participants,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "available_gb": available_gb,
        "core_cap": core_cap,
        "ram_cap": ram_cap,
        "resolved": resolved,
    }
    return resolved, decision


def resolve_training_plan(participants: int, debugging: bool):
    if debugging:
        return TrainingPlan("debug", False, torch.cuda.device_count(), 1), None

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        return TrainingPlan("multi_gpu", True, num_gpus, num_gpus), None

    if num_gpus == 0:
        workers, decision = resolve_cpu_pool_size(participants)
        if workers > 1:
            return TrainingPlan("cpu_parallel", True, num_gpus, workers), decision
        return TrainingPlan("cpu_single_worker", False, num_gpus, 1), decision

    return TrainingPlan("single_gpu", False, num_gpus, 1), None


def create_cpu_pool(ctx, workers: int):
    return ctx.Pool(processes=workers)


def create_gpu_pools(ctx, num_gpus: int):
    return [
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=pin_cuda_worker,
            initargs=(device_id,),
        )
        for device_id in range(num_gpus)
    ]


def close_pools(cpu_pool, gpu_pools):
    if cpu_pool is not None:
        cpu_pool.close()
        cpu_pool.join()
    for pool in gpu_pools:
        pool.shutdown(wait=True)


def print_system_capabilities(num_gpus: int):
    logical_cores = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    total_gb = memory.total / (1024 ** 3)

    print(
        yellow(
            "Hardware: "
            f"GPUs={num_gpus}; "
            f"CPU cores={physical_cores} physical/{logical_cores} logical; "
            f"RAM={available_gb:.2f}/{total_gb:.2f} GiB available"
        )
    )


def print_cpu_pool_decision(decision):
    if not decision:
        return

    print(
        yellow(
            "CPU worker sizing: "
            f"participants={decision['participants']}, "
            f"core_cap={decision['core_cap']}, "
            f"ram_cap={decision['ram_cap']} "
            f"({decision['available_gb']:.2f} GiB available) -> "
            f"workers={decision['resolved']}"
        )
    )
