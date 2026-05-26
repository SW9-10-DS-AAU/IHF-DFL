import sys
import multiprocessing as mp
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.runtime import NON_BLOCKING
from ml.cnn_models import Net_CIFAR, Net_MNIST
from typing import Tuple


def device_label(device: torch.device, device_id: int = 0) -> str:
    if device.type == "cuda":
        return f"GPU {device_id}"
    else:
        return "CPU"


def train(
        net,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,
) -> None: # tuple[float, float, float, int]
    # Compile ONCE per process (not per batch).
    # NOTE: mode="reduce-overhead" enables CUDA Graphs (cudagraph_trees), which
    # preallocate static memory pools per graph. With multiple worker processes
    # sharing a single GPU this caused CUBLAS_STATUS_ALLOC_FAILED. Default mode
    # keeps the inductor speedup without the static allocation.
    compile_start = time.perf_counter()
    if device.type == "cuda":
        try:
            net = torch.compile(net)
        except Exception:
            pass
    # compile_seconds = time.perf_counter() - compile_start

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    net.train()

    # data_wait_seconds = 0.0
    # compute_seconds = 0.0
    # batches = 0

    for _ in range(epochs):
        # batch_wait_start = time.perf_counter()
        for images, labels in trainloader:
            # if device.type == "cuda":
            #     torch.cuda.synchronize(device)
            # data_wait_seconds += time.perf_counter() - batch_wait_start

            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            optimizer.zero_grad(set_to_none=True)

            # compute_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if device.type == "cuda":
            #     torch.cuda.synchronize(device)
            # compute_seconds += time.perf_counter() - compute_start
            # batches += 1
            # batch_wait_start = time.perf_counter()

    # return compile_seconds, data_wait_seconds, compute_seconds, batches


def test(net, testloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on test set: forward pass only (no gradients), with optional AMP on CUDA
    Accumulate total cross-entropy loss and count correct predictions for accuracy
    Returns (total_loss, accuracy) over the entire test dataset
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()

    correct = 0
    total = 0
    loss = 0.0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    loss = min(sys.float_info.max, loss)

    return loss, accuracy


def _loader_worker_options(use_cuda: bool) -> tuple[int, bool]:
    # multiprocessing.Pool workers are daemonic. A DataLoader with workers
    # would try to create child processes from that pool worker.
    if not use_cuda or mp.current_process().daemon:
        return 0, False
    num_workers = min(4, (os.cpu_count() or 1) // 2)
    return num_workers, num_workers > 0


def train_user_proc(user_id, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory):
    # Multi-GPU Support
    # Select device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")

    # Recreate model based on dataset
    if dataset == "mnist":
        model = Net_MNIST()
        num_workers, persistent_workers = 0, False
    else: # CIFAR
        model = Net_CIFAR()
        num_workers, persistent_workers = _loader_worker_options(use_cuda)

    model.load_state_dict(model_state)
    model.to(device)

    # Rebuild dataloaders inside the process
    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True,
                            pin_memory=pin_memory, num_workers=num_workers,
                            persistent_workers=persistent_workers, prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=False,
                            pin_memory=pin_memory, num_workers=0,
                            persistent_workers=False)

    # train_start = time.perf_counter()
    # train_compile, train_data_wait, train_compute, train_batches = train(model, train_loader, epochs, device)  # Line 285 in original code
    train(model, train_loader, epochs, device)  # Line 285 in original code
    # train_seconds = time.perf_counter() - train_start
    # val_start = time.perf_counter()
    val_loss, val_acc = test(model, val_loader, device)  # Line 286 in original code
    # val_seconds = time.perf_counter() - val_start

    # del: Mark for GC
    del train_loader
    del val_loader

    # device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    # print(
    #     f"[{device_label(device, device_id)}] User {user_id} done | "
    #     f"device={device_name} daemon={mp.current_process().daemon} "
    #     f"loaders={num_workers} batches={train_batches} | "
    #     f"train={train_seconds:.2f}s compile={train_compile:.2f}s data_wait={train_data_wait:.2f}s "
    #     f"compute={train_compute:.2f}s val={val_seconds:.2f}s | "
    #     f"Acc: {val_acc:.3f}, Loss: {val_loss:.3f}"
    # )
    print(f"[{device_label(device, device_id)}] User {user_id} done | Acc: {val_acc:.3f}, Loss: {val_loss:.3f}")


    # Move state dict to CPU before returning so the parent receives plain CPU
    # tensors instead of CUDA IPC handles, avoiding the "shared CUDA tensors
    # released after producer exit" warning.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return user_id, cpu_state, val_loss, val_acc
