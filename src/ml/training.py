import sys
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
) -> None:
    # Compile ONCE per process (not per batch)
    if device.type == "cuda":
        try:
            net = torch.compile(net, mode="reduce-overhead")
        except Exception:
            pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


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


def train_user_proc(user_id, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory,
                    shuffle):
    # Multi-GPU Support
    # Select device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")

    # Recreate model based on dataset
    if dataset == "mnist":
        model = Net_MNIST()
    else:
        model = Net_CIFAR()

    model.load_state_dict(model_state)
    model.to(device)

    # Rebuild dataloaders inside the process
    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=shuffle,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=shuffle,
                            pin_memory=pin_memory)

    train(model, train_loader, epochs, device)  # Line 285 in original code
    val_loss, val_acc = test(model, val_loader, device)  # Line 286 in original code

    # del: Mark for GC
    del train_loader
    del val_loader

    print(f"[{device_label(device, device_id)}] User {user_id} done | Acc: {val_acc:.3f}, Loss: {val_loss:.3f}")

    # Move state dict to CPU before returning so the parent receives plain CPU
    # tensors instead of CUDA IPC handles, avoiding the "shared CUDA tensors
    # released after producer exit" warning.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return user_id, cpu_state, val_loss, val_acc
