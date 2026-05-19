import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from ml.runtime import PIN_MEMORY, NUM_WORKERS, PERSISTENT_WORKERS, DATASET_ROOT
import data.seeds as seeds


def load_data(pm, _print=False):
    if pm.DATA:
        return pm.DATA

    if pm.DATASET == "cifar-10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10(DATASET_ROOT / "CIFAR", train=True, download=True,
                           transform=transform)  # Since CIFAR does not makes its own subfolder, we make it.
        testset = CIFAR10(DATASET_ROOT / "CIFAR", train=False, download=True, transform=transform_test)
    else:
        trainset = MNIST(DATASET_ROOT, train=True, download=True, transform=transforms.ToTensor())
        testset = MNIST(DATASET_ROOT, train=False, download=True, transform=transforms.ToTensor())

    if _print:
        print("Data Loaded:")
        print("Nr. of images for training: {:,.0f}".format(len(trainset)))
        print("Nr. of images for testing:  {:,.0f}\n".format(len(testset)))

    # Split training set into partitions to simulate the individual dataset
    partition_size = len(trainset) // pm.NUMBER_OF_CONTRIBUTORS
    lengths = [partition_size] * pm.NUMBER_OF_CONTRIBUTORS
    if pm.run_id == 0:
        gen = torch.Generator().manual_seed(42)
        print("DATA DISTRIBUTION: Using fixed seed 42 for sample (single run) for reproducibility")
    else:
        gen = torch.Generator().manual_seed(seeds.seeds[str(pm.run_id)])
        print(f"DATA DISTRIBUTION: Using seed {seeds.seeds[str(pm.run_id)]} for run_id {pm.run_id} for reproducibility")


    images_needed = partition_size * pm.NUMBER_OF_CONTRIBUTORS
    if images_needed < len(trainset):
        trainset, _ = random_split(trainset, [images_needed, len(trainset) - images_needed], generator=gen)

    datasets = random_split(trainset, lengths, generator=gen)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []

    for ds in datasets:
        len_val = len(ds) // 10
        len_train = len(ds) - len_val
        tv_lengths = [len_train, len_val]

        ds_train, ds_val = random_split(ds, tv_lengths, generator=gen)

        trainloaders.append(DataLoader(
            ds_train,
            batch_size=pm.BATCHSIZE,
            shuffle=True,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
        ))
        valloaders.append(DataLoader(
            ds_val,
            batch_size=pm.BATCHSIZE,
            shuffle=False,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
        ))
    testloader = DataLoader(
        testset,
        batch_size=pm.BATCHSIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )
    pm.DATA = (trainloaders, valloaders, testloader)
    return trainloaders, valloaders, testloader