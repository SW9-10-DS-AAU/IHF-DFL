import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from ml.runtime import PIN_MEMORY, NUM_WORKERS, PERSISTENT_WORKERS, DATASET_ROOT
import data.seeds as seeds
from collections import Counter




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

    # dist = pm.data_distribution or "random_split"

    # if dist.startswith("random_split"):
    datasets = random_split(trainset, lengths, generator=gen)

    # elif dist.startswith("stratified_split"):
    #     datasets = stratified_split(trainset, pm.NUMBER_OF_CONTRIBUTORS, generator=gen)
    #
    # elif dist.startswith("dirichlet_split"):
    #     datasets = dirichlet_split(trainset, pm.NUMBER_OF_CONTRIBUTORS, alpha=pm.dirichlet_alpha, generator=gen)

    # else:
    #     raise ValueError(f"Data distribution {pm.data_distribution} not recognized")

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []

    for ds in datasets:
        len_val = len(ds) // 10
        len_train = len(ds) - len_val
        tv_lengths = [len_train, len_val]

        # if "stratified" in str(pm.data_distribution):
        #     ds_train, ds_val = stratified_split(ds, tv_lengths, generator=gen)
        # else:
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


# def dirichlet_split(dataset, num_clients, alpha=0.5, generator=None):
#     """
#     Dirichlet split that ensures:
#     - All clients have exactly len(dataset)//num_clients samples
#     - Non-IID distribution controlled by alpha
#     - Optional reproducibility with generator
#     """
#     # Handle Subset
#     if isinstance(dataset, Subset):
#         actual_dataset = dataset.dataset
#         indices = dataset.indices
#     else:
#         actual_dataset = dataset
#         indices = list(range(len(dataset)))
#
#     # Get labels
#     if hasattr(actual_dataset, "targets"):
#         targets = torch.as_tensor(actual_dataset.targets).clone().detach()[indices]
#     elif hasattr(actual_dataset, "labels"):
#         targets = torch.as_tensor(actual_dataset.labels).clone().detach()[indices]
#     else:
#         raise AttributeError("Dataset must have targets or labels attribute")
#
#     num_classes = len(torch.unique(targets))
#     class_indices = {c: (targets == c).nonzero(as_tuple=True)[0].tolist() for c in range(num_classes)}
#
#     # Shuffle class indices
#     seed = 42 if generator is not None else None
#     rng = np.random.default_rng(seed)
#     for c in class_indices:
#         rng.shuffle(class_indices[c])
#
#     splits = [[] for _ in range(num_clients)]
#     total_per_client = len(dataset) // num_clients
#     client_counts = [0] * num_clients  # track total assigned per client
#
#     # Allocate samples per class
#     for c in range(num_classes):
#         cls_idx = class_indices[c]
#         n_cls = len(cls_idx)
#
#         # Sample Dirichlet proportions
#         proportions = rng.dirichlet([alpha] * num_clients)
#         counts = (proportions * n_cls).astype(int)
#
#         # Fix rounding to match class total
#         diff = n_cls - counts.sum()
#         if diff > 0:
#             topk = np.argsort(proportions)[-diff:]
#             counts[topk] += 1
#
#         # Assign to clients, but do not exceed total_per_client
#         start = 0
#         for i in range(num_clients):
#             # adjust count if client is almost full
#             remaining_space = total_per_client - client_counts[i]
#             assign_count = min(counts[i], remaining_space)
#             end = start + assign_count
#             splits[i].extend(cls_idx[start:end])
#             client_counts[i] += assign_count
#             start += assign_count
#
#     # At this point, all clients should have exactly total_per_client
#     # If any client still has less (due to rounding), fill from leftover pool
#     assigned = set(idx for client in splits for idx in client)
#     all_indices = set(idx for cls in class_indices.values() for idx in cls)
#     remaining = list(all_indices - assigned)
#     rng.shuffle(remaining)
#
#     for i in range(num_clients):
#         current_len = len(splits[i])
#         if current_len < total_per_client:
#             needed = total_per_client - current_len
#             splits[i].extend(remaining[:needed])
#             remaining = remaining[needed:]
#
#     return [Subset(actual_dataset, split) for split in splits]



# def stratified_split(dataset, lengths, generator=None):
#     from torch.utils.data import Subset
#     import torch
#
#     # Normalize input
#     if isinstance(lengths, int):
#         num_splits = lengths
#         lengths = [1] * num_splits  # equal split
#     else:
#         num_splits = len(lengths)
#
#     total_length = sum(lengths)
#
#     # Handle subset
#     if isinstance(dataset, Subset):
#         actual_dataset = dataset.dataset
#         base_indices = list(dataset.indices)
#     else:
#         actual_dataset = dataset
#         base_indices = list(range(len(dataset)))
#
#     # Get targets
#     if hasattr(actual_dataset, "targets"):
#         targets = actual_dataset.targets
#     elif hasattr(actual_dataset, "labels"):
#         targets = actual_dataset.labels
#     else:
#         raise AttributeError("Dataset must have targets or labels")
#
#     targets = torch.as_tensor(targets, dtype=torch.long)[base_indices]
#
#     # Group by class
#     classes = torch.unique(targets)
#     class_indices = {
#         int(c): (targets == c).nonzero(as_tuple=True)[0].tolist()
#         for c in classes
#     }
#
#     # Shuffle per class
#     for c in class_indices:
#         perm = torch.randperm(len(class_indices[c]), generator=generator).tolist()
#         class_indices[c] = [class_indices[c][i] for i in perm]
#
#     splits = [[] for _ in range(num_splits)]
#
#     # Split per class proportionally
#     for c in class_indices:
#         cls_idx = class_indices[c]
#         cls_len = len(cls_idx)
#
#         proportions = [l / total_length for l in lengths]
#         raw = [p * cls_len for p in proportions]
#         cls_lengths = [int(x) for x in raw]
#
#         # distribute remainder
#         remainder = cls_len - sum(cls_lengths)
#         for i in range(remainder):
#             cls_lengths[i % num_splits] += 1
#
#         start = 0
#         for i in range(num_splits):
#             selected_local = cls_idx[start:start + cls_lengths[i]]
#             selected_global = [base_indices[idx] for idx in selected_local]
#             splits[i].extend(selected_global)
#             start += cls_lengths[i]
#
#     # Final shuffle
#     for i in range(num_splits):
#         perm = torch.randperm(len(splits[i]), generator=generator).tolist()
#         splits[i] = [splits[i][j] for j in perm]
#
#     return [Subset(actual_dataset, split) for split in splits]


# def get_label_distribution(loader):
#     counter = Counter()
#     for _, labels in loader:
#         counter.update(labels.tolist())
#     return dict(counter)




# def print_label_distribution(loader, name="Dataset"):
#     """
#     Prints the distribution of labels in a DataLoader.
#     """
#     label_counts = Counter()
#     for _, labels in loader:
#         label_counts.update(labels.numpy())  # Convert tensor to numpy for Counter
#
#     total = sum(label_counts.values())
#     print(f"{name} | Total samples: {total}")
#     for label in range(10):
#         count = label_counts[label]
#         print(f"  Label {label}: {count} ({count/total*100:.2f}%)")
#     print("-" * 50)


# def print_client_data_distributions(pm):
#     """Prints the label distribution for every client's train/val sets and the global test set."""
#     if pm.DATA is None:
#         load_data(pm, pm.NUMBER_OF_CONTRIBUTORS)
#
#     trainloaders, valloaders, testloader = pm.DATA
#
#     print("\n--- CLIENT DATA DISTRIBUTIONS ---")
#     print("\nUSING DATA DISTRIBUTION:", pm.data_distribution)
#     if "dirichlet" in pm.data_distribution:
#         print(f"\nDirichlet alpha: {pm.dirichlet_alpha}")
#
#     for i, (train_loader, val_loader) in enumerate(zip(trainloaders, valloaders)):
#         print(f"\nClient {i} Training Set:")
#         print_label_distribution(train_loader, name=f"Client {i} Train")
#
#         print(f"Client {i} Validation Set:")
#         print_label_distribution(val_loader, name=f"Client {i} Val")
#
#     print("\nGlobal Test Set:")
#     print_label_distribution(testloader, name="Test Set")



# def get_client_data_distribution(pm):
#     if pm.DATA is None:
#         load_data(pm, pm.NUMBER_OF_CONTRIBUTORS)
#
#     trainloaders, valloaders, testloader = pm.DATA
#
#     result = {
#         "clients": [],
#         "test": {}
#     }
#
#     for train_loader, val_loader in zip(trainloaders, valloaders):
#         train_dataset = train_loader.dataset
#         val_dataset = val_loader.dataset
#
#         client_info = {
#             "train": {
#                 "size": len(train_dataset),
#                 "label_counts": get_label_distribution(train_loader),
#                 "indices": train_dataset.indices if hasattr(train_dataset, "indices") else None
#             },
#             "val": {
#                 "size": len(val_dataset),
#                 "label_counts": get_label_distribution(val_loader),
#                 "indices": val_dataset.indices if hasattr(val_dataset, "indices") else None
#             }
#         }
#         result["clients"].append(client_info)
#
#     test_dataset = testloader.dataset
#     result["test"] = {
#         "size": len(test_dataset),
#         "label_counts": get_label_distribution(testloader),
#         "indices": test_dataset.indices if hasattr(test_dataset, "indices") else None
#     }
#
#     return result