import pytest
import torch
from types import SimpleNamespace
from torch.utils.data import Subset

import ml.data as data

# CIFAR-10 is downloaded on first use and is not in data/datasets/ on every machine.
pytestmark = pytest.mark.slow

# Three contributors leave 2 of the 50,000 images unused, so the training set is
# split twice (unused images, then partitions) before the train/val split.
NUMBER_OF_CONTRIBUTORS = 3

# Recorded from load_data before the validation split stopped using augmented images:
# per partition, the number of validation images, the first five indices into the
# CIFAR-10 training set and the sum of all of them.
VALIDATION_INDICES = {
    0: [
        (1666, [9552, 16201, 19217, 22494, 29406], 41650477),
        (1666, [44542, 1784, 36831, 41379, 28273], 41537873),
        (1666, [38910, 35411, 41665, 40142, 47238], 42350381),
    ],
    1: [
        (1666, [48700, 28582, 19224, 14710, 19509], 41944685),
        (1666, [34331, 29055, 29626, 40386, 17127], 41643947),
        (1666, [45229, 15068, 46371, 28658, 21867], 41636250),
    ],
}


def load_cifar10(run_id):
    pm = SimpleNamespace(
        DATASET="cifar-10",
        NUMBER_OF_CONTRIBUTORS=NUMBER_OF_CONTRIBUTORS,
        BATCHSIZE=32,
        run_id=run_id,
        DATA=None,
    )
    return data.load_data(pm)


def cifar10_indices(subset):
    indices = list(range(len(subset)))
    dataset = subset
    while isinstance(dataset, Subset):
        indices = [dataset.indices[i] for i in indices]
        dataset = dataset.dataset
    return indices


@pytest.fixture(scope="module")
def cifar10_run_0():
    return load_cifar10(run_id=0)


def test_cifar10_validation_sample_is_identical_on_every_load(cifar10_run_0):
    _, valloaders, _ = cifar10_run_0
    validation_set = valloaders[0].dataset
    torch.manual_seed(0)

    first_image, first_label = validation_set[0]
    second_image, second_label = validation_set[0]

    assert torch.equal(first_image, second_image)
    assert first_label == second_label


@pytest.mark.parametrize("run_id", [0, 1])
def test_cifar10_validation_indices_are_unchanged_for_run_id(run_id):
    _, valloaders, _ = load_cifar10(run_id)

    actual = []
    for loader in valloaders:
        indices = cifar10_indices(loader.dataset)
        actual.append((len(indices), indices[:5], sum(indices)))

    assert actual == VALIDATION_INDICES[run_id]


def test_cifar10_training_sample_is_augmented(cifar10_run_0):
    trainloaders, _, _ = cifar10_run_0
    training_set = trainloaders[0].dataset
    torch.manual_seed(0)

    images = [training_set[0][0] for _ in range(5)]

    assert any(not torch.equal(images[0], image) for image in images[1:])
