from types import SimpleNamespace

import pytest

import experiment.run_grid as run_grid
import ml.data as data
from ml.pytorch_model import PytorchModel


def test_load_data_rejects_unknown_dataset_name():
    pm = SimpleNamespace(DATASET="cfar.10", NUMBER_OF_CONTRIBUTORS=3, BATCHSIZE=32, run_id=0, DATA=None)

    with pytest.raises(ValueError, match="cfar.10"):
        data.load_data(pm)


def test_pytorch_model_rejects_unknown_dataset_name():
    with pytest.raises(ValueError, match="cfar.10"):
        PytorchModel("cfar.10", 2, 0, 0, epochs=1, batchsize=32, default_collateral=1, max_collateral=2)


@pytest.mark.parametrize("name", run_grid.DATASETS)
def test_every_command_line_dataset_name_is_accepted(name):
    data.is_cifar10(name)
