import random
import pytest
import torch
import ml.attacks as attacks
import ml.visualization as visualization
import ml.training as training


def test_manipulate_adds_noise():
    model = torch.nn.Linear(2, 2)
    original = {k: v.clone() for k, v in model.state_dict().items()}

    updated = attacks.manipulate(model, scale=0.0)

    # With scale 0.0, floating tensors should be unchanged
    for key, tensor in updated.items():
        assert torch.equal(tensor, original[key])


def test_add_noise_modifies_target_tensor():
    random.seed(0)
    model = torch.nn.Linear(3, 1)
    state_dict = attacks.add_noise(model, offset_from_end=1)

    items = list(model.state_dict().items())
    target_idx = max(0, len(items) - 1)
    target_key, target_tensor = items[target_idx]

    new_tensor = state_dict[target_key]
    diff = (new_tensor - target_tensor).abs().max().item()
    assert diff == pytest.approx(1e-05, rel=0, abs=2e-06)


def test_get_color_handles_known_and_unknown_indices():
    assert visualization.get_color(0, "bad") == visualization.bad_c
    assert visualization.get_color(0, "freerider") == visualization.free_c
    assert visualization.get_color(100, "good") is None


def test_device_label_formats_output():
    assert training.device_label(torch.device("cpu")) == "CPU"
    assert training.device_label(torch.device("cuda"), 2) == "GPU 2"