import pytest
import torch
from collections import OrderedDict
from unittest.mock import MagicMock, patch
from src.ml.attacks import manipulate, add_noise, _freerider_submit_with_noise, update_users_attitude


def mock_pytorch_model(prev_state_dict, current_state_dict, noise_scale=0.01):
    pm = MagicMock()
    pm.previous_global_model = prev_state_dict
    pm.global_model.state_dict.return_value = current_state_dict
    pm.freerider_noise_scale = noise_scale
    return pm


def make_model(state_dict):
    model = MagicMock()
    model.state_dict.return_value = state_dict
    return model

def make_attitude_user(attitude, future_attitude, switch_round):
    user = MagicMock()
    user.address = "a" * 20
    user.attitude = attitude
    user.futureAttitude = future_attitude
    user.attitudeSwitch = switch_round
    return user


class TestManipulate:
    def test_float_noise_within_scale_bounds(self):
        # Noise er inden for [-scale, scale] for float-tensorer
        scale = 0.5
        original = torch.tensor([0.0, 0.0, 0.0])
        result = manipulate(make_model(OrderedDict([("w", original)])), scale=scale)
        diff = result["w"] - original
        assert (diff >= -scale).all() and (diff <= scale).all()

    def test_scale_zero_leaves_float_unchanged(self):
        # uniform(-0, 0) = 0, dermed ingen ændring
        original = torch.tensor([1.0, 2.0])
        result = manipulate(make_model(OrderedDict([("w", original)])), scale=0.0)
        assert torch.equal(result["w"], original)

    def test_int_tensor_unchanged(self):
        original = torch.tensor([10, 20])
        result = manipulate(make_model(OrderedDict([("idx", original)])), scale=1.0)
        assert torch.equal(result["idx"], original)

    def test_original_not_modified(self):
        original = torch.tensor([1.0, 2.0])
        snapshot = original.clone()
        manipulate(make_model(OrderedDict([("w", original)])), scale=1.0)
        assert torch.equal(original, snapshot)

    def test_returns_ordered_dict(self):
        result = manipulate(make_model(OrderedDict([("w", torch.tensor([1.0]))])), scale=1.0)
        assert isinstance(result, OrderedDict)


class TestAddNoise:
    def test_only_target_tensor_modified(self):
        # 3 items, offset_from_end=1 og target_idx=2, dermed kun "c" ændres
        sd = OrderedDict([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0])), ("c", torch.tensor([3.0]))])
        result = add_noise(make_model(sd), offset_from_end=1)
        assert torch.equal(result["a"], sd["a"])
        assert torch.equal(result["b"], sd["b"])
        assert not torch.equal(result["c"], sd["c"])

    def test_large_offset_to_first_tensor(self):
        # max(0, 2-10) = 0, derfor target_idx=0, og "a" ændres, "b" uændret
        sd = OrderedDict([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))])
        result = add_noise(make_model(sd), offset_from_end=10)
        assert not torch.equal(result["a"], sd["a"])
        assert torch.equal(result["b"], sd["b"])

    def test_non_float_at_target_not_modified(self):
        # target_idx=1 er int-tensor, hvilket betyder betingelsen is_floating_point() er falsk, dermed ingen noise
        sd = OrderedDict([("a", torch.tensor([1.0])), ("b", torch.tensor([10]))])
        result = add_noise(make_model(sd), offset_from_end=1)
        assert torch.equal(result["b"], sd["b"])

    def test_original_not_modified(self):
        sd = OrderedDict([("w", torch.tensor([1.0, 2.0]))])
        snapshot = sd["w"].clone()
        add_noise(make_model(sd), offset_from_end=1)
        assert torch.equal(sd["w"], snapshot)

    def test_returns_ordered_dict(self):
        sd = OrderedDict([("w", torch.tensor([1.0]))])
        result = add_noise(make_model(sd))
        assert isinstance(result, OrderedDict)


class TestFreeriderSubmitWithNoise:
    def test_raises_for_negative_scale(self):
        pm = MagicMock()
        pm.freerider_noise_scale = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            _freerider_submit_with_noise(pm, MagicMock())

    def test_scale_zero_returns_clone_with_same_values(self):
        sd = OrderedDict([("w", torch.tensor([1.0, 2.0]))])
        user = MagicMock()
        user.model.state_dict.return_value = sd
        user.address = "a" * 20
        pm = MagicMock()
        pm.freerider_noise_scale = 0
        result = _freerider_submit_with_noise(pm, user)
        assert torch.equal(result["w"], sd["w"])

    def test_scale_zero_does_not_modify_original(self):
        sd = OrderedDict([("w", torch.tensor([1.0, 2.0]))])
        snapshot = sd["w"].clone()
        user = MagicMock()
        user.model.state_dict.return_value = sd
        user.address = "a" * 20
        pm = MagicMock()
        pm.freerider_noise_scale = 0
        _freerider_submit_with_noise(pm, user)
        assert torch.equal(sd["w"], snapshot)

    @patch("src.ml.attacks.manipulate")
    def test_positive_scale_calls_manipulate(self, mock_manipulate):
        user = MagicMock()
        user.address = "a" * 20
        pm = MagicMock()
        pm.freerider_noise_scale = 0.1
        _freerider_submit_with_noise(pm, user)
        mock_manipulate.assert_called_once_with(user.model, scale=0.1)

    def test_scale_zero_returns_ordered_dict(self):
        sd = OrderedDict([("w", torch.tensor([1.0]))])
        user = MagicMock()
        user.model.state_dict.return_value = sd
        user.address = "a" * 20
        pm = MagicMock()
        pm.freerider_noise_scale = 0
        result = _freerider_submit_with_noise(pm, user)
        assert isinstance(result, OrderedDict)


class TestUpdateUsersAttitude:
    @patch("src.ml.attacks.get_color")
    def test_switches_attitude_when_round_matches(self, _):
        user = make_attitude_user("good", "bad", switch_round=3)
        pm = MagicMock()
        pm.round = 3
        pm.participants = [user]
        update_users_attitude(pm)
        assert user.attitude == "bad"

    def test_does_not_switch_when_round_does_not_match(self):
        user = make_attitude_user("good", "bad", switch_round=5)
        pm = MagicMock()
        pm.round = 3
        pm.participants = [user]
        update_users_attitude(pm)
        assert user.attitude == "good"

    def test_does_not_switch_when_already_target_attitude(self):
        # attitude == futureAttitude, dermed betingelsen attitude != futureAttitude er falsk
        user = make_attitude_user("bad", "bad", switch_round=3)
        pm = MagicMock()
        pm.round = 3
        pm.participants = [user]
        update_users_attitude(pm)
        assert user.attitude == "bad"

    @patch("src.ml.attacks.get_color")
    def test_only_matching_user_switches(self, _):
        user1 = make_attitude_user("good", "bad", switch_round=3)
        user2 = make_attitude_user("good", "bad", switch_round=5)
        pm = MagicMock()
        pm.round = 3
        pm.participants = [user1, user2]
        update_users_attitude(pm)
        assert user1.attitude == "bad"
        assert user2.attitude == "good"