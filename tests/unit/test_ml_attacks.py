import pytest
import torch
from collections import OrderedDict
from unittest.mock import MagicMock, patch
from src.ml.attacks import delta_weight_attack, byzantine_attack, manipulate, add_noise, _freerider_submit_with_noise, update_users_attitude


def mock_pytorch_model(prev_state_dict, current_state_dict, noise_scale=0.01):
    pm = MagicMock()
    pm.previous_global_model = prev_state_dict
    pm.global_model.state_dict.return_value = current_state_dict
    pm.freerider_noise_scale = noise_scale
    return pm

def mock_byzantine_pytorch_model(prev_state_dict, current_state_dict, malicious_scale=1.0):
    pm = MagicMock()
    pm.previous_global_model = prev_state_dict
    pm.global_model.state_dict.return_value = current_state_dict
    pm.malicious_noise_scale = malicious_scale
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


class TestDeltaWeightAttack:
    @patch("src.ml.attacks.manipulate")
    def test_fallback_to_noise_when_no_previous_model(self, mock_manipulate):
        # previous_global_model=None, og derfor kaldes manipulate() som fallback (noise)
        pm = MagicMock()
        pm.previous_global_model = None
        pm.freerider_noise_scale = 0.05
        user = MagicMock()

        delta_weight_attack(pm, user)

        mock_manipulate.assert_called_once_with(user.model, scale=0.05)

    def test_crafted_weights_equal_previous_global_model(self):
        # crafted = value + (prev - value) = prev, dermed er resultatet forrige global model
        prev = OrderedDict([("w", torch.tensor([1.0, 2.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0, 4.0]))]) # nuværende global model
        result = delta_weight_attack(mock_pytorch_model(prev, curr), MagicMock())
        assert torch.equal(result["w"], prev["w"])

    def test_integer_layers_cloned_without_modification(self):
        # Ikke-float-tensorer klones direkte fra current model
        prev = OrderedDict([("idx", torch.tensor([10, 20]))]) # forrige global model
        curr = OrderedDict([("idx", torch.tensor([30, 40]))]) # nuværende global model
        result = delta_weight_attack(mock_pytorch_model(prev, curr), MagicMock())
        assert torch.equal(result["idx"], curr["idx"])

    def test_mixed_float_and_int_tensors(self):
        prev = OrderedDict([("w", torch.tensor([1.0])), ("idx", torch.tensor([10]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([5.0])), ("idx", torch.tensor([20]))]) # nuværende global model
        result = delta_weight_attack(mock_pytorch_model(prev, curr), MagicMock())
        assert torch.equal(result["w"],   prev["w"])
        assert torch.equal(result["idx"], curr["idx"])

    def test_returns_ordered_dict(self):
        prev = OrderedDict([("w", torch.tensor([1.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([2.0]))]) # nuværende global model
        result = delta_weight_attack(mock_pytorch_model(prev, curr), MagicMock())
        assert isinstance(result, OrderedDict)

    def test_does_not_modify_original_weights(self):
        # Sikrer at ingen ændringer af prev/curr model sker
        prev = OrderedDict([("w", torch.tensor([1.0, 2.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0, 4.0]))]) # nuværende global model
        prev_snapshot = prev["w"].clone()
        curr_snapshot = curr["w"].clone()
        delta_weight_attack(mock_pytorch_model(prev, curr), MagicMock())
        assert torch.equal(prev["w"], prev_snapshot)
        assert torch.equal(curr["w"], curr_snapshot)


class TestByzantineAttack:
    @patch("src.ml.attacks.manipulate")
    def test_fallback_to_noise_when_no_previous_model(self, mock_manipulate):
        pm = MagicMock()
        pm.previous_global_model = None
        pm.malicious_noise_scale = 0.05
        user = MagicMock()
        byzantine_attack(pm, user)
        mock_manipulate.assert_called_once_with(user.model, scale=0.05)

    def test_scale_1_reverses_to_previous_weights(self):
        # curr - 1*(curr - prev) = prev
        prev = OrderedDict([("w", torch.tensor([1.0, 2.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0, 4.0]))]) # nuværende global model
        result = byzantine_attack(mock_byzantine_pytorch_model(prev, curr, malicious_scale=1.0), MagicMock())
        assert torch.allclose(result["w"], prev["w"])

    def test_scale_0_leaves_current_weights_unchanged(self):
        # curr - 0*(curr - prev) = curr (ingen angreb)
        prev = OrderedDict([("w", torch.tensor([1.0, 2.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0, 4.0]))]) # nuværende global model
        result = byzantine_attack(mock_byzantine_pytorch_model(prev, curr, malicious_scale=0.0), MagicMock())
        assert torch.allclose(result["w"], curr["w"])

    def test_scale_2_overshoots_past_previous(self):
        # curr - 2*(curr - prev) = 2*prev - curr
        prev = OrderedDict([("w", torch.tensor([1.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0]))]) # nuværende global model
        result = byzantine_attack(mock_byzantine_pytorch_model(prev, curr, malicious_scale=2.0), MagicMock())
        assert torch.allclose(result["w"], torch.tensor([-1.0]))

    def test_int_tensors_equal_current_weights(self):
        prev = OrderedDict([("idx", torch.tensor([10, 20]))]) # forrige global model
        curr = OrderedDict([("idx", torch.tensor([30, 40]))]) # nuværende global model
        result = byzantine_attack(mock_byzantine_pytorch_model(prev, curr), MagicMock())
        assert torch.equal(result["idx"], curr["idx"])

    def test_mixed_float_and_int_tensors(self):
        prev = OrderedDict([("w", torch.tensor([1.0])), ("idx", torch.tensor([10]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0])), ("idx", torch.tensor([20]))]) # nuværende global model
        result = byzantine_attack(mock_byzantine_pytorch_model(prev, curr, malicious_scale=1.0), MagicMock())
        assert torch.allclose(result["w"],   prev["w"])
        assert torch.equal(result["idx"],    curr["idx"])

    def test_returns_ordered_dict(self):
        prev = OrderedDict([("w", torch.tensor([1.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([2.0]))]) # nuværende global model
        result = byzantine_attack(mock_byzantine_pytorch_model(prev, curr), MagicMock())
        assert isinstance(result, OrderedDict)

    def test_does_not_modify_original_weights(self):
        prev = OrderedDict([("w", torch.tensor([1.0, 2.0]))]) # forrige global model
        curr = OrderedDict([("w", torch.tensor([3.0, 4.0]))]) # nuværende global model
        prev_snapshot = prev["w"].clone()
        curr_snapshot = curr["w"].clone()
        byzantine_attack(mock_byzantine_pytorch_model(prev, curr), MagicMock())
        assert torch.equal(prev["w"], prev_snapshot)
        assert torch.equal(curr["w"], curr_snapshot)


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