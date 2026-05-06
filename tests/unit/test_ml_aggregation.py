import math
import pytest
from unittest.mock import MagicMock, patch
from src.ml.aggregation import positives_only, plus_one_normalize, plus_more_than_one_normalize, GRS_aggregation, binary_switch, partial_switch_fixed_loss, partial_switch_loss_retrospective

def func_1(scores): return {"func": "one"}
def func_2(scores): return {"func": "two"}

def func_all_to_a(scores): return {"a": 1.0, "b": 0.0}
def func_all_to_b(scores): return {"a": 0.0, "b": 1.0}

def mock_pytorch_model(has_switched=False, has_two_previous=False):
    pm = MagicMock()
    pm.has_switched = has_switched
    pm.two_previous_global_model = MagicMock() if has_two_previous else None
    pm.previous_global_model = MagicMock()
    return pm


def mock_user(address, last_globalrep):
    user = MagicMock()
    user.address = address
    user._globalrep = [last_globalrep]
    return user

class TestPositivesOnly:
    @pytest.mark.parametrize("scores_input, expected", [
        ({"a": 2.0, "b": 2.0, "c": 4.0}, {"a": 0.25, "b": 0.25, "c": 0.5}), # Alle positive — normaliseres til sum = 1
        ({"a": 5.0, "b": -1.0, "c": 0.0}, {"a": 1.0, "b": 0.0, "c": 0.0}), # Én positiv — bliver 1.0
        ({"a": 3.0, "b": -1.0, "c": 1.0}, {"a": 0.75, "b": 0.0, "c": 0.25}), # Blandede — negative og nul bliver 0
        ({"a": 2.0, "b": 2.0, "c": 2.0}, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), # Alle ens — ligeligt fordelt
        ({"a": 1.0, "b": 1.0, "c": -2.0}, {"a": 0.5, "b": 0.5, "c": 0.0}), # Sum = 0
        ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "b": 0.3, "c": 0.2}), # Kommatal
        ({"a": -1, "b": 1}, {"a": 0, "b":1}), # 1 negativ, 1 positiv}
    ])
    def test_positives_only(self, scores_input, expected):
        result = positives_only(scores_input)
        assert result == pytest.approx(expected)


    @pytest.mark.parametrize("scores_input", [
        {"a": -1.0, "b": -2.0},  # Alle negative
        {"a": 0.0, "b": 0.0},  # Alle nul
        {"a": -1.0, "b": 0.0},  # Blanding af negative og nul
    ])
    def test_positives_only_raises_when_no_positives(self, scores_input):
        with pytest.raises(Exception, match="No positive contribution scores"):
            positives_only(scores_input)


class TestPlusOneNormalize:
    @pytest.mark.parametrize("scores_input, expected", [
        ({"a": 2.0, "b": 2.0, "c": 4.0}, {"a": 3/11, "b": 3/11, "c": 5/11}), # Alle positive — shift +1, normaliser
        ({"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1/3, "b": 1/3, "c": 1/3}), # Alle ens — ligeligt fordelt
        ({"a": 0.0, "b": 0.0}, {"a": 0.5, "b": 0.5}), # Alle nul — ligeligt fordelt
        ({"a": -1.0, "b": 3.0}, {"a": 0.0, "b": 1.0}), # Negativ score der bliver nul efter +1 — får vægt 0
        ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 1.5/4.0, "b": 1.3/4.0, "c": 1.2/4.0}), # Kommatal
    ])
    def test_plus_one_normalize(self, scores_input, expected):
        result = plus_one_normalize(scores_input)
        assert result == pytest.approx(expected)


    @pytest.mark.parametrize("scores_input", [
        {"a": -1.0, "b": -1.0},  # Alle scores = -1 → sum = 0 → ZeroDivisionError
    ])
    def test_plus_one_normalize_raises_on_zero_sum(self, scores_input):
        with pytest.raises(ZeroDivisionError):
            plus_one_normalize(scores_input)


class TestPlusMoreThanOneNormalize:
    @pytest.mark.parametrize("scores_input, more_than_one, expected", [
        ({"a": 2.0, "b": 2.0, "c": 4.0}, 1.1, {"a": 3.1/11.3, "b": 3.1/11.3, "c": 5.1/11.3}), # Default (1.1): alle positive — shift +1.1
        ({"a": 1.0, "b": 1.0, "c": 1.0}, 1.1, {"a": 1/3, "b": 1/3, "c": 1/3}), # Default: alle ens — ligeligt fordelt
        ({"a": -1.1, "b": 3.0}, 1.1, {"a": 0.0, "b": 1.0}), # Default: score der præcis omvender fortegn → vægt 0
        ({"a": 0.5, "b": 0.3, "c": 0.2}, 1.1, {"a": 0.3721, "b": 0.3256, "c": 0.3023}), # Kommatal
        ({"a": 1.0, "b": 3.0}, 2.0, {"a": 3/8, "b": 5/8}), # Brugerdefineret more_than_one=2.0 (TEST)
    ])
    def test_plus_more_than_one_normalize(self, scores_input, more_than_one, expected):
        result = plus_more_than_one_normalize(scores_input, more_than_one)
        assert result == pytest.approx(expected, abs=1e-3)


    @pytest.mark.parametrize("scores_input, more_than_one", [
        ({"a": -1.1, "b": -1.1}, 1.1),  # sum = 0 → ZeroDivisionError
        ({"a": -2.0, "b": -2.0}, 2.0),  # Custom shift, sum = 0
    ])
    def test_plus_more_than_one_normalize_raises_on_zero_sum(self, scores_input, more_than_one):
        with pytest.raises(ZeroDivisionError):
            plus_more_than_one_normalize(scores_input, more_than_one)



class TestGRSAggregation:
    @pytest.mark.parametrize("user_data, expected", [
        ([("a", 1.0), ("b", 3.0)], {"a": 0.25, "b": 0.75}), # Proportional fordeling
        ([("a", 2.0), ("b", 2.0), ("c", 2.0)], {"a": 1/3, "b": 1/3, "c": 1/3}), # Alle ens — ligeligt fordelt
        ([("a", 5.0)], {"a": 1.0}), # Én bruger — 100%
        ([("a", 0.5), ("b", 1.5)], {"a": 0.25, "b": 0.75}), # Kommatal
    ])
    def test_grs_aggregation(self, user_data, expected):
        users = [mock_user(addr, rep) for addr, rep in user_data]
        result = GRS_aggregation(users)
        assert result == pytest.approx(expected)

    def test_grs_aggregation_raises_on_zero_total(self):
        users = [mock_user("a", 0.0), mock_user("b", 0.0)]
        with pytest.raises(ZeroDivisionError):
            GRS_aggregation(users)


SCORES = {"a": 2.0, "b": 2.0}

class TestBinarySwitch:
    def test_uses_func_1_at_round_1(self):
        # Round 1, hvor betingelsen current_round_no > 1 er falsk, dermed ingen switch-check
        result = binary_switch(mock_pytorch_model(), SCORES, func_1, func_2, None, _current_round_no=1)
        assert result == {"func": "one"}

    def test_uses_func_1_when_no_two_previous_model(self):
        # two_previous_global_model er None, hvor betingelsen er falsk, dermed ingen switch-check
        result = binary_switch(mock_pytorch_model(has_two_previous=False), SCORES, func_1, func_2, None, _current_round_no=2)
        assert result == {"func": "one"}

    @patch("src.ml.aggregation.models_are_equal", return_value=False)
    def test_uses_func_1_when_models_not_equal(self, _):
        # Modeller er ikke ens, og derfor ingen switch
        result = binary_switch(mock_pytorch_model(has_two_previous=True), SCORES, func_1, func_2, None, _current_round_no=2)
        assert result

    @patch("src.ml.aggregation.models_are_equal", return_value=True)
    def test_switches_to_func_2_on_convergence(self, _):
        # Konvergens detekteret, derfor bruges func_2 og sætter has_switched=True
        pm = mock_pytorch_model(has_two_previous=True)
        result = binary_switch(pm, SCORES, func_1, func_2, None, _current_round_no=2)
        assert result == {"func": "two"}
        assert pm.has_switched is True

    def test_uses_func_2_when_already_switched(self):
        # has_switched=True, dermed bruges func_2 direkte uden at tjekke modeller
        result = binary_switch(mock_pytorch_model(has_switched=True), SCORES, func_1, func_2, None, _current_round_no=1)
        assert result == {"func": "two"}

    def test_collector_updated_before_switch(self):
        # Før switch: func_1 har vægt 1.0, func_2 har vægt 0.0
        collector = {}
        binary_switch(mock_pytorch_model(), SCORES, func_1, func_2, collector, _current_round_no=1)
        assert collector == {"func_1": "func_1", "weight_1": 1.0, "func_2": "func_2", "weight_2": 0.0}

    @patch("src.ml.aggregation.models_are_equal", return_value=True)
    def test_collector_updated_after_switch(self, _):
        # Efter switch: func_1 har vægt 0.0, func_2 har vægt 1.0
        collector = {}
        binary_switch(mock_pytorch_model(has_two_previous=True), SCORES, func_1, func_2, collector, _current_round_no=2)
        assert collector == {"func_1": "func_1", "weight_1": 0.0, "func_2": "func_2", "weight_2": 1.0}

    def test_collector_none_does_not_raise(self):
        # collector=None, som betyder ingen fejl
        binary_switch(mock_pytorch_model(), SCORES, func_1, func_2, None, _current_round_no=1)


class TestPartialSwitchFixedLoss:
    @pytest.mark.parametrize("avg_prior_losses, threshold, expected", [
        (None,  100, {"a": 1.0, "b": 0.0}),  # None → alpha=1.0, derfor kun func_1
        (200.0, 100, {"a": 1.0, "b": 0.0}),  # loss > threshold, derfor alpha=1.0, og kun func_1
        (100.0, 100, {"a": 1.0, "b": 0.0}),  # loss == threshold, derfor alpha=1.0, og kun func_1
        (0.0,   100, {"a": 0.0, "b": 1.0}),  # loss=0, derfor alpha=sin(0°)=0, og kun func_2
    ])
    def test_boundary_cases(self, avg_prior_losses, threshold, expected):
        result = partial_switch_fixed_loss(SCORES, avg_prior_losses, func_all_to_a, func_all_to_b, threshold)
        assert result == pytest.approx(expected)

    def test_midpoint_follows_sin_curve(self):
        alpha = math.sin(math.radians(45))
        result = partial_switch_fixed_loss(SCORES, 50.0, func_all_to_a, func_all_to_b, threshold=100)
        assert result == pytest.approx({"a": alpha, "b": 1 - alpha})

    def test_custom_threshold(self):
        alpha = math.sin(math.radians(45))
        result = partial_switch_fixed_loss(SCORES, 25.0, func_all_to_a, func_all_to_b, threshold=50)
        assert result == pytest.approx({"a": alpha, "b": 1 - alpha})

    def test_output_sums_to_one(self):
        result = partial_switch_fixed_loss(SCORES, 50.0, func_all_to_a, func_all_to_b)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_fallback_uniform_when_total_is_zero(self):
        # Begge funcs returnerer 0, betydende total=0, derfor ensartet fordeling
        def func_zero(scores): return {k: 0.0 for k in scores}
        result = partial_switch_fixed_loss(SCORES, 50.0, func_zero, func_zero)
        assert result == pytest.approx({"a": 0.5, "b": 0.5})

    def test_collector_updated_when_fully_func1(self):
        collector = {}
        partial_switch_fixed_loss(SCORES, None, func_all_to_a, func_all_to_b, agg_switch_collector=collector)
        assert collector["func_1"] == "func_all_to_a"
        assert collector["weight_1"] == pytest.approx(1.0)
        assert collector["func_2"] == "func_all_to_b"
        assert collector["weight_2"] == pytest.approx(0.0)

    def test_collector_updated_on_partial_switch(self):
        collector = {}
        partial_switch_fixed_loss(SCORES, 50.0, func_all_to_a, func_all_to_b, threshold=100, agg_switch_collector=collector)
        expected_alpha = math.sin(math.radians(45))
        assert collector["weight_1"] == pytest.approx(expected_alpha)
        assert collector["weight_2"] == pytest.approx(1.0 - expected_alpha)

    def test_collector_none_does_not_raise(self):
        partial_switch_fixed_loss(SCORES, 50.0, func_all_to_a, func_all_to_b, agg_switch_collector=None)


class TestPartialSwitchLossRetrospective:
    @pytest.mark.parametrize("avg_prior_losses, expected", [
        ([5.0],       {"a": 1.0, "b": 0.0}),  # Kun 1 loss, improvement_ratio=1.0, defor kun func_1
        ([4.0, 4.0],  {"a": 0.0, "b": 1.0}),  # Ens værdier, men med forskellige output
        ([5.0, 3.0],  {"a": 0.0, "b": 1.0}),  # Forværring, med ratio=0, derfor kun func_2
        ([0.0, 10.0], {"a": 1.0, "b": 0.0}),  # Hurtig forbedring, hvor ratio fastsat til 1, derfor kun func_1
        ([0.0, 0.0],  {"a": 1.0, "b": 0.0}),  # mean_loss=0 med ratio=1.0, derfor kun func_1
    ])
    def test_boundary_cases(self, avg_prior_losses, expected):
        result = partial_switch_loss_retrospective(SCORES, avg_prior_losses, func_all_to_a, func_all_to_b)
        assert result == pytest.approx(expected)

    def test_improving_loss_follows_sin_curve(self):
        alpha = math.sin(math.radians(45))
        result = partial_switch_loss_retrospective(SCORES, [3.0, 5.0], func_all_to_a, func_all_to_b)
        assert result == pytest.approx({"a": alpha, "b": 1 - alpha})

    def test_multiple_points_regression(self):
        alpha = math.sin(math.radians(45))
        result = partial_switch_loss_retrospective(SCORES, [2.0, 4.0, 6.0], func_all_to_a, func_all_to_b)
        assert result == pytest.approx({"a": alpha, "b": 1 - alpha})

    def test_output_sums_to_one(self):
        result = partial_switch_loss_retrospective(SCORES, [3.0, 5.0], func_all_to_a, func_all_to_b)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_fallback_uniform_when_total_is_zero(self):
        def func_zero(scores): return {k: 0.0 for k in scores}
        result = partial_switch_loss_retrospective(SCORES, [3.0, 5.0], func_zero, func_zero)
        assert result == pytest.approx({"a": 0.5, "b": 0.5})

    def test_collector_updated_on_plateau(self):
        collector = {}
        partial_switch_loss_retrospective(SCORES, [4.0, 4.0], func_all_to_a, func_all_to_b, agg_switch_collector=collector)
        assert collector["func_1"] == "func_all_to_a"
        assert collector["weight_1"] == pytest.approx(0.0)
        assert collector["func_2"] == "func_all_to_b"
        assert collector["weight_2"] == pytest.approx(1.0)

    def test_collector_updated_on_partial_blend(self):
        # ratio=0.5 → alpha=sin(45°)
        collector = {}
        alpha = math.sin(math.radians(45))
        partial_switch_loss_retrospective(SCORES, [3.0, 5.0], func_all_to_a, func_all_to_b, agg_switch_collector=collector)
        assert collector["weight_1"] == pytest.approx(alpha)
        assert collector["weight_2"] == pytest.approx(1 - alpha)

    def test_collector_none_does_not_raise(self):
        partial_switch_loss_retrospective(SCORES, [3.0, 5.0], func_all_to_a, func_all_to_b, agg_switch_collector=None)