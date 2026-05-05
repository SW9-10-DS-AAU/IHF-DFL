import pytest
from unittest.mock import MagicMock
from src.ml.aggregation import positives_only, plus_one_normalize, plus_more_than_one_normalize, GRS_aggregation

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