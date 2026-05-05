import pytest
from src.ml.aggregation import positives_only

class TestPositivesOnly:
    @pytest.mark.parametrize("scores_input, expected", [
        ({"a": 2.0, "b": 2.0, "c": 4.0}, {"a": 0.25, "b": 0.25, "c": 0.5}), # Alle positive — normaliseres til sum = 1
        ({"a": 5.0, "b": -1.0, "c": 0.0}, {"a": 1.0, "b": 0.0, "c": 0.0}), # Én positiv — bliver 1.0
        ({"a": 3.0, "b": -1.0, "c": 1.0}, {"a": 0.75, "b": 0.0, "c": 0.25}), # Blandede — negative og nul bliver 0
        ({"a": 2.0, "b": 2.0, "c": 2.0}, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), # Alle ens — ligeligt fordelt
        ({"a": 1.0, "b": 1.0, "c": -2.0}, {"a": 0.5, "b": 0.5, "c": 0.0}), # Sum = 0
        ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "b": 0.3, "c": 0.2}), # kommatal
        ({"a": -1, "b": 1}, {"a": 0, "b":1}), # 1 negativ, 1 positiv}
    ])
    def test_positives_only(self, scores_input, expected):
        result = positives_only(scores_input)
        assert result == pytest.approx(expected)


    @pytest.mark.parametrize("scores_input", [
        {"a": -1.0, "b": -2.0},  # alle negative
        {"a": 0.0, "b": 0.0},  # alle nul
        {"a": -1.0, "b": 0.0},  # blanding af negative og nul
    ])
    def test_positives_only_raises_when_no_positives(self, scores_input):
        with pytest.raises(Exception, match="No positive contribution scores"):
            positives_only(scores_input)