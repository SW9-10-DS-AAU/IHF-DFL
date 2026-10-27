import pytest

from experiment.helper import recorded_run_counts_as_done


@pytest.mark.parametrize("malicious_round,malicious_noise", [(None, None), (1, None), (None, 0.01)])
def test_recorded_run_without_malicious_values_is_not_done(malicious_round, malicious_noise):
    # Runs recorded before #83 stored None here, and their malicious participant never acted.
    assert not recorded_run_counts_as_done("mnist_openfl_low_noise", malicious_round, malicious_noise)


def test_recorded_run_with_malicious_values_is_done():
    assert recorded_run_counts_as_done("mnist_openfl_low_noise", 1, 0.01)


def test_recorded_run_of_unknown_preset_without_malicious_values_is_not_done():
    assert not recorded_run_counts_as_done("no_such_preset", None, None)
