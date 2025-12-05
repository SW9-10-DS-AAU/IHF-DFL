from experiment.experiment_configuration import ExperimentConfiguration


def test_number_of_contributors_property():
    cfg = ExperimentConfiguration(
        number_of_good_contributors=2,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        number_of_inactive_contributors=0,
    )

    assert cfg.number_of_contributors == 4
