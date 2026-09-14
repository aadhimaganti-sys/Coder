from pi_ai.models.parameter_count import count_parameters


def test_parameters_positive(model):
    assert count_parameters(model) > 0
