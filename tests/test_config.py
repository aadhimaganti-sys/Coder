from pi_ai.config import load_config


def test_defaults_are_valid():
    assert load_config().threads > 0
