from pi_ai.monitoring.resource_monitor import max_memory_mb


def test_memory_is_nonnegative():
    assert max_memory_mb() >= 0
