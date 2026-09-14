from pi_ai.training.finetuner import freeze_features, trainable_parameters


def test_freeze_features(model):
    before = trainable_parameters(model)
    freeze_features(model)
    assert trainable_parameters(model) < before
