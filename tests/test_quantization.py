from pi_ai.inference.quantization import quantize_linear_layers


def test_quantization_returns_model(model):
    assert quantize_linear_layers(model) is not None
