from pi_ai.inference.quantization import quantize_linear_layers


def quantized_model(model):
    return quantize_linear_layers(model.eval())
