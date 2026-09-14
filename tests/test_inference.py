from pi_ai.inference.predictor import Predictor


def test_predictor(model, batch):
    result = Predictor(model).predict(batch, 2)
    assert len(result) == 4 and len(result[0]) == 2
