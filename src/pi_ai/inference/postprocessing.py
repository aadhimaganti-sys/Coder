from .predictor import Prediction


def best_prediction(predictions: list[Prediction]) -> Prediction:
    if not predictions:
        raise ValueError("prediction list is empty")
    return predictions[0]
