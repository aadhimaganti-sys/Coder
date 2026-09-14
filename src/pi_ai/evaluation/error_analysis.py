def misclassified(predictions, targets):
    return [index for index, (prediction, target) in enumerate(zip(predictions, targets)) if prediction != target]
