class EarlyStopping:
    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self.best = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        if self.best is None or value < self.best:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
