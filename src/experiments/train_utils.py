class EarlyStopper:
    def __init__(self, patience=1, min_rel_delta=0.0):
        """Stop training early if validation loss does not improve enough."""
        self.patience = patience
        self.min_rel_delta = min_rel_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (
            self.min_validation_loss + self.min_validation_loss * self.min_rel_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
