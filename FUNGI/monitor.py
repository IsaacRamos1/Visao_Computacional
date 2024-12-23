class EarlyStopping:
    def __init__(self, patience:int = 5, min_delta:float = 0.0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._best_loss = None
        self._must_stop = False

    def __call__(self, loss):
        if self._best_loss is None:
            self._best_loss = loss
        elif loss > self._best_loss - self._min_delta:
            self._counter += 1
            if self._counter >= self._patience:
                self._must_stop = True
        else:
            self._best_loss = loss
            self._counter = 0
        print(f'Best Loss: {self._best_loss} | Current loss: {loss} | strikes: {self._counter}')

    def must_stop_func(self):
        return self._must_stop