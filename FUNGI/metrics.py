from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

class ClassificationMetrics:
    def __init__(self):
        self._losses = []
        self._predicoes = []
        self._labels = []

    def update(self, loss, predicoes, labels):
        self._losses.append(loss)
        self._predicoes.extend(predicoes.cpu().numpy())
        self._labels.extend(labels.cpu().numpy())

    def metrics(self):
        loss_medio = np.mean(self._losses)
        acuracia = accuracy_score(self._labels, self._predicoes)
        recall = recall_score(self._labels, self._predicoes, average='macro')
        f1 = f1_score(self._labels, self._predicoes, average='macro')
        return loss_medio, acuracia, recall, f1

    def reset(self):
        self._losses = []
        self._predicoes = []
        self._labels = []