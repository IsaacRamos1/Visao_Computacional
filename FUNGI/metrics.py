from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score
import numpy as np

class ClassificationMetrics:
    def __init__(self, n_classes: int):
        self._losses = []
        self._predicoes = []
        self._labels = []
        self._n_classes = n_classes

    def update(self, loss, predicoes, labels):
        self._losses.append(loss)
        self._predicoes.extend(predicoes.cpu().numpy())
        self._labels.extend(labels.cpu().numpy())

    def metrics(self):
        #classes = np.unique(self._labels)
        #all_classes = np.arange(len(classes))

        loss_medio = np.mean(self._losses)
        acuracia = accuracy_score(self._labels, self._predicoes)
        recall = recall_score(self._labels, self._predicoes, average='weighted', labels=np.arange(self._n_classes), zero_division=0)
        f1 = f1_score(self._labels, self._predicoes, average='weighted', labels=np.arange(self._n_classes), zero_division=0)
        balanced_acc = balanced_accuracy_score(self._labels, self._predicoes)
        return loss_medio, acuracia, recall, f1, balanced_acc

    def reset(self):
        self._losses = []
        self._predicoes = []
        self._labels = []