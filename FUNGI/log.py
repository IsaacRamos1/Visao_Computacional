from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    def __init__(self, log_dir: str = 'runs'):
        """
        Inicializa o logger do TensorBoard.
        Args:
            log_dir (str): Diretório para salvar os logs do TensorBoard.
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, phase: str, epoch: int, loss: float, accuracy: float, recall: float, f1: float, balanced_acc: float):
        """
        Registra as métricas no TensorBoard.
        Args:
            phase (str): Fase atual ('train' ou 'val').
            epoch (int): Época atual.
            loss (float): Perda.
            accuracy (float): Acurácia.
            recall (float): Recall.
            f1 (float): F1-score.
            balanced_acc (float): Balanced Accuracy.
        """
        self.writer.add_scalar(f'{phase}/Loss', loss, epoch)
        self.writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        self.writer.add_scalar(f'{phase}/Recall', recall, epoch)
        self.writer.add_scalar(f'{phase}/F1-Score', f1, epoch)
        self.writer.add_scalar(f'{phase}/Balanced_Accuracy', balanced_acc, epoch)

    def close(self):
        """
        Fecha o logger do TensorBoard.
        """
        self.writer.close()
