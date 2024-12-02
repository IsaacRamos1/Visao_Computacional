from torch.utils.tensorboard import SummaryWriter

class TensorBoardMonitor:
    def __init__(self, log_dir: str = "runs", comment: str = "") -> None:
        """
        Inicializa o monitor do TensorBoard.
        :param log_dir: Diretório onde os logs serão armazenados.
        :param comment: Comentário para diferenciar experimentos.
        """
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def log_train_val_metrics(self, epoch: int, metric_name: str, train_value: float, val_value: float) -> None:
        """
        Loga métricas de treino e validação em um único gráfico.
        :param epoch: Número da época atual.
        :param metric_name: Nome da métrica (ex.: 'loss', 'accuracy').
        :param train_value: Valor da métrica para o treino.
        :param val_value: Valor da métrica para a validação.
        """
        # Registra treino e validação com a mesma tag, mas com labels diferentes
        self.writer.add_scalars(metric_name, {
            "train": train_value,
            "val": val_value
        }, epoch)
        
    def log_metrics(self, epoch: int, phase: str, metrics: dict, lr: float = None) -> None:
        """
        Loga as métricas no TensorBoard.
        :param epoch: Número da época atual.
        :param phase: Fase do treinamento ("train" ou "val").
        :param metrics: Dicionário contendo as métricas a serem logadas.
        :param lr: Taxa de aprendizado atual.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"{key}/{phase}", value, epoch)
            #self.writer.add_scalar(key, value, epoch)
        if lr is not None:
            self.writer.add_scalar("LearningRate", lr, epoch)

    def log_model_graph(self, model, inputs) -> None:
        """
        Loga a arquitetura do modelo.
        :param model: Modelo PyTorch.
        :param inputs: Exemplo de entrada para o modelo.
        """
        self.writer.add_graph(model, inputs)

    def close(self) -> None:
        """
        Fecha o escritor do TensorBoard.
        """
        self.writer.close()
