from torch.utils.tensorboard import SummaryWriter
import os

# Inicializar o SummaryWriter
writer = SummaryWriter(log_dir='runs')

# Exemplo de logging
for epoch in range(10):
    writer.add_scalar('Loss/train', 0.1 * (10 - epoch), epoch)
    writer.add_scalar('Accuracy/train', 0.9 + 0.01 * epoch, epoch)
writer.close()