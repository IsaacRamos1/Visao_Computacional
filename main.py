import cv2
import FUNGI
import FUNGI.model
from FUNGI.model import MyModel, MyPreTrainedModel
from FUNGI.metrics import ClassificationMetrics
from FUNGI.monitor import EarlyStopping
from FUNGI.log import TensorboardLogger
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime


if __name__ == '__main__':
    print('Executando...')
    #log_dir = f"runs/{datetime.now().strftime('%dd%mm%yy-%H%M%S')}"
    batch_size = 32
    dataloader = FUNGI.Dataloader(batch_size=batch_size, size=256, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    #model = MyModel(out_channels=5).to('cuda:0')
    model = MyPreTrainedModel(out_channels=5).to('cuda:0')
    early_stopping = EarlyStopping(patience=30, min_delta=0.0001)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    crit = nn.CrossEntropyLoss()

    epocas = 1000

    for epoca in range(epocas):
        train_metrics = ClassificationMetrics(n_classes=5)
        val_metrics = ClassificationMetrics(n_classes=5)

        model.train()
        train_loss_list = []
        train_acc_list = []
        train_recall_list = []
        train_f1_list = []
        train_balanced_acc_list = []

        train_samples = tqdm(train_dataloader)
        train_metrics.reset()
        val_metrics.reset()

        for image, label in train_samples:
            image = image.to('cuda:0')
            label = label.to('cuda:0')
            optimizer.zero_grad()
            output = model(image)
            loss = crit(output, label)
            loss.backward()
            optimizer.step()
            predicts = torch.argmax(output, dim=1)
            train_metrics.update(loss.item(), predicts, label)
            train_loss, train_acc, train_recall, train_f1, train_balanced_acc = train_metrics.metrics()
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            train_recall_list.append(train_recall)
            train_f1_list.append(train_f1)
            train_balanced_acc_list.append(train_balanced_acc)

            train_samples.set_description(f'Epoca {epoca+1}/{epocas}, loss: {np.mean(train_loss):0.6f}, balanced acc: {train_balanced_acc:0.6f}')

        train_loss_mean = np.mean(train_loss_list)
        train_acc_mean = np.mean(train_acc_list)
        train_recall_mean = np.mean(train_recall_list)
        train_f1_mean = np.mean(train_f1_list)
        train_balanced_acc_mean = np.mean(train_balanced_acc_list)

        model.eval()
        with torch.no_grad():
            val_loss_list = []
            val_acc_list = []
            val_recall_list = []
            val_f1_list = []
            val_balanced_acc_list = []

            val_samples = tqdm(val_dataloader)
            for image, label in val_samples:
                image = image.to('cuda:0')
                label = label.to('cuda:0')
                output = model(image)
                loss = crit(output, label)
                predicts = torch.argmax(output, dim=1)
                val_metrics.update(loss.item(), predicts, label)
                val_loss, val_acc, val_recall, val_f1, val_balanced_acc = val_metrics.metrics()
                val_loss_list.append(loss.item())
                val_acc_list.append(val_acc)
                val_recall_list.append(val_recall)
                val_f1_list.append(val_f1)
                val_balanced_acc_list.append(val_balanced_acc)

        val_loss_mean = np.mean(val_loss_list)
        val_acc_mean = np.mean(val_acc_list)
        val_recall_mean = np.mean(val_recall_list)
        val_f1_mean = np.mean(val_f1_list)
        val_balanced_acc_mean = np.mean(val_balanced_acc_list)

        scheduler.step(val_loss_mean)
        print(scheduler.get_last_lr())
        early_stopping(val_loss_mean)

        if early_stopping.must_stop_func():
            print('Early Stopping .......')
            break
        else:
            print('continuando...')


        print(f'\nEpoca {epoca+1}/{epocas}')
        print(f'Treino -> Loss: {train_loss_mean:0.6f}, Acurácia: {train_acc_mean:0.6f}, \
            \nRecall: {train_recall_mean:0.6f}, F1-Score: {train_f1_mean:0.6f}, Balanced_acc: {train_balanced_acc_mean: 0.6f}')
        
        print(f'Validação -> Loss: {val_loss_mean:0.6f}, Acurácia: {val_acc_mean:0.6f}, \
            \nRecall: {val_recall_mean:0.6f}, F1-Score: {val_f1_mean:0.6f}, Balanced_acc: {val_balanced_acc_mean:0.6f}\n')
