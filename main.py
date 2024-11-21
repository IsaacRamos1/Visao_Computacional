import cv2
import FUNGI
import FUNGI.model
from FUNGI.model import MyModel, MyPreTrainedModel
from FUNGI.metrics import ClassificationMetrics
from FUNGI.monitor import EarlyStopping
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch


if __name__ == '__main__':
    print('Executando...')
    batch_size = 32
    dataloader = FUNGI.Dataloader(batch_size=batch_size, size=256, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    #model = MyModel(out_channels=5).to('cuda:0')
    model = MyPreTrainedModel(out_channels=5).to('cuda:0')
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    optim = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    crit = nn.CrossEntropyLoss()

    epocas = 10

    for epoca in range(epocas):
        train_metrics = ClassificationMetrics()
        val_metrics = ClassificationMetrics()

        model.train()
        #run_acc = []
        #run_loss = []
        train_samples = tqdm(train_dataloader)

        for image, label in train_samples:
            image = image.to('cuda:0')
            label = label.to('cuda:0')
            optim.zero_grad()
            output = model(image)
            loss = crit(output, label)
            loss.backward()
            optim.step()
            #run_loss.append(loss.item())
            predicts = torch.argmax(output, dim=1)
            train_metrics.update(loss.item(), predicts, label)
            _, run_acc, _, _, balanced_acc = train_metrics.metrics()

            train_samples.set_description(f'Epoca {epoca+1}/{epocas}, loss: {np.mean(train_metrics._losses):0.6f}, balanced acc: {balanced_acc:0.6f}')


        model.eval()
        with torch.no_grad():
            val_running_loss = []
            val_running_acc = []

            val_samples = tqdm(val_dataloader)
            for image, label in val_samples:
                image = image.to('cuda:0')
                label = label.to('cuda:0')
                output = model(image)
                loss = crit(output, label)
                predicts = torch.argmax(output, dim=1)
                val_metrics.update(loss.item(), predicts, label)
                val_loss, val_acc, _, _, _ = val_metrics.metrics()
                val_running_loss.append(val_loss)
                val_running_acc.append(val_acc)

            scheduler.step(np.mean(val_running_loss))
            print(scheduler.get_last_lr())
            early_stopping(np.mean(val_running_loss))

            if early_stopping.must_stop_func():
                print('Early Stopping .......')
                print(f'\nEpoca {epoca+1}/{epocas}')
                print(f'Treino -> Loss: {train_loss:0.6f}, Acurácia: {train_acc:0.6f}, \
                  \nRecall: {train_recall:0.6f}, F1-Score: {train_f1:0.6f}, Balanced_acc: {train_balanced_acc: 0.6f}')
                
                print(f'Validação -> Loss: {val_loss:0.6f}, Acurácia: {val_acuracia:0.6f}, \
                  \nRecall: {val_recall:0.6f}, F1-Score: {val_f1:0.6f}, Balanced_acc: {val_balanced_acc:0.6f}\n')
                break
            else:
                print('continuando...')

            train_loss, train_acc, train_recall, train_f1, train_balanced_acc = train_metrics.metrics()
            val_loss, val_acuracia, val_recall, val_f1, val_balanced_acc = val_metrics.metrics()

            print(f'\nEpoca {epoca+1}/{epocas}')
            print(f'Treino -> Loss: {train_loss:0.6f}, Acurácia: {train_acc:0.6f}, \
                  \nRecall: {train_recall:0.6f}, F1-Score: {train_f1:0.6f}, Balanced_acc: {train_balanced_acc: 0.6f}')
            
            print(f'Validação -> Loss: {val_loss:0.6f}, Acurácia: {val_acuracia:0.6f}, \
                  \nRecall: {val_recall:0.6f}, F1-Score: {val_f1:0.6f}, Balanced_acc: {val_balanced_acc:0.6f}\n')


