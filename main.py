import cv2
import FUNGI
import FUNGI.model
from FUNGI.model import MyModel, MyPreTrainedModel
from FUNGI.metrics import ClassificationMetrics
from FUNGI.monitor import EarlyStopping
from FUNGI.log import TensorBoardMonitor
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report



if __name__ == '__main__':
    print('Executando...')
    #batch_size = 26
    #dataloader = FUNGI.Dataloader(batch_size=batch_size, size=300, shuffle=True, description=True)
    dataloader = FUNGI.Dataloader(size=224, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader(train_batch_size=32)
    val_dataloader = dataloader.get_val_dataloader(val_batch_size=32)
    test_dataloader = dataloader.get_test_dataloader(test_batch_size=32)

    #model = MyModel(out_channels=5).to('cuda:0')
    name = 'train_val_test_Efficientnet_b0_'
    model = MyPreTrainedModel(name=name, out_channels=5).to('cuda:0')
    log_dir = f"runs/{model._name + datetime.now().strftime('%dd%mm%yy-%H%M%S')}"
    tensorboard_monitor = TensorBoardMonitor(log_dir=(log_dir), comment="modelo_pre_treinado")
    #print(model)
    patience = 30
    min_delta = 0.0001
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    learning_rate = 0.0001
    factor = 0.05
    scheduler_patience = 5
    scheduler_cooldown = 20
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=scheduler_patience, cooldown=scheduler_cooldown)
    epocas = 1000

    wandb.init(
        # set the wandb project where this run will be logged
        project="isaac-project",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": name,
        "dataset": "DeFungi-5",
        "epochs": epocas,
        "early stopping": {'patience': patience, 'min_delta': min_delta},
        'optim': optimizer,
        'factor': factor,
        'scheduler_patience': scheduler_patience
        }
    )

    crit = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()

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
            #output = torch.nn.functional.softmax(output, dim=-1)
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

        wandb.log({"train_loss": train_loss_mean, "train_balanced_acc": train_balanced_acc_mean})

        tensorboard_monitor.log_metrics(epoca, "train", {
            "loss": train_loss_mean,
            "accuracy": train_acc_mean,
            "recall": train_recall_mean,
            "f1_score": train_f1_mean,
            "balanced_accuracy": train_balanced_acc_mean
        })


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

        wandb.log({"val_loss": val_loss_mean, "val_balanced_acc": val_balanced_acc_mean})

        tensorboard_monitor.log_metrics(epoca, "val", {
            "loss": val_loss_mean,
            "accuracy": val_acc_mean,
            "recall": val_recall_mean,
            "f1_score": val_f1_mean,
            "balanced_accuracy": val_balanced_acc_mean
        })

        tensorboard_monitor.log_train_val_metrics(epoca, "Loss", train_loss_mean, val_loss_mean)
        tensorboard_monitor.log_train_val_metrics(epoca, "accuracy", train_acc_mean, val_acc_mean)
        tensorboard_monitor.log_train_val_metrics(epoca, "recall", train_recall_mean, val_recall_mean)
        tensorboard_monitor.log_train_val_metrics(epoca, "f1_score", train_f1_mean, val_f1_mean)
        tensorboard_monitor.log_train_val_metrics(epoca, "balanced_accuracy", train_balanced_acc_mean, val_balanced_acc_mean)

        scheduler.step(val_loss_mean)
        print(scheduler.get_last_lr())
        tensorboard_monitor.writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoca)
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

    # TESTE VEM AQUI
    print("Iniciando teste...")

    model.eval()
    test_metrics = ClassificationMetrics(n_classes=5)

    test_loss_list = []
    test_acc_list = []
    test_recall_list = []
    test_f1_list = []
    test_balanced_acc_list = []

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        test_samples = tqdm(test_dataloader)
        for image, label in test_samples:
            image = image.to('cuda:0')
            label = label.to('cuda:0')
            output = model(image)
            loss = crit(output, label)

            predicts = torch.argmax(output, dim=1)
            all_predictions.extend(predicts.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            test_metrics.update(loss.item(), predicts, label)
            test_loss, test_acc, test_recall, test_f1, test_balanced_acc = test_metrics.metrics()

            test_loss_list.append(loss.item())
            test_acc_list.append(test_acc)
            test_recall_list.append(test_recall)
            test_f1_list.append(test_f1)
            test_balanced_acc_list.append(test_balanced_acc)
        
    test_loss_mean = np.mean(test_loss_list)
    test_acc_mean = np.mean(test_acc_list)
    test_recall_mean = np.mean(test_recall_list)
    test_f1_mean = np.mean(test_f1_list)
    test_balanced_acc_mean = np.mean(test_balanced_acc_list)

    #wandb.log({"test_loss": test_loss_mean, "test_balanced_acc": test_balanced_acc_mean})

    tensorboard_monitor.log_metrics(epoca, "test", {
        "loss": test_loss_mean,
        "accuracy": test_acc_mean,
        "recall": test_recall_mean,
        "f1_score": test_f1_mean,
        "balanced_accuracy": test_balanced_acc_mean
    })

    print(f'Teste -> Loss: {test_loss_mean:0.6f}, Acurácia: {test_acc_mean:0.6f}, \
    \nRecall: {test_recall_mean:0.6f}, F1-Score: {test_f1_mean:0.6f}, Balanced_acc: {test_balanced_acc_mean:0.6f}')

    class_report = classification_report(all_labels, all_predictions, target_names=[f'Classe {i}' for i in range(5)])
    print("Relatório de Classificação:\n", class_report) 

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.title('Matriz de Confusão')
    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.show()

    tensorboard_monitor.close()