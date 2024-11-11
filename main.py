import cv2
import FUNGI
import FUNGI.model
from FUNGI.model import MyModel
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

if __name__ == '__main__':
    print('Executando...')
    batch_size = 3
    dataloader = FUNGI.Dataloader(batch_size=batch_size, size=500, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    #classe_idx = {'BASH': 0, 'BBH': 1, 'GMA': 2, 'SHC': 3, 'TSH': 4}

<<<<<<< HEAD
    model = MyModel(out_channels=5).to('cuda:0')
    optim = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    epocas = 10

    for epoca in range(epocas):
        model.train()
        run_loss = []
        train_samples = tqdm(train_dataloader)

        for image, label in train_samples:
            image = image.to('cuda:0')
            label = label.to('cuda:0')
            optim.zero_grad()
            output = model(image)
            print(f'<><><><>{output.shape}')
            print(f'<><><><>{label.shape}')
            #exit()
            loss = crit(output, label)
            loss.backward()
            optim.step()
            run_loss.append(loss.item())
            train_samples.set_description(f'Epoca {epoca+1}/{epocas}, loss: {np.mean(run_loss):0.6f}')
            
            model.eval()
            val_samples = tqdm(val_dataloader)
            for image, label in val_samples:
                pass





#for image, label in train_dataloader:
#            for batch in range(batch_size):
#                image = image.detach().cpu().numpy()[batch].transpose(1, 2, 0)
#                print(f'Class label: {label[batch]}')
#                cv2.imshow('image', image)

#                if cv2.waitKey(0) == ord('q'):
#                    break
=======
            if cv2.waitKey(0) == ord('q'):
                break
##
>>>>>>> d12ae69e0ef19960b8889fa3f1aee21978470f1f
