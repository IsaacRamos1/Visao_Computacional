import cv2
import os
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# arquivo data.py
class Data(Dataset):
    def __init__(self, image_dir: str, split: str, transform=None) -> None:
        self._image_dir = image_dir
        self._transform = transform
        self._image_paths_class = []
        self._classe_idx = {'BASH': 0, 'BBH': 1, 'GMA': 2, 'SHC': 3, 'TSH': 4}
        
        for class_dir in glob(f'{image_dir}/{split}/*'):
            class_name = os.path.basename(class_dir)
            image_paths = glob(f'{class_dir}/*.jpg')
            for image_path in image_paths:
                self._image_paths_class.append([image_path, class_name])

        # teste: quantidade de imagens carregadas => caminho está funcionando
        print(f"\nTotal de imagens na pasta {split}: {len(self._image_paths_class)}")

    def __len__(self) -> int:
        return len(self._image_paths_class)

    def __getitem__(self, idx: int) -> tuple:
        image_path, class_name = self._image_paths_class[idx]
        class_name = self._classe_idx[class_name] # dicionario de classe-valores
        # [0, 0, 0, 1, 0]
        image = Image.open(image_path)
        image = np.asarray(image)
        image = image.astype(np.float32) / 255.0 #

        if self._transform:
            augmented = self._transform(image=image)
            image = augmented['image']

        return image, class_name



class Dataloader:
    def __init__(self, shuffle: bool, size: int, subset: int = 0, description: bool = False) -> None:
        # construtor do dataloader
        #self._batch_size = batch_size
        self._shuffle = shuffle
        self._size = size
        self._img_dir = 'data'
        self._subset = subset
        self._transform = self.compose()
        self._description = description

    def compose(self, p: float = 0.5):
        # retornar o compose
        transform_list_train = A.Compose([
            A.Resize(height=self._size, width=self._size),
            A.CLAHE(p=p),                                           
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.Rotate(limit=45, p=p),
            #A.RandomBrightnessContrast(p=p),
            #A.RandomResizedCrop(height=self._size, width=self._size, scale=(0.8, 1.0)),
            #A.CoarseDropout(p=p, max_height=16, max_width=16),
            ToTensorV2()
        ])

        transform_list_val = A.Compose([
            A.Resize(height=self._size, width=self._size),
            ToTensorV2()
        ])

        transform_list_test = A.Compose([
            A.Resize(height=self._size, width=self._size), 
            ToTensorV2()
        ])

        return {'train': transform_list_train,
                'test': transform_list_test,
                'val': transform_list_val}

    def get_dataloader(self, split: str, batch_size: str = 'None') -> DataLoader:
        # retornar o dataloader baseado no split

        dataset = Data(image_dir=self._img_dir, split=split, transform=self._transform[split])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=self._shuffle)

        if self._description:
            print(f'{split} Dataloader: {len(dataloader)} batches: {batch_size}')

        return dataloader

    def get_train_dataloader(self, train_batch_size) -> DataLoader:
        return self.get_dataloader('train', train_batch_size)
    
    def get_val_dataloader(self, val_batch_size) -> DataLoader:
        return self.get_dataloader('val', val_batch_size)
    
    def get_test_dataloader(self, test_batch_size) -> DataLoader:
        return self.get_dataloader('test', test_batch_size)
##