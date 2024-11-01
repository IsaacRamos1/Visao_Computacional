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
        self._image_paths_class = []

        for class_dir in glob(f'{image_dir}/{split}/*'):
            class_name = os.path.basename(class_dir)
            image_paths = glob(f'{class_dir}/*.jpg')
            for image_path in image_paths:
                self._image_paths_class.append([image_path, class_name])

        # teste: quantidade de imagens carregadas => caminho está funcionando
        print(f"Total de imagens encontradas: {len(self._image_paths_class)}")
        
        self._transform = transform

    def __len__(self) -> int:
        return len(self._image_paths_class)

    def __getitem__(self, idx: int) -> tuple:
        image_path, class_name = self._image_paths_class[idx]
        image = Image.open(image_path)
        image = np.asarray(image)

        if self._transform:
            augmented = self._transform(image=image)
            image = augmented['image']

        return image, class_name



class Dataloader:
    def __init__(self, batch_size: int, shuffle: bool, size: int, subset: int = 0, description: bool = False) -> None:
        # construtor do dataloader
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._size = size
        self._img_dir = 'data'
        self._subset = subset
        self._transform = self.compose()
        self._description = description

    def compose(self, p: float = 0.5):
        # retornar o compose
        # p = self._probability_aug[split] -----
        transform_list_train = A.Compose([
            A.RandomCrop(height=self._size, width=self._size),
            A.CLAHE(p=p),
            ToTensorV2()
        ])

        transform_list_test = A.Compose([
            A.RandomCrop(height=self._size, width=self._size),
            A.CLAHE(p=p),
            ToTensorV2()
        ])

        #if self._subset > 0:
        #    transform_list.append(A.RandomResizedCrop(height=self._subset, width=self._subset)) 

        return {'train': transform_list_train,
                'test': transform_list_test}

    def get_dataloader(self, split: str) -> DataLoader:
        # retornar o dataloader baseado no split
        dataset = Data(image_dir=self._img_dir, split=split, transform=self._transform[split])
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle)

        if self._description:
            print(f'{split} Dataloader: {len(dataloader)} batches: {self._batch_size}')

        return dataloader

    def get_train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')
    #def get_val_dataloader(self) -> DataLoader: return self.get_dataloader('val')
    #def get_test_dataloader(self) -> DataLoader: return self.get_dataloader('test')