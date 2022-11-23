import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class Lung_Dataset(Dataset):
    def __init__(self, csv_root):
        self.csv = pd.read_csv(csv_root)

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.csv.loc[idx, 'img']).convert('L'))
        mask = np.asarray(Image.open(self.csv.loc[idx, 'mask']).convert('L'))
        filename = self.csv.loc[idx, 'img']

        trans = transforms.Compose([transforms.ToTensor(),
                                    ])
        mask = trans(mask)
        image = trans(image)

        mask = TF.resize(mask, 224)
        image = TF.resize(image, 224)

        return {'image': image.float(),
                'mask': mask.float(),
                'filename': filename}

    def __len__(self):
        return self.csv.shape[0]


def train_val_dataloader(df='/subset/df.csv'):
    dataset = Lung_Dataset(df)

    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    data_loader_train = torch.utils.data.DataLoader(
     train_dataset, batch_size=32, shuffle=True)

    data_loader_val = torch.utils.data.DataLoader(
     val_dataset, batch_size=64, shuffle=False)

    return data_loader_train, data_loader_val
