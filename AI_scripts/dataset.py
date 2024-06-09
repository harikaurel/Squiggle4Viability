import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
import time

class SingleFileDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, label):
        self.data = torch.load(data_file)
        self.label = torch.tensor([label] * len(self.data))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]
        return X, y