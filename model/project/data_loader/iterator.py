import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset
import random

class StringDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)**2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        first_id = idx%len(self.data)
        second_id = idx//len(self.data)
        fist_set, second_set = random.choice(self.data[first_id]), random.choice(self.data[second_id])
        if len(fist_set)==0 or len(second_set)==0:
            print(fist_set), print(second_set)
        if self.transform:
            fist_set = self.transform(fist_set)
            second_set = self.transform(second_set)
        sample  = {
            'first_text': torch.FloatTensor(fist_set), 
            'second_text': torch.FloatTensor(second_set), 
            'loss_mul': 1. + float(first_id==second_id)*(len(self.data)-1), 
            'label': float(first_id==second_id)
        }

        return sample