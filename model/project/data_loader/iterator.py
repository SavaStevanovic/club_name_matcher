import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset
import random

class StringDataset(Dataset):
    def __init__(self, data, transform=None, training=True):
        self.transform = transform
        self.data = data
        self.training = training

    def __len__(self):
        return len(self.data)**2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        first_id = idx%len(self.data)
        second_id = idx//len(self.data)
        if self.training and 0.5>random.random():
            second_id = first_id
        fist_set, second_set = self.data[first_id], self.data[second_id]
        fist_text, second_text = random.choice(fist_set), random.choice(second_set)
        if len(fist_text)==0 or len(second_text)==0:
            print(fist_text), print(second_text)
        if self.transform:
            fist_text = self.transform(fist_text)
            second_text = self.transform(second_text)
        sample  = {
            'first_text': torch.FloatTensor(fist_text), 
            'second_text': torch.FloatTensor(second_text), 
            'loss_mul': 1., 
            'label': float(first_id==second_id)
        }

        return sample