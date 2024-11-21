import torch
from torch.utils.data.dataset import Dataset


class DummyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return (
            {'data': torch.FloatTensor(self.data[item,])},
            self.labels[item,]
        )
