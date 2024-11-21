import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.strategies.dummy.dummy_dataset import DummyDataset


class DummyFFN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size_1)
        self.h2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.h3 = nn.Linear(hidden_size_2, output_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, data):
        x = self.relu1(self.h1(data))
        x = self.relu2(self.h2(x))
        x = F.log_softmax(self.h3(x), dim=-1)
        return x


# Little example
if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader
    from torch.autograd.variable import Variable

    d = np.random.rand(100).reshape(10, 10)
    labels = np.random.randint(0, 2, 10, dtype=np.int32)
    ds = DummyDataset(d, labels)
    dl = DataLoader(ds, batch_size=2)
    data, labels = next(iter(dl))
    v = {k: Variable(v) for k, v in data.items()}
    model = DummyFFN(10, 4, 3, 2)
    model(**v)
