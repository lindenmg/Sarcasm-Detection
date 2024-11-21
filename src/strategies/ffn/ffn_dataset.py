import torch
from sklearn.datasets import make_classification
from torch.utils.data.dataset import Dataset


class FFNDataset(Dataset):
    def __init__(self, posts, replies, labels):
        self.replies = replies
        self.posts = posts
        self.labels = labels

    def __len__(self):
        return self.replies.shape[0]

    # ==> Problem: Just creating a tensor with torch.FloatTensor([posts, replies, labels]) does not work  !!!
    def __getitem__(self, idx):
        return ({
                    'posts': torch.FloatTensor(self.posts[idx // 2,].todense()).squeeze_(),
                    'replies': torch.FloatTensor(self.replies[idx,].todense()).squeeze_()},

                # That should be already a list of floats
                self.labels[idx,])


class FFNRandomDataset(Dataset):
    def __init__(self, vector_size=30, n_samples=10):
        self.replies, self.labels = make_classification(n_samples, vector_size)
        self.posts, _ = make_classification(n_samples // 2, vector_size)
        # self.replies = self.replies.reshape(n_samples, 1, vector_size)
        # self.posts = self.posts.reshape(n_samples // 2, 1, vector_size)

        pass

    def __len__(self):
        return self.replies.shape[0]

    def __getitem__(self, idx):
        return ({
                    'posts': torch.FloatTensor(self.posts[idx // 2,]),
                    'replies': torch.FloatTensor(self.replies[idx,])},

                # That should be already a list of floats
                self.labels[idx,])
