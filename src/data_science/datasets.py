# This file contains a collection of classes subclassing torch.utils.data.Dataset

import torch
from torch import FloatTensor
from torch.utils.data import Dataset


class EmbeddingDataSet(Dataset):

    def __init__(self, posts, replies, labels):
        """
        Supposed to be used for text data converted to the embedding indices of the words

        All data examples should have retained their original order.
        Meaning posts[0] fits replies[0] and labels[0] ...

        Parameters
        ----------
        posts: list of torch.LongTensor or torch.LongTensor
            Contains list of embedding indices for the words of the posts
        replies: list of torch.LongTensor or torch.LongTensor
        labels: list of torch.LongTensor or torch.LongTensor
        """
        self.posts = posts
        self.replies = replies
        self.labels = labels
        self.length = len(replies)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return ({
                    'posts': self.posts[idx // 2]
                    , 'replies': self.replies[idx]
                }, self.labels[idx])


class TfIdfDataSet(Dataset):

    def __init__(self, posts, replies, labels):
        """
        Supposed to be used for tf-idf vector data.

        All data examples should have retained their original order.
        Meaning posts[0] fits replies[0] and labels[0] ...

        Parameters
        ----------
        posts: scipy.sparse.csr.csr_matrix
            A matrix containing the tf-idf vectors of the posts.
            Unique posts. Shape: E x V - E are the examples and
            V the tf-idf vectors
        replies: scipy.sparse.csr.csr_matrix
            A matrix containing the tf-idf vectors of the posts.
            Shape: E x V - E are the examples and V the tf-idf vectors
        labels: torch.LongTensor
            Containing the labels (classes) in for posts and replies
        """
        self.replies = replies
        self.length = replies.shape[0]
        self.posts = posts
        self.labels = labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return ({
                    'posts': FloatTensor(self.posts[idx // 2].todense()).squeeze_()
                    , 'replies': FloatTensor(self.replies[idx].todense()).squeeze_()
                }, self.labels[idx])
