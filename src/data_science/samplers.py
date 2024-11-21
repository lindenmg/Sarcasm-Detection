# This file contains a collection of Samples and Batch_Samplers for PyTorch DataLoaders

import torch
from torch import randperm
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler


class BucketRandomSampler(Sampler):

    def __init__(self, data_source, data_dims, batch_size
                 , batch_search_window=7, cuda=True):
        """
        Samples elements randomly, without replacement. Batches with similar data length

        First it creates a random permutation then sorts it within windows
        of length ``eq_search_window`` after length and permutes those windows again.
        Then the result is finally yielded.
        The use case is to get "texts" with similar lengths for Neural Networks -
        to have similar amounts of padding and therefore zeros in the end, per batch.
        This is sensible because of the resulting gradient - pertaining to input of 
        similar shape.

        Parameters
        ----------
        data_source: Dataset
        data_dims : torch.LongTensor
            Contains the length for each data example in ``data_source``
        batch_size: int
            The size of the batch in which the data is yielded.
        batch_search_window: int
            Multiplier of ``batch_size``
            The window in which is searched for examples of similar length
            Higher values lead to more equality but less randomness
        cuda: bool
            True if tensors should be stored on GPU,
            False else
        """
        self.length = len(data_source)
        if self.length != len(data_dims):
            raise ValueError("Lengths of data_source & data_dims are unequal!"
                             "{:} != {:}".format(self.length, len(data_dims)))

        self.data_dims = data_dims.unsqueeze(1)
        self.batch_size = batch_size
        self.window = batch_search_window * batch_size
        self.cuda = cuda
        if cuda:
            self.data_dims.cuda()

    def __iter__(self):
        """Operates with torch.Tensor and tuple"""
        permutation = randperm(self.length)
        if self.cuda:
            permutation.cuda()
        permuted_dims = self.data_dims[permutation]
        permutation = permutation.unsqueeze(1)
        data = torch.cat([permuted_dims, permutation], dim=1)
        windows = data.split(self.window)
        windows = [w[:, 1][w.sort(dim=0)[1][:, 0]] for w in windows]
        windows = torch.cat(windows, dim=0)
        windows = windows.split(self.batch_size)
        permutation = randperm(len(windows))
        for batch_idx in permutation:
            yield windows[batch_idx]

    def __len__(self):
        return self.length


class LazyBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, drop_last):
        """
        Wraps another BatchSampler to yield a mini-batch of indices.

        Parameters
        ----------
        sampler: BucketRandomSampler
            The sampler which already provides indices in batch-form
        batch_size: int
            The size of the individual mini-batch
        drop_last: bool
            If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        See Also
        --------
        https://pytorch.org/docs/0.3.0/_modules/torch/utils/data/sampler.html#BatchSampler
        """
        super().__init__(sampler, batch_size, drop_last)

        if not isinstance(sampler, BucketRandomSampler):
            raise TypeError("sampler should be an instance of BucketRandomSampler, "
                            "but is {}".format(type(sampler)))

        # If we can't even have one complete batch we will point that out
        if self.drop_last and len(sampler) // batch_size == 0:
            raise ValueError("The data amount and batch size has been chosen in "
                             "a way that it gives you not even one complete batch "
                             "and you have set ``drop_last` to True, therefore you "
                             "would only get an empty batch per epoch!")

        if self.drop_last:
            self.length = len(self.sampler) // self.batch_size
        else:
            self.length = (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        small_batch = None
        batch_size = self.batch_size
        for batch_ids in self.sampler:
            if len(batch_ids) < batch_size:
                small_batch = batch_ids
            else:
                yield batch_ids
        if not (small_batch is None or self.drop_last):
            yield small_batch

    def __len__(self):
        return self.length
