import numpy as np
import pytest
import torch

from src.data_science.samplers import BucketRandomSampler, LazyBatchSampler

message = "a way that it gives you not even one complete batch " \
          "and you have set ``drop_last` to True, therefore you " \
          "would only get an empty batch per epoch!"


def create_brs_i(data_length, batch_size, eq_wdw, data_size):
    data = torch.rand(data_length, data_size)
    data_dims = torch.LongTensor(np.full(data_length, data_size))
    indices = np.arange(data_length, dtype=int)
    br_sampler = BucketRandomSampler(data, data_dims, batch_size
                                     , batch_search_window=eq_wdw, cuda=False)
    return br_sampler, indices, data_dims


def create_brs_ii(data_length, batch_size, eq_wdw):
    data_dims = []
    randint = np.random.randint
    for d in range(data_length):
        size = randint(0, 128)
        data_dims.append(size)
    data_dims = torch.LongTensor(data_dims)
    indices = np.arange(data_length, dtype=int)
    br_sampler = BucketRandomSampler(data_dims, data_dims, batch_size
                                     , batch_search_window=eq_wdw, cuda=False)
    return br_sampler, indices, data_dims


class TestBucketRandomSampler:

    def test__iter__i(self):
        randint = np.random.randint
        iters = 50
        data_length = randint(1000, 100000, (iters,)).tolist()
        batch_size = randint(256, 2048, (iters,)).tolist()
        eq_wdw = randint(2, 10, (iters,)).tolist()
        data_size = randint(5, 50, (iters,)).tolist()
        for l, b, w, s in zip(data_length, batch_size, eq_wdw, data_size):
            br_sampler, indices, _ = create_brs_i(l, b, w, s)
            uncomplete_batches = 1 if (l % b != 0) else 0
            b_len_l = [(b == batch.size()[0]) for batch in br_sampler]
            b_len_l = np.asarray(b_len_l)
            batch_list = [batch for batch in br_sampler]
            batch_list = torch.cat(batch_list, dim=0)
            batch_list = batch_list.sort(dim=0)[0]
            batch_list = batch_list.numpy().astype(int, copy=False)
            assert (indices == batch_list).all()
            assert (b_len_l == False).sum() == uncomplete_batches

    def test__iter__ii(self):
        randint = np.random.randint
        iters = 50
        data_length = randint(1009, 100004, (iters,)).tolist()
        batch_size = randint(251, 2054, (iters,)).tolist()
        eq_wdw = randint(2, 12, (iters,)).tolist()
        for l, b, w in zip(data_length, batch_size, eq_wdw):
            br_sampler, _, data_dims = create_brs_ii(l, b, w)
            batch_list = [batch for batch in br_sampler]
            batch_list = torch.cat(batch_list, dim=0)
            batch_list = batch_list.numpy().astype(int, copy=False)
            data_dims = data_dims.sort(dim=0)[0]
            data_dims = data_dims.numpy().astype(int, copy=False)
            assert (data_dims != batch_list).any()
            assert len(br_sampler) == l


class TestLazyBatchSampler:

    def test__iter__i(self):
        randint = np.random.randint
        iters = 50
        data_length = randint(1009, 100004, (iters,)).tolist()
        batch_size = randint(251, 2054, (iters,)).tolist()
        eq_wdw = randint(2, 12, (iters,)).tolist()
        droplast = randint(0, 2, iters)
        droplast = [(d == 1) for d in droplast]
        for l, b, w, d in zip(data_length, batch_size, eq_wdw, droplast):
            left_over = (l % b != 0 and not d)
            bs_length = (l // b + (l % b != 0))
            bs_length = (l // b) if d else bs_length
            uncomplete_batches = 1 if left_over else 0
            br_sampler, indices, _ = create_brs_ii(l, b, w)
            if bs_length == 0:
                with pytest.raises(ValueError) as val_err:
                    _ = LazyBatchSampler(br_sampler, b, d)
                assert message in str(val_err.value)
                continue
            else:
                batcher = LazyBatchSampler(br_sampler, b, d)
            b_len_l = [(b == batch.size()[0]) for batch in batcher]
            b_len_l = np.asarray(b_len_l)
            batch_list = [batch for batch in batcher]
            unique_idx_len = len(set(batch_list))
            batch_list_len = len(batch_list)
            batch_list = torch.cat(batch_list, dim=0)
            batch_list = batch_list.sort(dim=0)[0]
            batch_list = batch_list.numpy().astype(int, copy=False)
            assert np.in1d(batch_list, indices, assume_unique=True).all()
            assert (b_len_l == False).sum() == uncomplete_batches
            assert batch_list_len == bs_length
            assert unique_idx_len == batch_list_len
            assert len(batcher) == bs_length

    def test__iter__ii(self):
        randint = np.random.randint
        iters = 100
        batch_size = randint(13, 2054, (iters,))
        eq_wdw = randint(2, 18, (iters,))
        data_length = batch_size * eq_wdw
        batch_size = batch_size.tolist()
        eq_wdw = eq_wdw.tolist()
        data_length = data_length.tolist()
        for l, b, w in zip(data_length, batch_size, eq_wdw):
            assert l == b * w
            for i in range(2):
                br_sampler, indices, data_dims = create_brs_ii(l, b, w)
                batcher = LazyBatchSampler(br_sampler, b, (i == 1))
                indices = torch.LongTensor(indices)
                indices = indices[data_dims.sort(dim=0)[1]]
                goal_batches = indices.split(b)
                yield_batches = [batch for batch in batcher]
                equal_list = []
                for yb in yield_batches:
                    for gb in goal_batches:
                        if (data_dims[yb] == data_dims[gb]).all():
                            equal_list.append(True)
                assert len(equal_list) == w
