from collections import Counter, defaultdict

import numpy as np
import torch

from src.preprocessing.datahandler import DataHandler
from src.tools.helpers import idx_lookup_from_list, flatten
from src.data_science.networkhelper import NetworkHelper
from src.preprocessing.preprocessing import Preprocessing


class TestNetworkHelper:
    def test_create_tt_vocab_obj(self):
        datahandle = DataHandler()
        pp = Preprocessing()
        nh = NetworkHelper()
        vectors = np.random.normal(0, 0.5, (10001, 300))
        vectors = vectors.round(7)
        words = np.random.randint(10000, 11000, (10001,))
        words = words.astype(str).tolist()
        words = [[w] for w in words]
        filtered_words, _ = pp.filter_by_frequency(words, min_freq=7)
        filtered_words = list(set(flatten(filtered_words)))
        words = flatten(words)
        counter = Counter(words)
        words = list(set(words))
        word_idx = idx_lookup_from_list(words)
        vectors = datahandle.conv_inner_to_tensor(vectors)
        vocab = nh.create_tt_vocab_obj(counter, word_idx, vectors, min_freq=7)
        vocab_vectors = vocab.vectors
        vocab_idx = vocab.stoi
        for word in filtered_words:
            own_vector = vectors[word_idx[word]]
            voc_vector = vocab_vectors[vocab_idx[word]]
            assert torch.equal(own_vector, voc_vector)

    def test_convert_str_to_emb_idx(self):
        example_list = [["Hallo", "ich", "bin", "ein", "Informatiker"]
            , ["Jetzt", "ist", "gerade", "Nacht"]
            , [], ["Guten", "Tag"]]
        word_idx = defaultdict(lambda: 0)
        word_idx["Hallo"] = 1
        word_idx["ich"] = 2
        word_idx["bin"] = 3
        word_idx["Informatiker"] = 4
        word_idx["Jetzt"] = 5
        word_idx["ist"] = 7
        word_idx["gerade"] = 6
        word_idx["Tag"] = 8
        emd_idx_list = [[1, 2, 3, 0, 4], [5, 7, 6, 0], [], [0, 8]]
        for i, l in enumerate(emd_idx_list):
            emd_idx_list[i] = torch.LongTensor(l)
        t_list = NetworkHelper.convert_str_to_emb_idx(example_list, word_idx)
        for l, t in zip(emd_idx_list, t_list):
            assert (l == t).all()

        emd_idx_list = [[1, 2, 3], [5, 7, 6], [], [0, 8]]
        for i, l in enumerate(emd_idx_list):
            emd_idx_list[i] = torch.LongTensor(l)
        t_list = NetworkHelper.convert_str_to_emb_idx(example_list
                                                      , word_idx, max_ex_len=3)
        for l, t in zip(emd_idx_list, t_list):
            assert (l == t).all()
