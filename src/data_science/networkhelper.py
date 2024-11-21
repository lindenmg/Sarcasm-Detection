import sys

import torch
from torchtext.vocab import Vocab

import src.tools.helpers as helpers
from src.preprocessing.datahandler import DataHandler


class NetworkHelper:
    """
    This class contains methods that return complex Neural Network
    layers or objects that help to build them or for training and
    validation loops.
    """

    @staticmethod
    def __better_get_vectors(word_idx, vector_list, itos):
        """
        This is a better version of the torchtext.Vocab.set_vectors method

        Better suited for our needs here.
        Use base_model.DataHandler.load_word_vectors to load from a file on disk
        beforehand.
        Words that are not in the word-vector file won't get stored!

        Parameters.
        ----------
        word_idx: dict
            Should return the index of the vector in vector_list
            for token, if you use word_list[token]
        vector_list: list of torch.FloatTensor
            The vectors for the words. All should have the same length.
            Works with references! So you shouldn't change the values afterward
        itos: list
            A list of the words in the Vocabulary.
            torchtext.Vocab.itos should be used for this.

        Returns
        -------
        torch.Tensor, list

        See Also
        --------
        https://github.com/pytorch/text/blob/master/torchtext/vocab.py
        """

        # CHECK input (sanity checks)
        shapes = [t.shape for t in vector_list]
        dims = [t.dim() for t in vector_list]
        ones = [1] * len(vector_list)
        equal_dims = ([vector_list[0].shape] * len(vector_list) == shapes)
        one_dimensional = (ones == dims)

        if not equal_dims:
            raise ValueError("FloatTensors in vector_list should "
                             "all have the same shape!")
        if not one_dimensional:
            raise ValueError("FloatTensors in vector_list should have "
                             "only one dimension (.dim() == 1)!")

        # DELETE words that are not in the word_idx from itos & stoi
        # Leads to a smaller look-up table ==> Less memory & slightly faster
        word_idx_get = word_idx.get
        itos_new = ['<pad>']
        for i in range(1, len(itos)):
            word = itos[i]
            word_vec_idx = word_idx_get(word, None)

            if word_vec_idx is not None:
                itos_new.append(word)

        # FILL word vector table
        vec_dim = vector_list[0].shape[0]
        vectors = torch.FloatTensor(len(itos_new), vec_dim).zero_()
        for i in range(1, len(itos_new)):
            word_vec_idx = word_idx[itos_new[i]]
            vectors[i] = vector_list[word_vec_idx]

        return vectors, itos_new

    @staticmethod
    def create_tt_vocab_obj(counter, word_idx, vectors, max_size=None, min_freq=5):
        """
        Creates a torchtext Vocab object from a list of words and FloatTensors

        Parameters
        ----------
        counter: collections.Counter
             Holding the frequencies of each value found in the data
        word_idx: dict
            Should return the index of the vector in vector_list
            for token, if you use word_list[token]
        vectors: list of torch.FloatTensor
            The vectors for the words. All should have the same length
        max_size: int
            Maximum size of the vocabulary in terms of number of word types
        min_freq: int
            The minimum count number of a word in the counter object.
            If it is below that it will be excluded

        Returns
        -------
        torchtext.Vocab
        """
        vocab = Vocab(counter, max_size=max_size, min_freq=min_freq)
        vectors, itos = NetworkHelper.__better_get_vectors(word_idx, vectors, vocab.itos)
        vocab.itos = itos
        vocab.stoi = helpers.idx_lookup_from_list(vocab.itos, default_dict=True)
        vocab.vectors = vectors
        return vocab

    @staticmethod
    def convert_str_to_emb_idx(example_list, word_idx, max_ex_len=sys.maxsize):
        """
        Converts nested list of str to a list of tensors of the matching embedding indices

        This method is meant to be used as helper function in the
        process to prepare the data for the PyTorch embedding layer.
        Take also a look at create_tt_vocab_obj to get a mapping from
        string to index and the word vectors

        Parameters
        ----------
        example_list: list
            A list of list of string tokens
        word_idx: dict
            A mapping from word type (str) to the embedding index
            torchtext.Vocab().stoi should be used for that.
            If you have the nested lists already padded, the padding
            value should be '<pad>'!
        max_ex_len: int
            The maximum amount of tokens in the examples/inner lists that
            should be kept. In other words: How long the individual data
            examples are allowed to be. Everything longer is cut off.

        Returns
        -------
        list of torch.FloatTensor
            List with the tensors of the former inner lists
        """
        emb_idx_list = [[word_idx[w]
                         for i, w in enumerate(example)
                         if i < max_ex_len
                         ]
                        for example in example_list]
        return DataHandler.conv_inner_to_tensor(emb_idx_list
                                                , tensor_type=torch.LongTensor)
