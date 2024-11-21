from __future__ import unicode_literals

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchtext.vocab import Vocab

import src.tools.helpers as helpers
from src.data_science.networkhelper import NetworkHelper
from src.preprocessing.datahandler import DataHandler
from src.preprocessing.preprocessing import Preprocessing
from src.tools.config import Config


class CnnPreprocessing:

    def __init__(self, sw_full_file, sw_cut_file, model='en'
                 , pipeline_to_remove=list(['textcat'])):
        """
        Supplies methods for the pre-processing for CNN models

        Build up on the assumption, that word vectors and an
        embedding layer are used.
        No own save/load mechanics to store already processed data on disk!

        Parameters
        ----------
        sw_full_file: str
            Name of file with the full stop word list
        sw_cut_file: str
            Name of file with the cut stop word list.
            Cut dependent on the occurrences in the sarcastic
            & serious replies.
        model: str
            The name of the model which SpaCy should load.
            You should have downloaded it beforehand.
            As of SpaCy 2.0 that would be 'en', 'en_core_web_sm',
            'en_core_web_md' and 'en_core_web_lg'
        pipeline_to_remove: list of str
            The names of the pipeline steps to disable.
            Keep the dependencies of the parts in mind!

        Examples
        --------
        >>> sw_cut = 'stop_words_cut_ultra.txt'
        >>> sw_full = 'stop_words_full_ultra.txt'
        >>> pp = CnnPreprocessing(sw_full, sw_cut)
        >>> test, train = pp.load_train_test()
        >>> train_post, train_reply = pp.apply_spacy_pipeline(train)
        >>> test_post, test_reply = pp.apply_spacy_pipeline(test)
        >>> train_post, train_reply = pp.apply_token_to_x(train_post, train_reply)
        >>> test_post, test_reply = pp.apply_token_to_x(test_post, test_reply)
        >>> counter = pp.create_word_counter(train_post, test_post
        >>>                                  , train_reply, test_reply)
        >>> word_vec_file = 'data/word_vectors/fastText/ft_2M_300.csv'
        >>> vocab = pp.create_vocab(word_vec_file, counter)
        >>> train_labels, test_labels = pp.get_labels(train, test)
        >>> word_idx = vocab.stoi
        >>> train_post, train_reply = pp.conv_str_to_emb_idx(train_post, train_reply
        >>>                                                  , word_idx)
        >>> train_post = helpers.pad_tensor(train_post, data_dim=30)
        ...
        Split data for k-fold cross validation ...
        """
        self._pp = Preprocessing(model_type=model, pipeline_to_remove=pipeline_to_remove)
        self._nlp = self._pp.get_nlp()
        self._nh = NetworkHelper()
        self._dh = DataHandler()
        self.sw_full_path = Config.data_path(sw_full_file)
        self.sw_cut_path = Config.data_path(sw_cut_file)

    # ToDo: Refactor this (and methods it uses probably too)!
    def load_train_test(self, test_set='test.csv'):
        """
        Loads the train & test set of the SARC data-set

        Returns
        -------
        pandas.DataFrame, pandas.DataFrame
        """
        data_dir = Path(Config.path.data_folder)
        train_file = data_dir / "train.csv"
        test_file = data_dir / test_set
        if test_set != "test.csv":
            dtype = {'post_id': np.str, 'post': np.str,
                     'reply': np.str, 'sarcasm': np.int8}
            test = pd.read_csv(test_file, sep='\t', keep_default_na=False
                               , na_values="", dtype=dtype)
            self._dh.load_train_test(str(data_dir))
        elif not (train_file.is_file() or test_file.is_file()):
            comments = str(data_dir / "comments_cleaned.txt")
            annotation = str(data_dir / "annotation.txt")
            self._dh.load_data(comments, annotation)
            self._dh.split_in_train_test()
            self._dh.save_train_test_to_csv(str(data_dir))
            test = self._dh.get_test_df(deep_copy=False)
        else:
            self._dh.load_train_test(str(data_dir))
            test = self._dh.get_test_df(deep_copy=False)
        train = self._dh.get_train_df(deep_copy=False)
        return test, train

    def apply_spacy_pipeline(self, df, num_replies=None):
        """
        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame with the combined SARC data
        num_replies: int
            The number of replies to process by SpaCy.
            Use ``None`` for all replies
        Returns
        -------
        list of spacy.Docs, list of spacy.Docs
        """
        n_threads = Config.hardware.n_cpu
        posts = self._pp.run_spacy_pipeline(df['post'][:num_replies][0::2], n_threads=n_threads)
        replies = self._pp.run_spacy_pipeline(df['reply'][:num_replies], n_threads=n_threads)
        return posts, replies

    def apply_token_to_x(self, posts, replies, no_stop_words=False, no_punctuation=False
                         , transform_specials=True, token_kind='lower_'):
        """
        Filters and transforms the SpaCy tokens

        Parameters
        ----------
        posts: list-like of spacy.tokens.doc.Doc
            The SpaCy docs of the posts to convert to strings.
            Does not get overwritten.
        replies: list-like of spacy.tokens.doc.Doc
            The SpaCy docs of the replies to convert to strings.
            Does not get overwritten.
        no_stop_words: bool
            True if you want to filter stop words
        no_punctuation: bool
            True, if you want to filter punctuation
        transform_specials: bool
            True, if you want to transform special entities to their hypernym.
            See Preprocessing.convert_token_docs_text
        token_kind: str
            The name of the attribute of the parsed token you want to have.
            Options: 'text', 'lemma_', 'lower_', ... everything for which
            Token returns a string. Should be written correctly!
        Returns
        -------
        list of list, list of list
            posts tokens, reply tokens
        """
        self._nlp.add_stop_word_def(self.sw_full_path)
        post_docs = self._pp.filter_spacy_tokens(posts, no_stop_words=no_stop_words
                                                 , no_punctuation=no_punctuation)
        post_pcd = self._pp.convert_token_docs_text(post_docs, token_kind=token_kind
                                                    , transform_specials=transform_specials)
        self._nlp.add_stop_word_def(self.sw_cut_path)
        reply_docs = self._pp.filter_spacy_tokens(replies, no_stop_words=no_stop_words
                                                  , no_punctuation=no_punctuation)
        reply_pcd = self._pp.convert_token_docs_text(reply_docs, token_kind=token_kind
                                                     , transform_specials=transform_specials)
        return post_pcd, reply_pcd

    def conv_str_to_emb_idx(self, posts, replies, word_idx, max_ex_len=50):
        """
        Parameters
        ----------
        posts: list of list
            Contains the filtered & processed words of the posts
        replies: list of list
            Contains the filtered & processed words of the replies
        word_idx: dict
            A mapping from word type (str) to the embedding index
            torchtext.Vocab().stoi should be used for that.
            If you have the nested lists already padded, the padding
            value should be '<pad>'!
        max_ex_len: int
            The maximum amount of tokens in the examples/inner lists that
            should be kept. In other words: How long the individual data
            examples are allowed to be. Everything loner is cut off.
        Returns
        -------
        list of torch.LongTensor, list of torch.LongTensor
            post, reply
            Lists which contain the embedding indices in tensor form
        """
        post_emb = self._nh.convert_str_to_emb_idx(posts, word_idx, max_ex_len)
        reply_emb = self._nh.convert_str_to_emb_idx(replies, word_idx, max_ex_len)
        return post_emb, reply_emb

    @staticmethod
    def get_labels(train, test, num_train_replies=None, num_test_replies=None):
        """
        Parameters
        ----------
        train: pandas.DataFrame
        test: pandas.DataFrame
        num_train_replies: int
        num_test_replies: int

        Returns
        -------
        torch.LongTensor, torch.LongTensor
            The labels for train & test dataset, in this order
        """
        train = train['sarcasm'][:num_train_replies]
        test = test['sarcasm'][:num_test_replies]
        train = train.values.astype(dtype=np.long, copy=False)
        test = test.values.astype(dtype=np.long, copy=False)
        train = torch.from_numpy(train)
        test = torch.from_numpy(test)
        return train, test

    @staticmethod
    def create_word_counter(post_train, post_test, reply_train, reply_test):
        """
        Parameters
        ----------
        post_train: list of list
        post_test: list of list
        reply_train: list of list
        reply_test: list of list

        Returns
        -------
        Counter
            The counts of the individual word types in the processed dataset
        """
        complete_tokens = post_test + reply_test + post_train + reply_train
        complete_tokens = helpers.flatten(complete_tokens)
        return Counter(complete_tokens)

    def create_vocab(self, word_vector_path, counter, word_vec_count=1999995
                     , word_vec_dim=300, ft_format=False, min_freq=1, max_vocab_size=None):
        """
        Creates an object which contains a look-up table with word-vectors

        Parameters
        ----------
        word_vector_path: str
            The path to the word vector file which should be used
        counter: Counter
            Object which contains the counts of the word types in the data
        word_vec_count: int
            The number of word vectors in the word vector file
        word_vec_dim: int
            The dimensionality of the word vectors
            (the different numbers in each line in the file)
        ft_format: bool
            True, if the word vector file is of the fastText format.
            Which means it has one header row with the count and dim
            of the word vectors in the file
        min_freq: int
            The minimum frequency of each word type in the data.
            If it is below this threshold it gets filtered out.
            Because this method returns a word vector look-up table,
            the default value of 1 is recommended.
        max_vocab_size: int
            The maximum number of word-vectors in the look-up table.
            Starts counting with the words of highest frequency,
            declines from there. If the accumulated words reach this
            threshold, all other words are filtered out.

        Returns
        -------
        Vocab
        """
        word_list, vectors = self._dh.load_word_vectors(word_vector_path
                                                        , word_vec_count
                                                        , word_vec_dim
                                                        , ft_format)
        word_idx = helpers.idx_lookup_from_list(word_list, default_dict=False)
        vector_t = self._dh.conv_inner_to_tensor(vectors)
        vocab = self._nh.create_tt_vocab_obj(counter, word_idx, vector_t
                                             , max_size=max_vocab_size, min_freq=min_freq)
        assert len(vocab.itos) == len(vocab.vectors)
        assert len(vocab.itos) <= 1 + len({w for w in counter if counter[w] >= 1})
        return vocab
