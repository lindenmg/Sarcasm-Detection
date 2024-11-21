from __future__ import unicode_literals

import os
from copy import deepcopy
from pathlib import Path

import src.tools.helpers as helpers
from src.data_science.cnn_preprocessing import CnnPreprocessing
from src.tools.config import Config
from src.training.abstract_data_factory import AbstractDataFactory


class CnnDataFactory(AbstractDataFactory):

    def __init__(self, **args):
        """
        Factory that applies pre-processing for the CNN models

        Build up on the assumption, that word vectors and an
        embedding layer are used.
        Be aware of the implications of minimum word count and
        max vocab size in case of different preprocessing runs
        with the same training set & parameters but different test sets!

        Parameters
        ----------
        **args: dict
            The init parameters as defined in ``AbstractDataFactory.__init__``
        """
        super().__init__(**args)
        self._pp = CnnPreprocessing(self.pp_params['sw_full_filename']
                                    , self.pp_params['sw_cut_filename']
                                    , self.pp_params['tokenization']['spacy_model'])

    def get_data(self):
        """
        Provides the pre-processing to create embedding indices for post & reply.

        Due to the way it works it will always preprocess and return test & train sets

        Returns
        -------
        dict
            The dict shall fulfill the following requirements:
            * Key 'test_data' which returns dict with data for testing
            * Key 'train_data' which returns dict with data for training
            * Key 'word_vectors' which returns torch.FloatTensor
              for pretrained embedding layer
            * Key 'reply_lengths' which returns torch.LongTensor according to
              helpers.create_length_tensor for the replies of the training set
              This is for the initialization of a data_science.samplers.BucketRandomSampler
            * Key 'embedding_size' which returns the (int) amount of word-vectors
              in the value of the key 'word_vectors'.
        """
        if Config.debug_mode:
            helpers.section_text("BEGIN preprocessing for CNN")
        test, train = self._pp.load_train_test(self.pp_params["test_file"])

        # Because we give the ability to use different test sets - e.g. survey -
        # we have to create a tag from the specific name for correct caching
        test_tag = os.path.splitext(self.pp_params["test_file"])[0]
        train_tag = 'train'

        if Config.debug_mode:
            helpers.section_text("Run the Spacy pipeline")
        conv_tokens_params = self.combine_params(['conversion', 'filter'])
        tokens_train_step = self.create_step_name('token_conversion', train_tag)
        tokens_test_step = self.create_step_name('token_conversion', test_tag)
        tokens_train = self.eventually_load_cache(conv_tokens_params, tokens_train_step)
        tokens_test = self.eventually_load_cache(conv_tokens_params, tokens_test_step)
        num_train_replies = self.pp_params["raw_data"]["num_train_replies"]
        num_test_replies = self.pp_params["raw_data"]["num_test_replies"]

        if tokens_train is None:
            train_post, train_reply = self._apply_spacy_pipeline(train, num_train_replies, train_tag)
        if tokens_test is None:
            test_post, test_reply = self._apply_spacy_pipeline(test, num_test_replies, test_tag)

        if Config.debug_mode:
            helpers.section_text("Convert and filter the post & reply tokens")
        if tokens_train is None:
            tokens_train = self._convert_tokens(train_post, train_reply
                                                , tokens_train_step
                                                , conv_tokens_params)
        if tokens_test is None:
            tokens_test = self._convert_tokens(test_post, test_reply
                                               , tokens_test_step
                                               , conv_tokens_params)
        train_reply = tokens_train[1]
        train_post = tokens_train[0]
        test_reply = tokens_test[1]
        test_post = tokens_test[0]

        if Config.debug_mode:
            helpers.section_text("Create word vector look-up table")
        counter = self._pp.create_word_counter(train_post, test_post
                                               , train_reply, test_reply)

        # With only one steady train set this relies on the test set form
        # although considering the fact that we probably only use test data
        # from the given dataset a more complicated implementation would be
        # possible. One which saves some caching, but needs preprocessing with
        # the full original dataset first.
        vocab = self._create_vocab_obj(counter, 'all')

        if Config.debug_mode:
            helpers.section_text("Convert preprocessed tokens to embedding indices")
        word_idx = vocab.stoi
        train_post, train_reply = self._create_embedding_indices(train_post, train_reply
                                                                 , word_idx, train_tag)
        test_post, test_reply = self._create_embedding_indices(test_post, test_reply
                                                               , word_idx, test_tag)

        if Config.debug_mode:
            helpers.section_text("Create train & test tensors")
        data_dim = self.pp_params['tensor_data_dim']
        reply_lengths = helpers.create_length_tensor(train_reply)
        train_post = helpers.pad_tensor(list_of_tensors=train_post, data_dim=data_dim)
        train_reply = helpers.pad_tensor(list_of_tensors=train_reply, data_dim=data_dim)
        test_post = helpers.pad_tensor(list_of_tensors=test_post, data_dim=data_dim)
        test_reply = helpers.pad_tensor(list_of_tensors=test_reply, data_dim=data_dim)

        if Config.debug_mode:
            helpers.section_text("Create label tensors")
        train_labels, test_labels = self._pp.get_labels(train, test, num_train_replies
                                                        , num_test_replies)
        self._pp = None

        train_dict = {'posts': train_post, 'replies': train_reply, 'labels': train_labels}
        test_dict = {'posts': test_post, 'replies': test_reply, 'labels': test_labels}
        result_dict = {'train_data': train_dict, 'test_data': test_dict,
                       'word_vectors': vocab.vectors, 'reply_lengths': reply_lengths,
                       'embedding_size': len(vocab.vectors)}

        if Config.debug_mode:
            helpers.section_text("FINISH preprocessing for CNN")
        return result_dict

    def _apply_spacy_pipeline(self, dataframe, num_replies, tag):
        pp_step_params = self.combine_params(['tokenization', 'raw_data'])

        def __create_data():
            return self._pp.apply_spacy_pipeline(dataframe, num_replies)

        step_name = self.create_step_name('tokenization', tag)
        return self.create_data(step_name, pp_step_params, __create_data)

    def _convert_tokens(self, post_tokens, reply_tokens, step_name, pp_step_params):
        def __create_data():
            return self._pp.apply_token_to_x(post_tokens, reply_tokens, **pp_step_params)

        return self.create_data(step_name, pp_step_params, __create_data)

    def _create_vocab_obj(self, counter, tag):
        pp_step_params = self.combine_params(['vocab'])

        # We want no Python object in our caching name creation
        params = deepcopy(pp_step_params)
        params.update({"counter": counter})
        wv_path = Path(Config.path.data_folder) / params["word_vector_path"]
        params.update({"word_vector_path": str(wv_path)})

        def __create_data():
            return self._pp.create_vocab(**params)

        step_name = self.create_step_name('vocab', tag)
        return self.create_data(step_name, pp_step_params, __create_data)

    def _create_embedding_indices(self, post_tokens, reply_tokens, word_idx, tag):
        pp_step_params = self.combine_params(['embedding'])

        def __create_data():
            return self._pp.conv_str_to_emb_idx(post_tokens, reply_tokens
                                                , word_idx, **pp_step_params)

        step_name = self.create_step_name('embedding', tag)
        return self.create_data(step_name, pp_step_params, __create_data)
