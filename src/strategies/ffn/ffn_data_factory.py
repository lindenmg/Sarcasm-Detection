from functools import reduce

import numpy as np

from src.preprocessing.datahandler import DataHandler
from src.preprocessing.preprocessing import Preprocessing
from src.tools.config import Config
from src.tools.helpers import filter_dict
from src.training.abstract_data_factory import AbstractDataFactory


class FFNDataFactory(AbstractDataFactory):
    """
    Factory for generating preprocessing
    """

    def __init__(self, **args):
        """

        Parameters
        ----------
        **args: dict
            The init parameters as defined in ``AbstractDataFactory.__init__``
        """

        self._pp = None

        super().__init__(**args)
        self.cache_prefix = 'ffn_fact'

    def get_data(self):
        """
        Provides the preprocessing for generating tfidf values for sarcasm detection.
        Returns
        -------
        dict, dict
            First dict contains posts, replies, labels for training.
            Second dict contains posts, replies, labels, for testing
        """
        train_post = None
        train_reply = None
        train_labels = None
        test_post = None
        test_reply = None
        test_labels = None
        sw_full_path = Config.data_path(self.pp_params['sw_full_filename'])
        sw_cut_path = Config.data_path(self.pp_params['sw_cut_filename'])
        self._pp = Preprocessing(self.pp_params['tokenization']['spacy_model'])
        if self.train:
            self._pp.nlp.add_stop_word_def(sw_full_path)
            train_post = self._load_vectors('train_post')
            self._pp.nlp.add_stop_word_def(sw_cut_path)
            train_reply = self._load_vectors('train_reply')
            train_labels = self._load_labels('train')
        if self.test:
            self._pp.nlp.add_stop_word_def(sw_full_path)
            test_post = self._load_vectors('test_post')
            self._pp.nlp.add_stop_word_def(sw_cut_path)
            test_reply = self._load_vectors('test_reply')
            test_labels = self._load_labels('test')
        self._pp = None
        train_dict = {'posts': train_post, 'replies': train_reply, 'labels': train_labels}
        test_dict = {'posts': test_post, 'replies': test_reply, 'labels': test_labels}
        return {'train_data': train_dict, 'test_data': test_dict}

    def _load_raw_data(self, tag):
        pp_step_params = self.pp_params['raw_data']
        dh = DataHandler()
        dh.load_train_test(Config.path.data_folder)
        n_replies = pp_step_params['n_replies']
        self._console_output('raw_data', tag, pp_step_params)
        if n_replies is not None:
            n_replies = int(n_replies)
        if tag.find('test') >= 0:
            df = dh.get_test_df(deep_copy=False)
        else:
            df = dh.get_train_df(deep_copy=False)
        if tag.find('post') >= 0:
            return df['post'][:n_replies][0::2]
        else:
            return df['reply'][:n_replies]

    def _load_labels(self, tag):
        pp_step_params = self.pp_params['raw_data']
        self._console_output('load_labels', tag, pp_step_params)
        dh = DataHandler()
        dh.load_train_test(Config.path.data_folder)
        if tag.find('test') >= 0:
            df = dh.get_test_df(deep_copy=False)
        else:
            df = dh.get_train_df(deep_copy=False)
        return df['sarcasm'][:pp_step_params['n_replies']] \
            .values.astype(dtype=np.long, copy=False)

    def _load_tokenization(self, tag):
        pp_step_params = self.combine_params(['tokenization', 'raw_data'])

        def __create_data():
            docs = self._load_raw_data(tag)
            self._console_output('load tokenization', tag, pp_step_params)
            return self._pp.run_spacy_pipeline(docs)

        step_name = self.create_step_name('tokenization', tag)
        return self.create_data(step_name, pp_step_params, __create_data)

    # Unfortunately caching doesn't work here due to a pickle issue
    def _load_filter(self, tag):
        pp_step_params = self.combine_params(['filter', 'tokenization', 'raw_data'])

        def __create_data():
            tokens = self._load_tokenization(tag)
            self._console_output('load_filter', tag, pp_step_params)
            return self._pp.filter_spacy_tokens(
                tokens, **self.pp_params['filter'])

        step_name = self.create_step_name('filter', tag)
        return self.create_data(step_name, pp_step_params, __create_data, False)

    def _load_conversion(self, tag):
        pp_step_params = self.combine_params(['conversion', 'filter', 'tokenization', 'raw_data'])

        def __create_data():
            filtered = self._load_filter(tag)
            self._console_output('load conversion', tag, pp_step_params)
            return self._pp.convert_token_docs_text(
                filtered, concat_text_str=' ', **self.pp_params['conversion'])

        step_name = self.create_step_name('conversion', tag)
        return self.create_data(step_name, pp_step_params, __create_data)

    def _load_vocab(self, tag):
        self.pp_params['vocab'] = filter_dict(self.pp_params['vectorization'],
                                              ['analyzer', 'max_features', 'min_df', 'ngram_range'])
        pp_step_params = self.combine_params(
            ['vocab', 'conversion', 'filter', 'tokenization', 'raw_data'])

        def __create_data():
            vocab = [
                self._load_conversion(t)
                for t in ['train_post', 'train_reply', 'test_post', 'test_reply']]
            vocab = reduce(lambda acc, docs: acc + docs, vocab)
            self._console_output('load_vocab', tag, pp_step_params)
            return self._pp.str_list_to_vocab(vocab, **self.pp_params['vocab'])

        step_name = self.create_step_name('vocab')
        return self.create_data(step_name, pp_step_params, __create_data)

    def _load_vectors(self, tag):
        pp_step_params = self.combine_params(
            ['vectorization', 'conversion', 'filter', 'tokenization', 'raw_data'])

        def __create_data():
            l_str = self._load_conversion(tag)
            vocab = self._load_vocab(tag)
            self._console_output('load vectorization', tag, pp_step_params)
            if self.pp_params['vectorization']['tfidf']:
                return self._pp.str_list_to_vectors(l_str, vocab, **self.pp_params['vectorization'])
            else:
                return self._pp.str_list_to_vectors(l_str, vocab, **self.pp_params['vectorization'])

        step_name = self.create_step_name('vectorization', tag)
        return self.create_data(step_name, pp_step_params, __create_data)

    @staticmethod
    def _console_output(preprocessing_step, tag, pp_step_params):
        print('\n*** preprocessing step: %s' % preprocessing_step)
        print('tag: %s' % tag)
        print('parameters: ')
        print(pp_step_params)
