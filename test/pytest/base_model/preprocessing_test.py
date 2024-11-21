from src.preprocessing.datahandler import DataHandler
from src.preprocessing.preprocessing import Preprocessing
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import pandas as pd
import os
import spacy


def create_data_path():
    path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
    return os.path.join(path, 'test_data')


# for better readability of DataFrame output in the console
pd.options.display.max_colwidth = 0
pd.options.display.expand_frame_repr = False

# preprocessing in general
df_base = pd.DataFrame({'foo': [
    'this is really, really a document!',
    'this is another',
    'completely ... new']})
dh = DataHandler()
dh.load_train_test(create_data_path())
df_test = dh.get_test_df(deep_copy=False)
pp_test = Preprocessing(model_type='en')
nlp = pp_test.get_nlp()
nlp.add_stop_word_def(stop_words=STOP_WORDS)
piped = pp_test.run_spacy_pipeline(df_base['foo'])
replies = pp_test.run_spacy_pipeline(df_test['reply'][0:5000])
reply_filtered = pp_test.filter_spacy_tokens(replies)

# SOA tests
feats_docs = [["Hello", "I", "am", "a", "Computer", "Scientist"]
    , ["Hi", "nice", "to", "meet", "you", "I", "am", "a", "Dialog", "System"]
    , ["Can", "I", "help", "you", "with", "something"]
    , ["No", "thanks"]]
feats_labels = [0, 1, 1, 0]

# Not that this is about sarcasm, but the keys are hardcoded for simplicity
true_class_tokens = {'sarcasm':
                         [["Hi", "nice", "to", "meet", "you", "I", "am", "a", "Dialog", "System"]
                             , ["Can", "I", "help", "you", "with", "something"]]
    , 'serious': [["Hello", "I", "am", "a", "Computer", "Scientist"], ["No", "thanks"]]}


class TestCleaning:

    def test_str_list_to_vocab(self):
        df_sub = df_base.foo[:2]
        vocab_0 = pp_test.str_list_to_vocab(df_sub, min_df=1, ngram_range=(1, 2))
        vocab_1 = pp_test.str_list_to_vocab(df_sub, min_df=2, ngram_range=(1, 2))
        assert vocab_0 == {'document': 3, 'this': 10, 'this is': 11, 'a document': 1, 'really a': 8,
                           'really really': 9, 'really': 7, 'a': 0, 'another': 2, 'is another': 5,
                           'is': 4, 'is really': 6}
        assert vocab_1 == {'this': 1, 'this is': 2, 'is': 0}

    def test_str_list_to_vectors_i(self):
        df_sub = df_base.foo[:2]
        tfidf_vecs = pp_test.str_list_to_vectors(df_sub, min_df=1, ngram_range=(1, 1), ).todense()
        # 2 sentences
        # 6 types
        assert tfidf_vecs.shape == (2, 6)

        tfidf_vecs_2gram = pp_test.str_list_to_vectors(df_sub, min_df=1, ngram_range=(1, 2)).todense()
        # 2 sentences
        # 6 types + 6 2grams ((this is), (is really), (really really), (really a), (a document), (is another))
        assert tfidf_vecs_2gram.shape == (2, 12)

    def test_str_list_to_vectors_ii(self):
        df_sub = df_base.foo[:2]
        vocab_0 = pp_test.str_list_to_vocab(df_sub, min_df=2, ngram_range=(1, 2))
        vocab_1 = pp_test.str_list_to_vocab(df_sub, min_df=1, ngram_range=(1, 1))

        tfidf_vecs_0 = pp_test.str_list_to_vectors(df_sub, vocab_0, min_df=2, ngram_range=(1, 2)).todense()
        tfidf_vecs_1 = pp_test.str_list_to_vectors(df_sub, vocab_1, min_df=1, ngram_range=(1, 1)).todense()
        tfidf_vecs_2 = pp_test.str_list_to_vectors(df_sub, vocab_1, min_df=100, ngram_range=(1, 5)).todense()
        tfidf_vecs_3 = pp_test.str_list_to_vectors(df_sub, vocab_0, min_df=1, ngram_range=(1, 1)).todense()
        assert tfidf_vecs_0.shape == (2, 3)
        assert tfidf_vecs_1.shape == tfidf_vecs_2.shape == (2, 6)
        assert False not in (tfidf_vecs_1 == tfidf_vecs_2)
        assert False in (tfidf_vecs_0 == tfidf_vecs_3)
        assert tfidf_vecs_3[0, 2] == 0

    def test_run_spacy_pipeline(self):
        rand_0 = np.random.randint(0, len(piped))
        rand_1 = np.random.randint(0, len(piped[rand_0]))
        assert isinstance(piped[rand_0][rand_1], spacy.tokens.token.Token)

    def test_filter_spacy_tokens(self):
        filtered_tokens = pp_test.filter_spacy_tokens(piped, no_punctuation=True, no_stop_words=True)
        assert len(filtered_tokens[0]) == 1
        assert len(filtered_tokens[1]) == 0
        assert len(filtered_tokens[2]) == 2

    def test_filter_spacy_tokens_ii(self):
        filtered_tokens = pp_test.filter_spacy_tokens(piped, no_punctuation=False, no_stop_words=True)
        assert len(filtered_tokens[0]) == 3
        assert len(filtered_tokens[1]) == 0
        assert len(filtered_tokens[2]) == 3

    def test_filter_spacy_tokens_iii(self):
        filtered_tokens = pp_test.filter_spacy_tokens(piped, no_punctuation=False, no_stop_words=False)
        assert len(filtered_tokens[0]) == 8
        assert len(filtered_tokens[1]) == 3
        assert len(filtered_tokens[2]) == 3

    def test_filter_spacy_tokens_iv(self):
        filtered_tokens = pp_test.filter_spacy_tokens(piped, no_punctuation=False, no_stop_words=True)
        assert len(filtered_tokens[0]) == 3
        assert len(filtered_tokens[1]) == 0
        assert len(filtered_tokens[2]) == 3

    def test_convert_token_docs_text(self):
        lemmas = pp_test.convert_token_docs_text(piped, token_kind='lemma_')
        assert lemmas == [['this', 'be', 'really', ',', 'really', 'a', 'document', '!'],
                          ['this', 'be', 'another'],
                          ['completely', '...', 'new']]
        lemmas = pp_test.convert_token_docs_text(piped, token_kind='lemma_', concat_text_str=' ')
        assert lemmas == ['this be really , really a document !',
                          'this be another',
                          'completely ... new']

    def test_filter_by_frequency(self):
        lemmas = pp_test.convert_token_docs_text(piped, token_kind='lemma_')
        filtered, _ = pp_test.filter_by_frequency(lemmas, min_freq=2)
        assert filtered == [['this', 'be', 'really', 'really'], ['this', 'be'], []]

    @staticmethod
    def __assert_equal_filtering(tokens_main, mirror_list, attr='lemma_'):
        for doc, text_feats in zip(tokens_main, mirror_list[0]):
            for token, word in zip(doc, text_feats):
                assert getattr(token, attr) == word

    def test_filter_by_frequency_ii(self):
        text_docs = pp_test.convert_token_docs_text(reply_filtered, token_kind='lemma_')
        filtered, filtered_docs = pp_test.filter_by_frequency(reply_filtered, min_freq=5
                                                              , mirror_docs_list=[text_docs])
        self.__assert_equal_filtering(filtered, filtered_docs)
        reply_ner = pp_test.convert_token_docs_label(reply_filtered)
        filtered, mirrors = pp_test.filter_by_frequency(reply_filtered, min_freq=3
                                                        , mirror_docs_list=[text_docs, reply_ner])
        self.__assert_equal_filtering(filtered, mirrors[0])
        self.__assert_equal_filtering(filtered, mirrors[1], attr='ent_type')

    def test_inner_str_join_2d_list(self):
        lemmas = pp_test.convert_token_docs_text(piped, token_kind='lemma_')
        filtered, _ = pp_test.filter_by_frequency(lemmas, min_freq=2)
        joined = pp_test.inner_str_join_2d_list(filtered)
        assert joined == ['this be really really', 'this be', '']

    @staticmethod
    def __assert_norm_array(array, eps=1e-2):
        assert (1. - eps) <= array.var() <= (1. + eps)
        assert (0 - eps) <= array.mean() <= (0 + eps)

    def __test_feats_groups_2d(self, array, eps=1e-2):
        norm_ar = pp_test.normalize_np_array(array, feature_grouping=True)
        self.__assert_norm_array(norm_ar, eps)
        assert (norm_ar != array).all()

        for el in norm_ar:
            self.__assert_norm_array(el, eps)

    @staticmethod
    def __test_feats_shift_norm(array, eps=1e-8):
        norm_ar = pp_test.normalize_np_array(array, variance_of_one=False)
        assert (norm_ar != array).all()
        assert norm_ar.max() <= (1. + eps)
        assert norm_ar.min() >= (-1 - eps)

    def test_normalize_np_array_i(self):
        df_sub = df_base.foo[:2]
        tfidf_vecs = pp_test.str_list_to_vectors(df_sub, min_df=1, ngram_range=(1, 1))
        tfidf_vecs = tfidf_vecs.toarray().astype(dtype=np.float32)
        norm_ar = pp_test.normalize_np_array(tfidf_vecs)
        self.__assert_norm_array(norm_ar)

    def test_normalize_np_array_ii(self):
        array = np.random.uniform(size=(9987, 51))
        self.__test_feats_groups_2d(array)
        array = np.random.uniform(-1., 1., size=(8000, 10))
        self.__test_feats_groups_2d(array)
        array = np.random.uniform(-17., -3., size=(311, 9323))
        self.__test_feats_groups_2d(array)

    def test_normalize_np_array_iii(self):
        array = np.random.normal(13.573, 4.43284, size=(10000, 20))
        self.__test_feats_groups_2d(array)
        array = np.random.normal(-(1 / 3), (2 / 3), size=(9973, 31))
        self.__test_feats_groups_2d(array)
        array = np.random.normal(-3, 5, size=(263, 4547))
        self.__test_feats_groups_2d(array)

    def test_normalize_np_array_iv(self):
        array = np.random.normal(13.573, 4.43284, size=(10000, 20))
        self.__test_feats_shift_norm(array)
        array = np.random.normal(-(1 / 3), (2 / 3), size=(9973, 31))
        self.__test_feats_shift_norm(array)
        array = np.random.normal(-3, 5, size=(263, 4547))
        self.__test_feats_shift_norm(array)

    def test_normalize_np_array_v(self):
        array = np.random.uniform(size=(9987, 51))
        self.__test_feats_shift_norm(array)
        array = np.random.uniform(-1., 1., size=(8000, 10))
        self.__test_feats_shift_norm(array)
        array = np.random.uniform(-17., -3., size=(311, 9323))
        self.__test_feats_shift_norm(array)

    def test_normalize_np_array_vi(self):
        array = np.random.normal(13.573, 4.43284, size=(10000, 20))
        norm_ar = pp_test.normalize_np_array(array)
        assert (norm_ar != array).all()
        self.__assert_norm_array(norm_ar)
        array = np.random.normal(-(1 / 3), (2 / 3), size=(9973, 31))
        norm_ar = pp_test.normalize_np_array(array)
        self.__assert_norm_array(norm_ar)
        array = np.random.normal(-3, 5, size=(263, 4547))
        norm_ar = pp_test.normalize_np_array(array)
        self.__assert_norm_array(norm_ar)

    def test_normalize_np_array_vii(self):
        array = np.random.uniform(size=(9987, 51))
        norm_ar = pp_test.normalize_np_array(array)
        assert (norm_ar != array).all()
        self.__assert_norm_array(norm_ar)
        array = np.random.uniform(-1., 1., size=(8000, 10))
        norm_ar = pp_test.normalize_np_array(array)
        self.__assert_norm_array(norm_ar)
        array = np.random.uniform(-17., -3., size=(311, 9323))
        norm_ar = pp_test.normalize_np_array(array)
        self.__assert_norm_array(norm_ar)

    @staticmethod
    def __compare_nested_lists(list_of_list, list_of_ndarray):
        compare_list = []

        for list_, array in zip(list_of_list, list_of_ndarray):
            assert isinstance(list_, list) and isinstance(array, np.ndarray)
            if len(list_) != 0 and array.size != 0:
                temp_array = np.asarray(list_, dtype=np.float32)
                compare_list.append((temp_array != array).all())
        assert (np.asarray(compare_list) != True).any()

    def test_normalize_numeric_lists(self):
        reply_ner = pp_test.convert_token_docs_label(reply_filtered)
        norm_er = pp_test.normalize_numeric_lists(reply_ner, dtype=np.float32)
        self.__compare_nested_lists(reply_ner, norm_er)
        norm_er = pp_test.normalize_numeric_lists(reply_ner, feature_grouping=True
                                                  , dtype=np.float32)
        self.__compare_nested_lists(reply_ner, norm_er)
        norm_er = pp_test.normalize_numeric_lists(reply_ner, variance_one=False
                                                  , dtype=np.float32)
        self.__compare_nested_lists(reply_ner, norm_er)

    def test_create_class_dict(self):
        class_dict = pp_test.create_class_dict(feats_docs, feats_labels)
        assert len(class_dict.keys()) == 2
        assert sorted(class_dict['serious']) == sorted(true_class_tokens['serious'])
        assert sorted(class_dict['sarcasm']) == sorted(true_class_tokens['sarcasm'])
