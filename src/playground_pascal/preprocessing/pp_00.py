from __future__ import unicode_literals

import math

import numpy as np
import torch

import src.tools.helpers as helpers
from src.preprocessing.datahandler import DataHandler
from src.preprocessing.pipeline import *
from src.preprocessing.preprocessing import Preprocessing
from src.tools.config import Config

#           _   _   _
#  ___  ___| |_| |_(_)_ __   __ _ ___
# / __|/ _ \ __| __| | '_ \ / _` / __|
# \__ \  __/ |_| |_| | | | | (_| \__ \
# |___/\___|\__|\__|_|_| |_|\__, |___/
#                           |___/

path_sw_full = Config.data_path('stop_words_full.txt')
path_sw_cut = Config.data_path('stop_words_cut.txt')

path_train_post_spacy = Config.data_path('pw_00_train_post_spacy.pkl')
path_test_post_spacy = Config.data_path('pw_00_test_post_spacy.pkl')
path_train_reply_spacy = Config.data_path('pw_00_train_reply_spacy.pkl')
path_test_reply_spacy = Config.data_path('pw_00_test_reply_spacy.pkl')

path_train_post_lemma = Config.data_path('pw_00_train_post_lemma.pkl')
path_train_reply_lemma = Config.data_path('pw_00_train_reply_lemma.pkl')
path_test_post_lemma = Config.data_path('pw_00_test_post_lemma.pkl')
path_test_reply_lemma = Config.data_path('pw_00_test_reply_lemma.pkl')

spacy_model = 'en_core_web_md'
# spacy_model = 'en'

load_cache = True
create_cache = True

# Number of maximum examples. -1 if you don't want to use all data
max_train = -1
max_test = -1

#                                    _
#  _ __  _ __ ___   ___ ___  ___ ___(_)_ __   __ _
# | '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
# | |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
# | .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
# |_|                                        |___/

if __name__ == '__main__':

    print("Starting preprocessing for TF-IDF........")
    pp = Preprocessing(spacy_model)
    nlp = pp.get_nlp()

    dh = DataHandler()
    dh.load_train_test(Config.path.data_folder)
    df_train = dh.get_train_df(deep_copy=False)
    df_test = dh.get_test_df(deep_copy=False)

    if max_train > 0: df_train = df_train[:math.floor(max_train)]
    if max_test > 0: df_test = df_test[:math.floor(max_test)]

    train_post_token = tokenize(df_train, path_train_post_spacy, 'post', pp, create_cache, load_cache)
    test_post_token = tokenize(df_test, path_test_post_spacy, 'post', pp, create_cache, load_cache)

    train_reply_token = tokenize(df_train, path_train_reply_spacy, 'reply', pp, create_cache, load_cache)
    test_reply_token = tokenize(df_test, path_test_reply_spacy, 'reply', pp, create_cache, load_cache)

    print(helpers.get_time_duration() + " - Set stop words: " + path_sw_full)
    nlp.add_stop_word_def(path_sw_full)
    train_post_lemma = lemmatize(train_post_token, path_train_post_lemma, 'post', pp, create_cache=create_cache,
                                 load_cache=load_cache)
    test_post_lemma = lemmatize(test_post_token, path_test_post_lemma, 'post', pp, create_cache=create_cache,
                                load_cache=load_cache)

    print(helpers.get_time_duration() + " - Set stop words: " + path_sw_cut)
    nlp.add_stop_word_def(path_sw_cut)
    train_reply_lemma = lemmatize(train_reply_token, path_train_reply_lemma, 'reply', pp, create_cache=create_cache,
                                  load_cache=load_cache)
    test_reply_lemma = lemmatize(test_reply_token, path_test_reply_lemma, 'reply', pp, create_cache=create_cache,
                                 load_cache=load_cache)

    print(helpers.get_time_duration() + " - finished")


def get_tfidf(train=True, test=True, data_size=None, max_tfidf_features=int(1e4)):
    pp = Preprocessing()

    train_reply_strings = None
    train_post_strings = None
    test_post_strings = None
    test_reply_strings = None
    vocab_path = Config.data_path('pw_00_vocab.pkl')
    if Path(vocab_path).is_file():
        vocab = load_from_disk(vocab_path)
    else:
        train_post_lemma = load_from_disk(Config.data_path('pw_00_train_post_lemma.pkl'))
        train_reply_lemma = load_from_disk(Config.data_path('pw_00_train_reply_lemma.pkl'))
        train_post_strings = pp.inner_str_join_2d_list(train_post_lemma, concat_text_str=' ')
        train_reply_strings = pp.inner_str_join_2d_list(train_reply_lemma, concat_text_str=' ')
        test_post_lemma = load_from_disk(Config.data_path('pw_00_test_post_lemma.pkl'))
        test_reply_lemma = load_from_disk(Config.data_path('pw_00_test_reply_lemma.pkl'))
        test_post_strings = pp.inner_str_join_2d_list(test_post_lemma, concat_text_str=' ')
        test_reply_strings = pp.inner_str_join_2d_list(test_reply_lemma, concat_text_str=' ')
        string_list = train_post_strings + train_reply_strings + test_post_strings + test_reply_strings
        vocab = create_vocab(string_list, pp, path=vocab_path)

    result = {}
    post_size = int(data_size / 2) if data_size is not None else None
    if train:
        if not (train_reply_strings and train_post_strings):
            train_post_lemma = load_from_disk(Config.data_path('pw_00_train_post_lemma.pkl'))
            train_reply_lemma = load_from_disk(Config.data_path('pw_00_train_reply_lemma.pkl'))
            train_post_strings = pp.inner_str_join_2d_list(train_post_lemma, concat_text_str=' ')
            train_reply_strings = pp.inner_str_join_2d_list(train_reply_lemma, concat_text_str=' ')

        train_post_tfidf = create_tfidf(train_post_strings, 'post', pp, vocab=vocab, min_df=3, ngram_range=(1, 2),
                                        max_tfidf_features=max_tfidf_features)
        train_reply_tfidf = create_tfidf(train_reply_strings, 'reply', pp, vocab=vocab, min_df=3, ngram_range=(1, 2),
                                         max_tfidf_features=max_tfidf_features)
        result.update({'train_post': train_post_tfidf[:post_size, ],
                       'train_reply': train_reply_tfidf[:data_size, ]})
    if test:
        if not (test_post_strings and test_reply_strings):
            test_post_lemma = load_from_disk(Config.data_path('pw_00_test_post_lemma.pkl'))
            test_reply_lemma = load_from_disk(Config.data_path('pw_00_test_reply_lemma.pkl'))
            test_post_strings = pp.inner_str_join_2d_list(test_post_lemma, concat_text_str=' ')
            test_reply_strings = pp.inner_str_join_2d_list(test_reply_lemma, concat_text_str=' ')
        test_post_tfidf = create_tfidf(test_post_strings, 'post', pp, vocab=vocab, min_df=3, ngram_range=(1, 2),
                                       max_tfidf_features=max_tfidf_features)
        test_reply_tfidf = create_tfidf(test_reply_strings, 'reply', pp, vocab=vocab, min_df=3, ngram_range=(1, 2),
                                        max_tfidf_features=max_tfidf_features)

        result.update({'test_post': test_post_tfidf[:post_size, ],
                       'test_reply': test_reply_tfidf[:data_size, :]})
    return result


def get_labels(data_size=None):
    dh = DataHandler()
    dh.load_train_test(Config.path.data_folder)

    # We don't need this for CrossEntropy, the Loss function we actually should use
    # (or NLLLoss, which is the same, with log_softmax beforehand, more numerically stable this way)
    # train = dh.add_negative_label(dh.get_train_df())
    # test = dh.add_negative_label(dh.get_test_df())
    train = dh.get_train_df()
    test = dh.get_test_df()
    train = train['sarcasm'].values.astype(dtype=np.long, copy=False)
    test = test['sarcasm'].values.astype(dtype=np.long, copy=False)
    train = torch.from_numpy(train)
    test = torch.from_numpy(test)
    return train[:data_size], test
