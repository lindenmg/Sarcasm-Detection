from pathlib import Path

from src.tools.helpers import get_time_duration, load_from_disk, save_to_disk


# Process Spacy dump
def tokenize(df, path, type, pp, create_cache=True, load_cached_file=True):
    tokens = []
    if not load_cached_file or not Path(path).is_file():
        # and not Path(path).is_file():
        print(get_time_duration() + " - " + type + " : apply spacy pipeline")
        docs = df[type]
        if type == 'post': docs = docs[0::2]
        tokens = pp.run_spacy_pipeline(docs)
        if create_cache:
            print(get_time_duration() + " - " + type + " : dump spacy pipeline to " + path)
            save_to_disk(tokens, path)
    elif Path(path).is_file():
        print(get_time_duration() + " - " + type + " : load spacy dump from " + path)
        tokens = load_from_disk(path)
    return tokens


# Lemma dump
def lemmatize(token, path, type, pp,
              no_stop_words=True, no_punctuation=True, token_kind='lemma_', transform_num=True,
              create_cache=True, load_cache=True):
    if not load_cache or not Path(path).is_file():
        print(get_time_duration() + " - " + type + " : lemmatize")
        list_lemma = pp.filter_spacy_tokens(token, no_stop_words=no_stop_words, no_punctuation=no_punctuation)
        list_lemma = pp.convert_token_docs_text(list_lemma, token_kind=token_kind, transform_num=transform_num)
        if create_cache:
            print(get_time_duration() + " - " + type + " : dump lemma to " + path)
            save_to_disk(list_lemma, path)
    elif Path(path).is_file():
        print(get_time_duration() + " - " + type + " : load lemmas from " + path)
        list_lemma = load_from_disk(path)
    return list_lemma


# TfIdf Dump
def create_inner_str_join_2d_list(docs, type, pp, concat_text_str=' '):
    print(get_time_duration() + " - " + type + " : create strings")
    return pp.inner_str_join_2d_list(docs, concat_text_str=concat_text_str)


def create_vocab(list_of_strings, pp, path=None, min_df=3, ngram_range=(1, 2), analyzer='word',
                 max_tfidf_features=int(1e4), create_cache=True, load_cache=True):
    vocab = None
    if not load_cache or (path is not None and not Path(path).is_file()):
        print(get_time_duration() + " - : create vocab")
        vocab = pp.str_list_to_vocab(list_of_strings, min_df=min_df, ngram_range=ngram_range, analyzer=analyzer,
                                     max_features=max_tfidf_features)
        if create_cache:
            save_to_disk(vocab, path)
    elif path is not None and Path(path).is_file():
        print(get_time_duration() + " - : load vocab from " + path)
        vocab = load_from_disk(path)
    return vocab


def create_tfidf(list_of_strings, type, pp, vocab=None, min_df=3
                 , ngram_range=(1, 2), analyzer='word', max_tfidf_features=int(1e4)):
    print(get_time_duration() + " - " + type + " : create tfidf")
    return pp.str_list_to_vectors(list_of_strings, min_df=min_df, vocabulary=vocab,
                                  ngram_range=ngram_range, analyzer=analyzer, max_features=max_tfidf_features)
