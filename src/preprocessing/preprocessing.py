from collections import Counter

import numpy as np
from DocumentFeatureSelection import interface
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import src.tools.helpers as helpers
from src.preprocessing.spacyoverlay import SpacyOverlay


class Preprocessing:
    """
    This class is for preprocessing with Natural Language Processing.

    SpaCy is used to parse list-like (iterable) objects of strings.
    It has light-weight methods in the sense that you have to invoke them
    probably more often. But they don't do much at once and are easy to handle.
    An additional bonus is, that you can use similar operations repeatedly without
    losing the result of intermediate steps.
    It aims at a multi-feature preprocessing where you can stack your features in
    the manner you want. The price is that you have to define a bit more yourself
    in terms of a pipeline consisting of methods.

    What features you can theoretically get:
    * Every string attribute of the SpaCy Token class
    * Labels for special word properties like Entity Recognition results
    * tfidf, SOA vectors for each post/reply
    * Normalizing

    If you want to get text data without words below a certain frequency,
    use the final data transformations like tf-idf or word embeddings (vectors)
    for that.
    If you want to modify the SpaCy language model further use the get_nlp method.
    And do it through the SpacyOverlay class.

    Do only transform it to a Numpy array if necessary due to operations.
    Conversion in the methods of this class happens automatically if necessary

    Attributes
    ----------
    nlp : SpacyOverlay
        Class which offers some useful functions for a better SpaCy experience.
        Please use the getter.
    soa_features: dict of dict
        Each class is a key which returns:
        dict of features - as keys - and their SOA scores as value
        Only available after create_soa_features method

    Methods
    -------
    get_nlp()
        Get the SpacyOverlay object, so you can change the pipeline, etc.
    run_spacy_pipeline()
        Apply the SpaCy pipeline to an iterable list of strings, e.g. Pandas Series
    filter_spacy_tokens()
        Filter SpaCy tokens in preprocessed Docs out
    convert_token_docs_text()
        Convert the through SpaCy preprocessed data to text-form
    convert_token_docs_label()
        Convert the through SpaCy preprocessed data to count-labels
    filter_by_frequency()
        Filter every string token (word) below a certain frequency out
    inner_str_join_2d_list()
        Concat the string elements of the inner lists of a nested list
    normalize_np_array()
        Normalize a Numpy array so that the values are between 0 & 1.
    normalize_numeric_lists()
        Normalize a (nested) list. Return a Numpy array. Uses normalize_np_array()
    str_list_to_vectors()
        Create a matrix with tf-idf vectors out of a list of data strings
    create_class_dict()
        Intermediate step for Strength of Association (SOA) features
    create_soa_features()
        Compute the SOA scores for the given feature class dict
    convert_features_to_soa()
        Compute the SOA scores for the features in the given training examples
    soa_train_data_score_pipe()
        Use this to compute the SOA scores at once for train data. Wraps methods above

    See Also
    --------
    - https://spacy.io/usage/
    - For SOA: https://saifmohammad.com/WebDocs/hashtags-MK.pdf

    Examples
    --------
    # A simple way to use it:
    pp = Preprocessing()
    nlp = pp.get_nlp()
    nlp.add_stop_word_def(stop_word_file=my_file)
    replies = pp.run_spacy_pipeline(df['reply'])
    posts = load_from_disk('data/posts.pkl')

    filtered_tokens = pp.filter_spacy_tokens(replies)
    lemmas = pp.convert_token_docs_text(filtered_tokens, token_kind='lemma_')
    ners = pp.convert_token_docs_label(filtered_tokens)
    lemmas, ners = pp.filter_by_frequency(lemmas, mirror_docs_list=ners)

    str_data = pp.inner_str_join_2d_list(lemmas)
    tfidf_vecs = pp.str_list_to_vectors(str_data)

    # Concat reply and post data examples to one [work in progress, but in DataHandler]
    soa_scores = pp.soa_train_data_score_pipe(lemmas, df['sarcasm'].tolist())
    pp.normalize_np_array(soa_scores)

    # Concat features to 2D for each data example [work in progress, but in DataHandler]
    # Now plug it into the DNN
    """

    def __init__(self, model_type='en_core_web_md', pipeline_to_remove=list(['textcat'])):
        """
        This is the class for preprocessing the text data of SARC

        If you further want to modify the SpaCy language model use the get_nlp method.

        Parameters
        ----------
        model_type: str
            The name of the model which SpaCy should load.
            You should have downloaded it beforehand.
            As of SpaCy 2.0 that would be 'en', 'en_core_web_sm',
            'en_core_web_md' and 'en_core_web_lg'
        pipeline_to_remove: list of str
            The names of the pipeline steps to disable.
            Keep the dependencies of the parts in mind!
        """
        self.nlp = SpacyOverlay(model_type, pipeline_to_remove)
        self.soa_features = None

    def get_nlp(self):
        """
        Returns
        -------
        SpacyOverlay
            Wrapper class for SpaCy language model.
            Has also a getter for the SpaCy class itself.
        """
        return self.nlp

    def run_spacy_pipeline(self, data, n_threads=-1, batch_size=500):
        """
        Parses the text data with the internal SpaCy language model

        TIP: For stop word extension, etc.:
        Use SpacyOverlay class to extends the pipeline!

        Parameters
        ----------
        data : pandas.Series of str or list-like of str
            The object with the textual data.
            Doesn't have to be SARC data
        n_threads : int
            Number of CPU threads. -1 means, it uses all
        batch_size : int
            Weak influence on performance if above certain threshold
        Returns
        -------
        list of spacy.tokens.doc.Doc
            The list with the parsed texts as SpaCy Doc objects
        """
        pipe = self.nlp.get_nlp().pipe
        return [doc for doc in pipe(data, n_threads=n_threads, batch_size=batch_size)]

    @staticmethod
    def filter_spacy_tokens(docs, no_stop_words=True, no_punctuation=True):
        """
        Filters out SpaCy Tokens.

        You can get from the text parsed with SpaCy the information
        that is more relevant. Will ignore tokens that are whitespaces!

        Parameters
        ----------
        docs : list-like of spacy.tokens.doc.Doc
            The SpaCy docs to convert to strings.
            Does not get overwritten.
        no_stop_words : bool
            True if you don't want stop words, False else
        no_punctuation: bool
            True if you don't want punctuation signs, False else
        Returns
        -------
        list of list
            Nested list with filtered tokens of the SpaCy Docs
        """
        return [[token
                 for token in doc
                 if not ((no_punctuation and token.is_punct)
                         or (no_stop_words and token._.is_stop)
                         or token.is_space)]
                for doc in docs]

    def convert_token_docs_text(self, docs, token_kind='lower_', transform_specials=False
                                , concat_text_str=None):
        """
        Converts the (parsed) text in the way that it filters specific information.

        You can get from the text parsed with SpaCy certain information
        in a more productive form. E.g. the lemmatized text.
        TIP: Use filter_spacy_tokens() beforehand.
        This is only intended to be used for textual results of the parsing.

        Parameters
        ----------
        docs : list-like of spacy.tokens.doc.Doc
            The SpaCy docs to convert to strings.
            Does not get overwritten.
        token_kind : str
            The name of the attribute of the parsed token you want to have.
            Options: 'text', 'lemma_', 'lower_', ... everything for which
            Token returns a string. Should be written correctly!
        transform_specials : bool
            True if you want to have all numbers as '<number>', email addresses as
            <email> and URLs as <url>, False else.
            Transforms only numbers like '10.9', '10', '500,000.9 'ten', etc.
            NOT 'tens', thirties' or '50€' or an Airbus 'A320', ...
        concat_text_str : str
            None results in a list of list with individual tokens ==>
            You will need that if you want to use a word embedding layer
            (word vectors)! | Other option:
            With which string you want to concatenate the resulting words
            after the conversion. '' results in a string without whitespaces
            for each document (former reply or post in the data set).
        Returns
        -------
        list of str or list of list
            The converted tokens of the SpaCy Docs in string form
        See Also
        --------
        https://spacy.io/api/token#attributes
        """

        def get_special_value(token):
            if transform_specials and token.like_num:
                return 'number'
            elif transform_specials and token.like_url:
                return '<url>'
            elif transform_specials and token.like_email:
                return '<email>'
            else:
                return getattr(token, token_kind)

        # Expand with dict look up as 'switch case' if you want to add more such stuff
        result = [[get_special_value(token) for token in doc] for doc in docs]
        if concat_text_str:
            result = self.inner_str_join_2d_list(result, concat_text_str)
        return result

    @staticmethod
    def convert_token_docs_label(docs, token_kind='ent_type'):
        """
        Get the labels for more abstract parsing results like Named Entity Recognition or POS

        Creates a nested list with the same shape as the input which has numeric labels.
        Works with NER and POS... Creates only a scalar output for each token.
        No relations or begin & end as for now. That would lead to two dimensions.
        Judgement: Not useful. CAUTION: Normalizing later on required!

        Parameters
        ----------
        docs : list-like of spacy.tokens.doc.Doc
            The SpaCy docs to convert to strings.
            Does not get overwritten.
        token_kind : str
            The name of the attribute of the parsed token you want to have.
            Options: 'ent_type', 'pos', 'tag', 'prob', ... everything for which
            Token returns an int and makes sense as individual value - considering,
            that it is applied to all tokens. Should be written correctly
        Returns
        -------
        list of list
            The nested list, with the into integer converted tokens, of the SpaCy Docs
        See Also
        --------
        https://spacy.io/api/token#attributes
        """
        # More complex constructions, e.g. conditions possible of course, if necessary
        result = [[getattr(token, token_kind)
                   for token in doc]
                  for doc in docs]
        return result

    @staticmethod
    def __white_list_filter(list_, compare_list):
        return [[token
                 for do_keep, token in zip(mirror, doc) if do_keep]
                for mirror, doc in zip(compare_list, list_)]

    def filter_by_frequency(self, docs, min_freq=5, mirror_docs_list=None):
        """
        Removes items whose frequency is too low.

        Theoretically you can filter here anything - with anything mirrored.
        As long as it is countable, iterable, behaves like a list and the
        objects for which to mirror have the same structure as the original.
        And the final elements of the list of lists should be hashable.

        Parameters
        ----------
        docs: list of list
            A nested list of string tokens
        min_freq : int
            The minimal corpus frequency for a token as absolute value.
            If it is below that, it gets filtered out
        mirror_docs_list : list of list
            If None, do nothing, else:
            Lists where the tokens with the SAME POSITION shall be
            filtered out. Does not apply any criteria to it. Just matches
            the lists. USE-CASE: If you have additional features which are
            not as unique as the words - like POS or NER - but match them.
        Returns
        -------
        list of list, list of list
            1. Nested list with string tokens, only containing
            those with a greater corpus frequency than min_freq
            2. If mirror_docs_list not None: nested mirrored list,
            else None
        """
        if mirror_docs_list:
            for list_ in mirror_docs_list:
                helpers.nested_list_test(list_)
        helpers.nested_list_test(docs)
        word_freq = Counter(helpers.flatten(docs))
        remaining_types = {x for x in word_freq if word_freq[x] >= min_freq}

        if mirror_docs_list:
            mirror = [[True
                       if (token in remaining_types)
                       else False
                       for token in doc]
                      for doc in docs]
            docs_filtered = self.__white_list_filter(docs, mirror)

            for i, likeness in enumerate(mirror_docs_list):
                mirror_docs_list[i] = self.__white_list_filter(likeness, mirror)
        else:
            docs_filtered = [[token for token in doc
                              if token in remaining_types]
                             for doc in docs]
        return docs_filtered, mirror_docs_list

    @staticmethod
    def inner_str_join_2d_list(nested_list, concat_text_str=' '):
        """
        Concatenate the str elements of a list of list

        Parameters
        ----------
        nested_list : list of list
            A nested list of string tokens
        concat_text_str : str
            None results in a list of list with individual tokens ==>
            You will need that if you want to use a word embedding layer
            (word vectors)! | Other option:
            With which string you want to concatenate the resulting words
            after the conversion. '' results in a string without whitespaces
            for each document (former reply or post in the data set).
        Returns
        -------
        list of str
            A list where each element is a str which consists
            of the joined former elements of the list that
            has been at the same index
        """
        return [concat_text_str.join(doc) for doc in nested_list]

    @staticmethod
    def normalize_np_array(array, variance_of_one=True, epsilon=1e-12
                           , feature_grouping=False, dtype=np.float32):
        """
        Normalizes the values of an n-dimensional Numpy array to interval [-1,1]

        Parameters
        ----------
        array : np.ndarray
            Should have equal element length along a dimension.
            Does not get overwritten.
        variance_of_one : bool
            If True: variance = 1, mean = 0. Like in 'tricks' lecture, slide 43.
            False: keep distribution, basically compress it to [-1,1]
        epsilon: float
            Small value to avoid division by zero.
        feature_grouping : bool
            If True and variance_one is True, apply the normalizing to the
            feature group of a data example, not the feature dimension.
            False: CAUTION: Does not work properly with more than two dimensions!
            Normalize each 2D feature set separately
        dtype : type
            Data type of the returned array
        Returns
        -------
        np.ndarray
            Array with 0 <= values => 1.
            Same shape as input. dtype=np.float32.
        """
        if variance_of_one and feature_grouping:
            dim = array.ndim - 1
            mean = array.mean(axis=dim)
            std = array.std(axis=dim) + epsilon
            array_nlzd = (array - mean[:, np.newaxis]) / std[:, np.newaxis]
        elif variance_of_one:
            mean = array.mean(axis=0)
            std = array.std(axis=0) + epsilon
            array_nlzd = (array - mean[np.newaxis, :]) / std[np.newaxis, :]
        else:
            array_nlzd = (array - array.min()) / (array.ptp() / 2 + epsilon)
            array_nlzd -= 1
        return array_nlzd.astype(dtype=dtype, copy=False)

    @staticmethod
    def __normalize_nested_list_across_features(ar_list, eps, dtype):
        def __dict_plus_op(dict_, key, function_):
            dict_[key] += function_

        longest_ar = np.array([ar.size for ar in ar_list]).max()
        size_dict = {}
        sum_dict = {}
        mean_dict = {}
        std_dict = {}

        for dim in range(0, longest_ar):
            size_dict[dim] = 0
            sum_dict[dim] = 0
            mean_dict[dim] = 0
            std_dict[dim] = []
        [__dict_plus_op(size_dict, _dim_, 1) for _ar_ in ar_list for _dim_, _ in enumerate(_ar_)]
        [__dict_plus_op(sum_dict, _dim_, _el_) for _ar_ in ar_list for _dim_, _el_ in enumerate(_ar_)]
        [mean_dict.update([(_dim_, (sum_dict[_dim_] / size_dict[_dim_]))]) for _dim_ in mean_dict]
        [std_dict[_dim_].append(_el_) for _ar_ in ar_list for _dim_, _el_ in enumerate(_ar_)]

        for dim in std_dict:
            std_dict[dim] = np.array(std_dict[dim]).std() + eps

        return [np.asarray([(_el_ - mean_dict[_dim_]) / std_dict[_dim_]
                            for _dim_, _el_ in enumerate(_ar_)], dtype=dtype)
                for _ar_ in ar_list]

    def normalize_numeric_lists(self, list_, dtype=np.float32, epsilon=1e-12
                                , variance_one=True, feature_grouping=False):
        """
        Normalizes the values of (nested) lists.

        Slow. Better use padding beforehand to get equally long data examples,
        and then:
        If you have a list of list, where all the nested lists have equal
        length, just convert it into a Numpy array and use normalize_np_array()

        Parameters
        ----------
        list_ : list
            The list or nested list which values shall be normalized.
            Does not get overwritten.
        dtype : type
            The data type of the values in the (nested) list
        epsilon: float
            Small value to avoid division by zero.
        variance_one : bool
            If True: variance = 1, mean = 0. Like in 'tricks' lecture, slide 43.
            False: keep distribution, basically compress it to [-1,1]
        feature_grouping : bool
            If True and variance_one is True, apply the normalizing to the
            feature group of a data example, not the feature dimension,
            retains the distribution in great part, var = 1 across all dimensions.
            False: Leads to more or less uniform distribution.
            var = 1 only in first dimension.
        Returns
        -------
        list of np.ndarray
            Nested List with arrays of values in [-1, 1].
            Same shape as input.
        """
        ar_list = [np.array(doc, dtype=dtype) for doc in list_]

        if variance_one and feature_grouping:
            size_ar = np.array([ar.size for ar in ar_list])
            sum_ar = np.array([ar.sum(axis=0) for ar in ar_list])
            mean_ar = sum_ar / size_ar
            std_ar = np.array([(ar.std(axis=0) + epsilon) for ar in ar_list])
            ar_list = [(ar - mean_ar[i]) / std_ar[i]
                       for i, ar in enumerate(ar_list)]
        elif variance_one:
            ar_list = self.__normalize_nested_list_across_features(ar_list, epsilon, dtype)
        else:
            min_ar = np.array([(ar.min() if ar.size != 0 else 0) for ar in ar_list])
            min_ = min_ar.min()
            max_ar = np.array([(ar.max() if ar.size != 0 else 0) for ar in ar_list])
            ptp = (max_ar.max() - min_) / 2 + epsilon
            ar_list = [(ar - min_) / ptp for ar in ar_list]

        return ar_list

    @staticmethod
    def str_list_to_vocab(list_str, min_df=3, ngram_range=(1, 2), max_features=int(1e4), analyzer='word'):
        """
        The vocab of a list of docs. Can be used to ensure that the str_to_tfidf returns the same
        order of tokens/ngrams for various data sets (e.g. train-post, train-reply, test-post, test-reply).
        Parameters
        ----------
        list_str : list of str
            A sequence of strings. They will be transformed
        min_df : int
            float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold.
            This value is also called cut-off in the literature.
            If of type float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        max_features : int
            For the sklearn tf-idf default vectorizer.
            If not None, build a vocabulary that only considers the
            top max_features ordered by term frequency across the corpus.
        analyzer: (string, {'word', 'char', 'char_wb'} or callable)
            Whether the feature should be made of word or character n-grams. Option 'char_wb'
            creates character n-grams only from text inside word boundaries; n-grams at the
            edges of words are padded with space. If a callable is passed it is used to
            extract the sequence of features out of the raw, unprocessed input.

        Returns
        -------
        dict
            dict of form {'<token>': <index>}
        """
        count_vtr = CountVectorizer(token_pattern=r'(\w+[^\s]*\w+|\w+)', min_df=min_df, ngram_range=ngram_range,
                                    dtype=np.float32, analyzer=analyzer, max_features=int(max_features))
        return count_vtr.fit(list_str).vocabulary_

    @staticmethod
    def str_list_to_vectors(list_str, vocabulary=None, min_df=3, ngram_range=(1, 2)
                            , max_features=int(3e4), analyzer='word', tfidf=True):
        """
        Transforms a list of strings to tf-idf feature vectors.

        Parameters
        ----------
        list_str : list of str
            A sequence of strings. They will be transformed
        min_df : int
            float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold.
            This value is also called cut-off in the literature.
            If of type float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        vocabulary: (Mapping or iterable, optional)
            Either a Mapping (e.g., a dict) where keys are terms and values are indices
            in the feature matrix, or an iterable over terms. If not given, a vocabulary
            is determined from the input documents.
        tfidf: bool
            `True` for tfidf, `False` for document-freq.
        max_features : int
            For the sklearn tf-idf default vectorizer.
            If not None, build a vocabulary that only considers the
            top max_features ordered by term frequency across the corpus.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        analyzer: (string, {'word', 'char', 'char_wb'} or callable)
            Whether the feature should be made of word or character n-grams. Option 'char_wb'
            creates character n-grams only from text inside word boundaries; n-grams at the
            edges of words are padded with space. If a callable is passed it is used to
            extract the sequence of features out of the raw, unprocessed input.
        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Tf-idf-weighted document-term matrix, same order as input
        """
        params = {'token_pattern': r'(\w+[^\s]*\w+|\w+)', 'vocabulary': vocabulary,
                  'max_features': int(max_features), 'min_df': min_df,
                  'ngram_range': ngram_range, 'dtype': np.float32,
                  'analyzer': analyzer}
        if tfidf:
            vectorizer = TfidfVectorizer(sublinear_tf=True, **params)
        else:
            vectorizer = CountVectorizer(**params)
        return vectorizer.fit_transform(list_str)

    @staticmethod
    def create_class_dict(docs, labels):
        """
        Create a dict for the SOA score calculation

        Parameters
        ----------
        docs : list of list
            Nested list with data examples or features of them.
            Strings shouldn't be joined but be in token form.
            Otherwise, one example will have only this one feature.
            Therefore, a most likely unique one => leads to random
            guessing in the end.
        labels: list of int
            The corresponding labels for the nested lists in docs.
            1 as element value for sarcasm and 0 for no sarcasm.
        Returns
        -------
        dict
            Each class label in the data set represents a key -
            'sarcasm' and 'serious'.
            The values are nested list of lists. The inner list
            is one data example, which can contain single elements
            or tuples of complex features.
        """
        # Because we can't be certain that the docs are strictly
        # in the order 10101, we have to do that the 'slow' way
        docs_sarcasm = [doc for i, doc in enumerate(docs) if labels[i]]
        docs_serious = [doc for i, doc in enumerate(docs) if not labels[i]]
        return {"sarcasm": docs_sarcasm, "serious": docs_serious}

    @staticmethod
    def __soa_score_obj_to_dict(word_score_items, label_dict):
        feature_dicts = {}

        for label in label_dict:
            feature_dicts[label] = {}
        for item in word_score_items:
            feature_dicts[item['label']][item['feature']] = item['score']
        return feature_dicts

    def create_soa_features(self, class_dict):
        """
        Creates SOA features where each feature is saved in a dict

        Parameters
        ----------
        class_dict : dict
            Each class label in the data set represents a key -
            'sarcasm' and 'serious'.
            The values are nested list of lists. The inner list
            is one data example, which can contain single elements
            or tuples of complex features.
        Returns
        -------
        dict of dict
            Each class is a key which returns:
            dict of features - as keys - and their SOA scores as value
        See Also
        --------
        https://pypi.python.org/pypi/DocumentFeatureSelection/1.4.1
        Examples
        --------
        # example for the input class_dict:
        input_dict = {"label_a": [["I", "aa", "aa", "aa", "aa", "aa"]
                                  , ["bb", "aa", "aa", "aa", "aa", "aa"]],
                      "label_b": [["bb", "bb", "bb"]]}
        """
        soa_srobj = interface.run_feature_selection(
            input_dict=class_dict
            , method="soa"
            , n_jobs=-1
            , use_cython=True
        )
        score_object = soa_srobj.convert_score_matrix2score_record()
        self.soa_features = self.__soa_score_obj_to_dict(score_object
                                                         , soa_srobj.label2id_dict)
        return self.soa_features

    @staticmethod
    def convert_features_to_soa(feature_docs, labels, soa_dict):
        """
        Coverts the features of the examples in class_dict to their SOA score

        In case of test data use the soa_dict of train data.
        In case of unknown feature, the median of the classes scores is used

        Parameters
        ----------
        feature_docs : list of list
            The nested list with the features which represent a training example
        labels: list of int
            The corresponding labels for the nested lists in docs.
            1 as element value for sarcasm and 0 for no sarcasm.
        soa_dict: dict of dict
            Each class is a key which returns:
            dict of all features - as keys - and their SOA scores as value
        Returns
        -------
        list of np.ndarray
            The nested list with the SOA scores which represent a training example.
            Same shape and order as feature_docs parameter. NOT normalized!
        See Also
        --------
        https://saifmohammad.com/WebDocs/hashtags-MK.pdf
        """
        # This part is for the default value
        sarc = soa_dict["sarcasm"]
        norm = soa_dict["serious"]
        sarc_ar = np.array(list(sarc.values()), dtype=np.float32)
        norm_ar = np.array(list(norm.values()), dtype=np.float32)
        sarc_default = np.median(sarc_ar)
        norm_default = np.median(norm_ar)

        # This part for the feature <-> score mapping
        sarc_dict = soa_dict['sarcasm'].get
        norm_dict = soa_dict['serious'].get

        def __feature_scoring(is_sarc, doc):
            return np.array([sarc_dict(feature, sarc_default)
                             if is_sarc
                             else norm_dict(feature, norm_default)
                             for feature in doc], dtype=np.float32)

        list_ = [__feature_scoring(is_sarc, doc)
                 for is_sarc, doc in zip(labels, feature_docs)]
        return list_

    def soa_train_data_score_pipe(self, feature_docs, labels):
        """
        Converts the input to a resembling output with the SOA score.

        In case of test data: use the individual methods.
        You can get the soa_dict from the class attribute soa_features
        after execution. This method consists of a pipeline,
        which wraps create_class_dict() and create_soa_features()
        and convert_features_to_soa()

        Parameters
        ----------
        feature_docs : list of list
            The nested list with the features which represent a training example
        labels: list of int
            The corresponding labels for the nested lists in docs.
            1 as element value for sarcasm and 0 for no sarcasm.
        Returns
        -------
        list of np.ndarray
            Nested list with the same shape as the input. NOT normalized!
            Has at each place of a former element it's SOA score
        See Also
        --------
        https://saifmohammad.com/WebDocs/hashtags-MK.pdf
        """
        class_dict = self.create_class_dict(feature_docs, labels)
        soa_dict = self.create_soa_features(class_dict)
        soa_docs = self.convert_features_to_soa(feature_docs, labels, soa_dict)
        return soa_docs
