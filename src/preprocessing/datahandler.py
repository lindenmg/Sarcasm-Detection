import os
from pathlib import Path

import gensim
import numpy as np
import pandas as pd
import torch
from gensim import corpora, models
from pandas import DataFrame
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm

import src.tools.helpers as helpers


class DataHandler:
    """
    For the loading and splitting of the data.

    It can load and merge the data of the SARC dat subset provided by IMS.
    It can also split it, shuffle it and transform it to (training and test)
    sets in addition to the basic loaded one
    """

    def __init__(self):
        self.data_df = None
        self.test_df = None
        self.train_df = None

    def load_data(self, comments_path, annotations_path):
        """
        Loads the reduced SARC data set and merges the two CSV files

        It expects '\t' as CSV separator, assumes us-ascii encoding.
        IMPORTANT! CSV values shouldn't contain uneven numbers of `"´

        Parameters
        ----------
        comments_path : str
            The relative or absolute path to the file comments.txt
            Expects the header (comment_id comment)
        annotations_path : str
            The relative or absolute path to the file annotations.txt
            Expects the header (post_id reply_id sarcasm)
        """
        comments = pd.read_csv(comments_path, sep='\t', keep_default_na=False
                               , quoting=3, encoding='us-ascii', quotechar=None
                               , dtype={'comment_id': np.str, 'comment': np.str})
        annotations = pd.read_csv(annotations_path, sep='\t', keep_default_na=False
                                  , quoting=3, encoding='us-ascii', quotechar=None
                                  , dtype={'post_id': np.str, 'reply_id': np.str, 'sarcasm': np.int8})

        data_df = pd.merge(comments, annotations, left_on='comment_id', right_on='reply_id')
        data_df = data_df.merge(comments, left_on='post_id', right_on='comment_id', suffixes=['_l', '_r'])
        data_df.drop(['comment_id_r', 'comment_id_l', 'reply_id'], axis=1, inplace=True)
        data_df.rename(index=str, columns={'comment_l': 'reply', 'comment_r': 'post'}, inplace=True)

        # Thereby we get a column order which is more sensible for humans: (post, reply, sarcasm)
        columns = np.asarray(data_df.columns.tolist())
        columns = columns[np.array([1, 3, 0, 2])]
        data_df = data_df[columns]
        data_df.index = data_df.index.map(int)
        self.data_df = data_df

    def get_data_df(self, deep_copy=True):
        """
        Parameters
        ----------
        deep_copy : bool
            If it shall return a deep copy of the pandas.DataFrame or the reference
        Returns
        -------
        DataFrame
            The complete SARC subset data.
            Columns: (post, reply, sarcasm)
        """
        if self.data_df is None:
            raise ReferenceError("'data_df' is None, hasn't been loaded yet!")

        # Pandas seems to make a deep copy as long as you use .copy()
        if deep_copy:
            return self.data_df.copy(deep=deep_copy)
        else:
            return self.data_df

    def set_data_df(self, data_df, deep_copy=True):
        """
        Assigns the internal pandas.DataFrame data_df

        Parameters
        ----------
        data_df : DataFrame
            Should contain the SARC data and have the columns (post, reply, sarcasm)
            with the appropriate row values - (str, str, [0,1])
        deep_copy : bool
            If it shall make a deep copy of data_df or not
        """
        # Pandas seems to make a deep copy as long as you use .copy()
        if deep_copy:
            self.data_df = data_df.copy(deep=deep_copy)
        else:
            self.data_df = data_df

    def get_train_df(self, deep_copy=True):
        """
        Parameters
        ----------
        deep_copy : bool
            If it shall return a deep copy of the pandas.DataFrame or the reference
        Returns
        -------
        DataFrame
            The training set of the sarc subset data.
            Columns: (post, reply, sarcasm)
        """
        if self.train_df is None:
            self.split_in_train_test()

        # Pandas seems to make a deep copy as long as you use .copy()
        if deep_copy:
            return self.train_df.copy(deep=deep_copy)
        else:
            return self.train_df

    def set_train_df(self, train_df, deep_copy=True):
        """
        Sets the internal training set.

        Use this also for preprocessing of a validation training set

        Parameters
        ----------
        train_df : DataFrame
            Should contain the SARC data and have the columns (post, reply, sarcasm)
            with the appropriate row values - (str, str, [0,1])
        deep_copy : bool
            If it shall make a deep copy of data_df or not
        """
        # Pandas seems to make a deep copy as long as you use .copy()
        if deep_copy:
            self.train_df = train_df.copy(deep=deep_copy)
        else:
            self.train_df = train_df

    def get_test_df(self, deep_copy=True):
        """
        Parameters
        ----------
        deep_copy : bool
            If it shall return a deep copy of the pandas.DataFrame or the reference
        Returns
        -------
        DataFrame
            The test set of the SARC subset data.
            Columns: (post, reply, sarcasm)
        """
        if self.test_df is None:
            self.split_in_train_test()

        # Pandas seems to make a deep copy as long as you use .copy()
        if deep_copy:
            return self.test_df.copy(deep=deep_copy)
        else:
            return self.test_df

    def set_test_df(self, test_df, deep_copy=True):
        """
        Assigns the internal pandas.DataFrame data_df

        Use this also for preprocessing of a validation test set

        Parameters
        ----------
        test_df : DataFrame
            Should contain the SARC data and have the columns (post, reply, sarcasm)
            with the appropriate row values - (str, str, [0,1])
        deep_copy : bool
            If it shall make a deep copy of data_df or not
        """
        # Pandas seems to make a deep copy as long as you use .copy()
        if deep_copy:
            self.test_df = test_df.copy(deep=deep_copy)
        else:
            self.test_df = test_df

    def save_dataset_to_csv(self, result_path):
        """
        Saves the SARC base data in the current state as CSV to the disk

        It doesn't save the index of the DataFrame.
        Uses '\t' as separator for the saved CSV file

        Parameters
        ----------
        result_path : str
            The relative or absolute path where the merged result shall be saved.
        Returns
        -------
        bool
            True if result_path was valid, False else.
            In the later case nothing has been saved.
        """
        return self.__save_df_to_csv(self.data_df, result_path)

    def save_train_test_to_csv(self, result_dir):
        """
        Saves the SARC train and test data in the current state as CSV to the disk

        It doesn't save the index of the DataFrame.
        It saves the test set as 'test.csv' and the train set as 'train.csv'.
        Uses '\t' as separator for the saved CSV file

        Parameters
        ----------
        result_dir : str
            The relative or absolute path of the directory
            where the data sets shall be saved.
        Returns
        -------
        bool
            True if result_path was valid, False else.
            In the later case nothing has been saved.
        """
        path = os.path.join(result_dir, 'test.csv')
        valid = self.__save_df_to_csv(self.test_df, path)
        path = os.path.join(result_dir, 'train.csv')
        self.__save_df_to_csv(self.train_df, path)
        return valid

    @staticmethod
    def __save_df_to_csv(df, path):
        valid = False

        if path is not None and path != "":
            df.to_csv(path, '\t', index=False)
            valid = True
        return valid

    def load_train_test(self, dir_path):
        """
        Loads the SARC train and test set

        Filename of train predefined as 'train.csv' and test as 'test.csv'.
        Expects '\t' as separator for the saved CSV file.
        Assumes the CSV header is (post_id, post, reply, sarcasm)

        Parameters
        ----------
        dir_path: str
            The relative or absolute path of the directory
            where the data sets are stored.
        """
        dtype = {'post_id': np.str, 'post': np.str, 'reply': np.str, 'sarcasm': np.int8}
        test_path = os.path.join(dir_path, 'test.csv')
        train_path = os.path.join(dir_path, 'train.csv')
        self.test_df = pd.read_csv(test_path, sep='\t', keep_default_na=False
                                   , na_values="", dtype=dtype)
        self.train_df = pd.read_csv(train_path, sep='\t', keep_default_na=False
                                    , na_values="", dtype=dtype)

    @staticmethod
    def add_negative_label(df):
        """
        Adds a column 'not_sarcasm'.
        Parameters
        ----------
        df: DataFrame
            The df to which the additional column shall be added
        Returns
        -------
        DataFrame
            The df with the additional column 'not_sarcasm'
        """
        df['not_sarcasm'] = abs(df['sarcasm'] - 1)
        return df

    @staticmethod
    def shuffle_dataset(df, seed=3384322866, ordered_pairwise=False):
        """
        Shuffles SARC data randomly. Preserves pairs of posts.

        A new pandas.DataFrame gets created and the old one is
        overwritten by it! Preserves the shuffled index.
        The same posts with the sarcastic and serious answer each
        stay not only together, their internal order is also preserved

        Parameters
        ----------
        seed : int
            The seed for reproducibility
        df : DataFrame
            Should have the column 'post_id'
        ordered_pairwise: bool
            If in df the rows with the same post_id are always
            consecutive, then set this parameter to True.
            This method is around 70 times faster
        Returns
        -------
        DataFrame
            The randomly shuffled data, with index in an order
            which indicates the shuffling
        """
        np.random.seed(seed=seed)

        if ordered_pairwise:
            dim_1 = df.shape[0] // 2
            dim_3 = df.shape[1]
            index_values = df.index.values
            tmp_index = np.zeros_like(index_values)
            tmp_index[:] = index_values
            np.random.shuffle(tmp_index.reshape((dim_1, -1, 1)))

            # Because we want to retain the index and the values in the same order
            np.random.seed(seed=seed)
            ar = df.values
            np.random.shuffle(ar.reshape(dim_1, -1, dim_3))
            df = pd.DataFrame(ar, columns=['post_id', 'post', 'reply', 'sarcasm'])
            df['sarcasm'] = pd.to_numeric(df['sarcasm'], downcast='integer')
            df.set_index(tmp_index, inplace=True)
        else:
            grouped = df.groupby('post_id')
            post_pairs = [df for _, df in grouped]
            np.random.shuffle(post_pairs)
            df = pd.concat(post_pairs)
        return df

    @staticmethod
    def shuffle_post_pairs(df, seed=3384322866):
        """
        Shuffles internally the serious and sarcastic answers of a post pair

        It will only shuffle the two rows of a post each,
        while preserving the order of the posts compared to
        other posts. Reference of the DataFrame is not preserved.
        Execution time is quite long.

        Parameters
        ----------
        df : DataFrame
        seed : int
            The seed for reproducibility
        Returns
        -------
        DataFrame
            The data with the internally shuffled post pairs.
        """
        np.random.seed(seed=seed)
        post_id = df['post_id']
        df = df.groupby('post_id').transform(np.random.permutation)
        df['post_id'] = post_id
        return df

    def split_in_train_test(self, ordered_pairwise=True):
        """
        Splits the SARC data into 80/20 for testing and training

        Saves the sets also internally for further preprocessing.
        Uses NO deep copy for those, only references!
        The posts are kept in one set, meaning they stay together
        as post pair with the sarcastic and serious answer.
        It also shuffles the data beforehand with a hardcoded seed
        for reproducibility, but base set is not overwritten through that.

        Parameters
        ----------
        ordered_pairwise: bool
            If in the base DataFrame the rows with the same post_id are
            always consecutive, then set this parameter to True.
            This method is around 70 times faster
        """
        if self.data_df is None:
            raise ReferenceError("'data_df' is None, hasn't been loaded yet! "
                                 "Therefore can't split, because there is no data!")

        df = self.shuffle_dataset(self.data_df.copy(deep=True)
                                  , ordered_pairwise=ordered_pairwise)
        self.test_df, self.train_df = self.split_df_data(df)

    @staticmethod
    def _split_helper(percentage, data_count):
        if percentage > 1 or percentage < 0:
            raise ValueError("Parameter 'percentage' should be between 0 "
                             "and 1, but is {:.2f}".format(percentage))

        # Because we have to keep the post-pairs together in one split part
        split_index = int(round(percentage * (data_count + 1)))
        return split_index - split_index % 2

    def split_df_data(self, df, percentage=0.1):
        """
        Splits data of a DataFrame into two parts. Post-pairs stay together

        Parameters
        ----------
        df : DataFrame
            df doesn't get changed in this function
        percentage : float
            Defines the percentage of the smaller part.
            Should be a float between 0 and 1. Because of
            rounding issue due to binary representation it
            might to give exactly the expected split.

        Returns
        -------
        DataFrame, DataFrame
            Smaller part, bigger part
        """
        split_index = self._split_helper(percentage, df.shape[0])
        smaller, bigger = df.iloc[:split_index, :], df.iloc[split_index:, :]
        return smaller, bigger

    @staticmethod
    def load_word_vectors(vector_file, nr_row, nr_dim, ft_format=False):
        """
        Loads word vectors stored as space separated text.

        The file should be of the format:
        word1 <floating point dim1> <floating point dim2> ...
        word2 <floating point dim1> <floating point dim2> ...

        Parameters
        ----------
        vector_file: str
            Path to the text file with the word vectors
        nr_row: int
            The number of the word vectors in the text file
        nr_dim: int
            The dimension of the word vectors in the file
        ft_format: bool
            True if the word vector file is in the fastText format
            - the one which has as header row <no. vectors,vector dim>
        Returns
        -------
        np.ndarray, np.ndarray
            1. A Numpy array with the words in string form
            2. A Numpy array with the vector values as floats
        """
        rows_to_skip = 1 if ft_format else None

        # We don't keep default NaN, because specific words in the file
        # will be interpreted as NaN-values. We also set the quoting in
        # a way that the reading processing is more compatible with quotes
        word_vectors = pd.read_csv(vector_file, sep=' ', index_col=0
                                   , header=None, skiprows=rows_to_skip
                                   , keep_default_na=False, quoting=3)
        word_vectors.dropna(axis='index', inplace=True, how='any')
        word_vectors.dropna(axis='columns', inplace=True, how='all')
        vectors = word_vectors.values.astype('float32', copy=False)
        words = word_vectors.index.values.astype(str, copy=False)

        # Together with the NaN-check we can be quite sure about the
        # integrity of the word vector file
        assert nr_row == words.shape[0]
        assert nr_dim == vectors.shape[1]
        return words, vectors

    @staticmethod
    def load_word_vectors_row_wise(file_path, vec_count, vec_dim=None, ft_format=True):
        """
        Loads a word vector file row wise with standard Python code

        Parameters
        ----------
        file_path: str
        vec_count: int
            Number of word vectors - rows without eventual header -
            in the file
            Always obligatory, because the numbers in the word vector header
            can be wrong. Use command line tools like wc to count the lines
            minus the header line
        vec_dim: int
            Dimension of the word vectors
            (amount of numbers after a word in the file).
            Obligatory, if ``ft_format==False``
        ft_format: bool
            True, if the word vector file is in the fastText format.
            False else.

        Returns
        -------
        np.array, np.array
        Words as strings, vectors as 2D float32 array
        """
        word_vec_file = Path(file_path)
        if not word_vec_file.is_file():
            raise IOError("File {:} does not exist!".format(file_path))

        with open(file_path, 'rb') as file_:
            if ft_format:
                header = file_.readline()
                _, vec_dim = header.split()
                vec_dim = int(vec_dim)
            word_vectors = np.zeros((vec_count, vec_dim), dtype='float32')
            words = []
            for i, line in tqdm(enumerate(file_)):
                line = line.rstrip().decode('utf8')
                pieces = line.rsplit(' ', int(vec_dim))
                words.append(pieces[0])

                # Patching write error p.2383 in file crawl-300d-2M.vec:
                if i == 1791991:
                    for idx in range(1, len(pieces) - 1):
                        pieces[idx] = pieces[idx].replace('p', '0')
                word_vectors[i] = np.asarray(pieces[1:], dtype='float32')
        return words, word_vectors

    @staticmethod
    def conv_inner_to_tensor(array, tensor_type=torch.FloatTensor):
        """
        Converts a list-like of list-like to a list of Tensors

        Parameters
        ----------
        array: list, np.ndarray
            Containing list-like of compatible data types for FloatTensor
        tensor_type: torch.Tensor
            Which type of tensor it should be
        Returns
        -------
        list of torch.FloatTensor
            A list of FloatTensors which values and shape matches the array
            The rows (axis 0) are now FloatTensors, stored in a list object
        """
        return [tensor_type(a) for a in array]

    @staticmethod
    def col_to_dense_matrix(df, col, tfidf=False):
        """
        Creates a dense matrix with dimension m,n where m denotes the size of the types
        and n denotes the number of documents.

        Parameters
        ----------
        df: DataFrame
            df which holds the column with the corpus.
            Every row of the respective column must be a list of tokens
            (not a single string with white space)
        col: str
            the name of the column (of the df), which hold the corpus
        tfidf: bool
            True, if the values of the resulting matrix shall be the tf-idf values.
            False, if the values of the resulting matrix shall be the freq values

        Returns
        -------
        np.array
            a np.array with the resulting matrix
        """
        helpers.nested_list_test(df[col], test_for_str_items=True)
        d = corpora.Dictionary(df[col])
        corp = [d.doc2bow(el) for el in df[col]]
        if tfidf:
            corp = models.TfidfModel(corp)[corp]
        return gensim.matutils.corpus2dense(corp, num_terms=len(d))

    @staticmethod
    def split_np_data(array, percentage=(1 / 9), pairs=True):
        """
        Splits data of an array or list into two parts. Can preserve post-pairs

        Doesn't work very well with a very low ``percentage``
        for small array sizes. This is due to the fact, that
        the split indices always have to be integers.

        Parameters
        ----------
        array : list, np.ndarray
            The array-like variable you want to split.
            array doesn't get changed in this function
        percentage : float
            Defines the percentage of the smaller part.
            len(``array``) * ``percentage`` should not be smaller than 1
            Should be a float between 0 and 1. Because of
            rounding issue due to binary representation it
            might to give exactly the expected split.
        pairs: bool
            True, if you have an array with sarcasm+serious reply
            pairs following on each other.
            False, you don't have that, e.g. an array with singular
            posts, where you have to split it in the same way as the
            one with the reply.
        Returns
        -------
        DataFrame, DataFrame
            Smaller part, bigger part
        """
        if isinstance(array, list):
            length = len(array)
        else:
            length = array.shape[0]

        length = length if pairs else 2 * length
        split_index = DataHandler._split_helper(percentage, length)
        split_index = split_index if pairs else split_index // 2
        smaller, bigger = array[:split_index], array[split_index:]
        return smaller, bigger

    @staticmethod
    def cv_train_val_indices(array, n_folds=None, split_ratio=0.9, pairs=True):
        """
        Creates the indices for every fold of a cross validation.
        Parameters
        ----------
        array:
            The data for which the indices shall be created
        pairs: bool
            Ensures that row pairs (index 0,1; 2,3; ... n,n+1) are preserved
        n_folds: int
            The number of folds to return. If 'None' all folds are returned.
            Shouldn't be higher than the splits resulting through ´´split_ratio´´
        split_ratio: float
            The training ratio of the split train for each fold
            (the rest is for validation)
        Returns
        -------
        list of tuples
            Every element of the list represents one fold.
            Every element of the list is a tuple with of the form
            (<indices for train-data>, <indices for val-data>)
        Examples
        --------
        arr = np.array([(el, el) for el in range(100, 110)]).flatten()
        idx = dh.cv_train_val_indices(arr, pairs=True, n_splits = 9)
        for train_idx, val_idx in idx_0_0:
            train_data = arr[train_idx]
            val_data = arr[val_idx]
        """
        if isinstance(array, list):
            length = len(array)
        else:
            length = array.shape[0]
        split_fraction = 1 / (1. - split_ratio)
        no_of_splits = int(round(split_fraction, 0))
        if n_folds is not None and n_folds > no_of_splits:
            raise ValueError("n_folds can't be higher then the number of splits"
                             "resulting from the split_ratio! {:} > {:}".format(n_folds
                                                                                , no_of_splits))

        if pairs:
            groups = np.array([(el, el) for el in range(length // 2)]).flatten()

            kf = GroupKFold(n_splits=no_of_splits)
            indices = [(train, val) for train, val in kf.split(array, groups=groups)]
        else:
            kf = KFold(n_splits=no_of_splits)
            indices = [(train, val) for train, val in kf.split(array)]
        return indices[:n_folds]

    @staticmethod
    def conv_cv_idx_to_single(idx):
        def _to_single(arr):
            return np.array(arr[arr % 2 == 0] / 2, dtype=int)

        return [(_to_single(train_idx), _to_single(val_idx)) for train_idx, val_idx in idx]

    @staticmethod
    def shuffle_pair_matrix(*datasets, seed=42):
        """
        Shuffles matrices and lists. Ensures that all inputs are shuffled alike.

        Also ensures that row pairs (index 0,1; 2,3; ... n,n+1)
        are preserved for every input.

        Parameters
        ----------
        datasets: One or more np.array or scipy.sparse.csr_matrix
            The datasets, that shall be shuffled.
            All datasets must have the same length
        seed: int
            See for Numpy random functions
        Returns
        -------
        tuple
            shuffled datasets with preserved row pairs
        Examples
        --------
        >>> np_train = np.array([[0,0,0],[0,0,0],[1,1,1],[1,1,1]
        >>>                      ,[2,2,2],[2,2,2],[3,3,3],[3,3,3]])
        >>> labels = np.array([0,1,1,0,1,0,1,0])
        >>> np_train_shuffled, labels_shuffled = DataHandler.shuffle_pair_matrix(np_train, labels)
        >>> print(shuffled.todense())
        >>> # Output of print:
        >>> # matrix([[3, 3, 3],
        >>> # [3, 3, 3],
        >>> # [2, 2, 2],
        >>> # [2, 2, 2],
        >>> # [0, 0, 0],
        >>> # [0, 0, 0],
        >>> # [1, 1, 1],
        >>> # [1, 1, 1]], dtype=int64)
        """

        def __apply_idx(doc, idx_0, idx_1):
            if doc.shape[0] == len(idx_0):
                doc = doc[idx_0,]
            elif doc.shape[0] == len(idx_1):
                doc = doc[idx_1,]
            else:
                raise ValueError('Given this input the documents must either '
                                 'be of length %i or %i' % (len(idx_0), len(idx_1)))
            return doc

        result = None
        if len(datasets) > 0:
            np.random.seed(seed)
            longest_doc = helpers.longest_doc(*datasets)
            aux_0 = np.arange(0, longest_doc.shape[0], 2)
            np.random.shuffle(aux_0)
            aux_1 = aux_0 + 1
            idx_0 = np.stack((aux_0, aux_1)).T.flatten()
            idx_1 = np.array(aux_0 / 2, dtype=int)
            result = tuple(__apply_idx(doc, idx_0, idx_1) for doc in datasets)
        return result
