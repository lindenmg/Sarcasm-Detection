import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import regex
from scipy.sparse import csr_matrix

from src.preprocessing.datahandler import DataHandler as DH
from src.tools.helpers import flatten


def create_data_path():
    path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
    return os.path.join(path, 'test_data')


def prepare_data_loading_test(file1, file2):
    data_folder = create_data_path()
    comments_file = os.path.join(data_folder, file1)
    annotations_file = os.path.join(data_folder, file2)
    pp = DH()
    pp.load_data(comments_file, annotations_file)
    return pp


gl_datahandler = prepare_data_loading_test('comments_cleaned.txt', 'annotation.txt')


class TestPreprocessing:

    def test_col_to_dense_matrix(self):
        df = pd.DataFrame({'foo': [
            ['this', 'is', 'really', 'really', 'a', 'document'],
            ['this', 'is', 'another'],
            ['completely', 'new']]})
        m = DH.col_to_dense_matrix(df, col='foo', tfidf=True)

        assert m.shape == (8, 3)
        assert np.isclose(m[0, 0], 0.3992843)
        assert np.isclose(m[7, 2], 0.70710677)

        raised = False
        df_wrong = pd.DataFrame({'foo': [
            'this string should first be split',
            'this string also'
        ]})
        try:
            DH.col_to_dense_matrix(df_wrong, col='foo')
        except TypeError:
            raised = True
        assert raised

    def test_shuffle_pair_matrix(self):
        val_err_msg = 'Given this input the documents must ' \
                      'either be of length %i or %i'
        csr_res_1 = np.array([[1, 1, 1], [1, 1, 1], [3, 3, 3], [3, 3, 3],
                              [0, 0, 0], [0, 0, 0], [2, 2, 2], [2, 2, 2]])
        csr_m_1 = csr_matrix([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1],
                              [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]])

        csr_res_2 = csr_matrix([[1, 1, 1], [3, 3, 3], [0, 0, 0], [2, 2, 2]])
        csr_m_2 = csr_matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        labels_res = np.array([1, 0, 1, 0, 0, 1, 1, 0])

        csr_sh_1, csr__2, labels_sh = DH.shuffle_pair_matrix(csr_m_1, csr_m_2, labels, seed=42)
        assert (csr_sh_1.todense() == csr_res_1).all()
        assert (csr__2.todense() == csr_res_2).all()

        np_m = csr_m_1.todense()
        np_sh, labels_sh = DH.shuffle_pair_matrix(np_m, labels, seed=42)
        assert (np_sh == csr_res_1).all()
        assert (labels_res == labels_sh).all()

        csr_m_3 = csr_matrix([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])

        with pytest.raises(ValueError) as val_err:
            _, _, _ = DH.shuffle_pair_matrix(csr__2, csr_m_3)
        assert (val_err_msg % (6, 3)) in str(val_err.value)

    def test_cv_train_val_indices_i(self):
        arr_0 = np.arange(100, 110)
        idx_0 = DH.cv_train_val_indices(arr_0, pairs=False, split_ratio=0.9)
        assert len(idx_0[0][1]) == 1
        assert len(idx_0[1][1]) == 1
        assert len(idx_0[2][1]) == 1

        arr_1 = flatten([(el, el) for el in range(100, 110)])
        arr_2 = np.array(arr_1)
        idx_2 = DH.cv_train_val_indices(arr_2, pairs=True, split_ratio=0.9)
        for train_idx, val_idx in idx_2:
            assert len(val_idx) % 2 == 0
            assert len(train_idx) % 2 == 0
        [arr_0[train_idx] for train_idx, val_idx in idx_0][0]
        [arr_2[train_idx] for train_idx, val_idx in idx_2][0]
        arr_3 = np.append(arr_2, 110)
        with pytest.raises(ValueError) as err:
            _ = DH.cv_train_val_indices(arr_3, pairs=True, split_ratio=0.9)
        assert err is not None
        with pytest.raises(ValueError) as val_err:
            _ = DH.cv_train_val_indices(arr_2, pairs=True, n_folds=11, split_ratio=0.9)
        assert val_err is not None

    def test_cv_train_val_indices_ii(self):
        float64_split_ratio = 0.888888888888888
        replies = np.random.random(size=1000)
        posts = np.random.random(size=500)
        idx_r = DH.cv_train_val_indices(replies, pairs=True, split_ratio=float64_split_ratio)
        idx_p = DH.cv_train_val_indices(posts, pairs=False, split_ratio=float64_split_ratio)
        assert len(idx_r) == 9
        assert len(idx_p) == 9
        idx_r_len = []
        idx_p_len = []
        for train_idx, val_idx in idx_r:
            idx_r_len.append(len(train_idx))
            assert not np.isin(train_idx, val_idx).all()
            assert round(len(train_idx) / len(val_idx)) == 8
        for train_idx, val_idx in idx_p:
            idx_p_len.append(len(train_idx))
            assert not np.isin(train_idx, val_idx).all()
            assert round(len(train_idx) / len(val_idx)) == 8
        for len_r, len_p in zip(idx_r_len, idx_p_len):
            assert abs(len_r - 888) < 3
            assert abs(len_p - 444) < 3
        for i in range(9):
            for j in range(9):
                if i != j:
                    assert not np.isin(idx_r[i][1], idx_r[j][1]).all()
                    assert not np.isin(idx_p[i][1], idx_p[j][1]).all()

    def test_cv_train_val_indices_iii(self):
        float64_split_ratio = 0.9
        replies = np.random.random(size=1000)
        posts = np.random.random(size=500)
        idx_r = DH.cv_train_val_indices(replies, pairs=True, split_ratio=float64_split_ratio)
        idx_p = DH.cv_train_val_indices(posts, pairs=False, split_ratio=float64_split_ratio)
        assert len(idx_r) == 10
        assert len(idx_p) == 10
        idx_r_len = []
        idx_p_len = []
        for train_idx, val_idx in idx_r:
            idx_r_len.append(len(train_idx))
            assert not np.isin(train_idx, val_idx).all()
            assert round(len(train_idx) / len(val_idx)) == 9
        for train_idx, val_idx in idx_p:
            idx_p_len.append(len(train_idx))
            assert not np.isin(train_idx, val_idx).all()
            assert round(len(train_idx) / len(val_idx)) == 9
        for len_r, len_p in zip(idx_r_len, idx_p_len):
            assert abs(len_r - 900) < 2
            assert abs(len_p - 450) < 2
        for i in range(10):
            for j in range(10):
                if i != j:
                    assert not np.isin(idx_r[i][1], idx_r[j][1]).all()
                    assert not np.isin(idx_p[i][1], idx_p[j][1]).all()

    def test_convert_cv_fold_train_val_indices_to_single(self):
        arr_single = np.array(range(100, 110))
        arr_pairs = np.array([(el, el) for el in arr_single]).flatten()
        idx_pairs = DH.cv_train_val_indices(arr_pairs, pairs=True, n_folds=7)
        idx_single = DH.conv_cv_idx_to_single(idx_pairs)
        for (s_train, s_val), (p_train, p_val) in zip(idx_single, idx_pairs):
            assert len(s_train) * 2 == len(p_train)
            assert len(s_val) * 2 == len(p_val)
            for i, v in enumerate(s_train):
                assert v * 2 == p_train[i * 2]

    def test_load_data_i(self):
        datahandler = prepare_data_loading_test('comments_cleaned.txt', 'annotation.txt')
        data_df = datahandler.get_data_df(deep_copy=False)
        assert data_df.shape == (218362, 4)

        _5womoy = "OP doesn't understand why people choose to work during college"
        _62p5z8 = "TIL that in 41 states, parents without high school diplomas or " \
                  "GEDs are permitted to homeschool their children"
        _7uaac = "I've been searching for the answer for this for some time, but I " \
                 "still can't find any answer... Can anyone please explain to me what this is?"
        _54f79l = "Caveira got a new buff"
        result_df = data_df[data_df['post'].isin([_5womoy, _62p5z8, _7uaac, _54f79l])]
        result_df.reset_index(drop=True, inplace=True)
        test_file = os.path.join(create_data_path(), 'sarc_loading_test.csv')
        test_df = pd.read_csv(test_file, sep='\t', dtype={'sarcasm': np.int8})
        assert result_df.equals(test_df)

    def test_get_data_df_i(self):
        daahandler = prepare_data_loading_test('comments_cleaned.txt', 'annotation.txt')
        data_df = daahandler.get_data_df(deep_copy=False)
        data_df.drop([0, 101746, 218361], inplace=True)
        assert data_df.equals(daahandler.get_data_df())

        data_df = daahandler.get_data_df(deep_copy=False)
        data_df.drop('post', axis=1, inplace=True)
        assert data_df.equals(daahandler.get_data_df(deep_copy=True))

    def test_get_data_df_ii(self):
        datahandler = prepare_data_loading_test('comments_cleaned.txt', 'annotation.txt')
        data_df = datahandler.get_data_df(deep_copy=True)
        data_df.drop([0, 101746, 218361], inplace=True)
        assert not data_df.equals(datahandler.get_data_df(deep_copy=True))

        data_df = datahandler.get_data_df(deep_copy=True)
        data_df.drop('post', axis=1, inplace=True)
        assert not data_df.equals(datahandler.get_data_df())

    def test_get_data_df_iii(self):
        datahandler = DH()
        message = "'data_df' is None, hasn't been loaded yet!"
        with pytest.raises(ReferenceError) as ref_err:
            datahandler.get_data_df()
        assert message in str(ref_err.value)

    def test_set_data_df_i(self):
        datahandler = DH()
        dir_path = create_data_path()
        df = pd.read_csv(os.path.join(dir_path, 'annotation.txt'), sep='\t')
        datahandler.set_data_df(df, deep_copy=False)
        df.drop([0, 101746, 218361], inplace=True)
        df.drop('post_id', axis=1, inplace=True)
        assert df.equals(datahandler.get_data_df())
        assert df.equals(datahandler.get_data_df(deep_copy=True))

    def test_set_data_df_ii(self):
        datahandler = DH()
        dir_path = create_data_path()
        df = pd.read_csv(os.path.join(dir_path, 'annotation.txt'), sep='\t')
        datahandler.set_data_df(df, deep_copy=True)
        df.drop([0, 101746, 218361], inplace=True)
        df.drop('post_id', axis=1, inplace=True)
        assert not df.equals(datahandler.get_data_df(deep_copy=True))
        assert not df.equals(datahandler.get_data_df())

    def test_save_data_to_csv(self):
        datahandler = prepare_data_loading_test('comments_cleaned.txt', 'annotation.txt')
        data_df = datahandler.get_data_df(deep_copy=False)
        dir_path = create_data_path()
        csv_path = os.path.join(dir_path, 'data_df.csv')
        assert datahandler.save_dataset_to_csv(csv_path)
        test_df = pd.read_csv(csv_path, sep='\t', keep_default_na=False, na_values=""
                              , encoding='us-ascii', dtype={'post_id': np.str
                , 'post': np.str, 'reply': np.str, 'sarcasm': np.int8})
        os.remove(csv_path)
        assert data_df.equals(test_df)

    def test_split_data_i(self):
        datahandler = DH()
        message = "Parameter 'percentage' should be between 0 and 1, but is {:.2f}"
        number = -0.0000001
        df = pd.DataFrame()
        with pytest.raises(ValueError) as ref_err:
            datahandler.split_df_data(df, number)
        assert message.format(number) in str(ref_err.value)

        number = 1.00001
        df = pd.DataFrame()
        with pytest.raises(ValueError) as ref_err:
            datahandler.split_df_data(df, number)
        assert message.format(number) in str(ref_err.value)

    @staticmethod
    def __get_split_dataset_size__(size, split, pairs=True):
        size = size if pairs else 2 * size
        test_size = int(round((size + 1) * split))
        test_size = test_size - test_size % 2
        test_size = test_size if pairs else test_size // 2
        return test_size

    def __intern_split_test_helper(self, train, test, orig_length
                                   , split, columns, pairs=True):
        test_size = self.__get_split_dataset_size__(orig_length, split, pairs)
        train_size = self.__get_split_dataset_size__(orig_length, (1 - split), pairs)

        if len(test.shape) > 1 and len(train.shape) > 1:
            assert test.shape[1] == columns and train.shape[1] == columns
        assert test.shape[0] == test_size
        assert train.shape[0] == train_size
        if pairs:
            assert train_size % 2 == 0
            assert test_size % 2 == 0

    def test_split_data_ii(self):
        datahandler = gl_datahandler
        df = datahandler.get_data_df()
        test, train = datahandler.split_df_data(df, (1 / 3))
        self.__intern_split_test_helper(train, test, df.shape[0], (1 / 3), 4)
        assert pd.concat([test, train]).reset_index(drop=True).equals(df)
        assert test[test.isin(train.pop('post_id'))].dropna().shape == (0, 4)

    def test_np_data_i(self):
        datahandler = gl_datahandler
        df = datahandler.get_data_df()
        posts = df["post"].values[0::2]
        replies = df["reply"].values
        test, train = DH.split_np_data(posts, percentage=(1 / 9), pairs=False)
        self.__intern_split_test_helper(train, test, posts.shape[0]
                                        , (1 / 9), 1, pairs=False)
        assert (np.concatenate([test, train], axis=0) == posts).all()
        test, train = DH.split_np_data(replies, percentage=0.1)
        self.__intern_split_test_helper(train, test, replies.shape[0], 0.1, 1)
        assert (np.concatenate([test, train], axis=0) == replies).all()

    def __shuffle_split_test__(self, preprocess, df_before, test, train):
        df_after = preprocess.get_data_df(deep_copy=True)
        df_sorted = pd.concat([test, train]).sort_index()
        test_size = self.__get_split_dataset_size__(df_before.shape[0], 0.1)
        train_size = self.__get_split_dataset_size__(df_before.shape[0], 0.9)

        assert df_before.equals(df_after)
        assert test.shape[1] == 4 and train.shape[1] == 4
        assert test.shape[0] == test_size
        assert train.shape[0] == train_size
        assert train_size % 2 == 0
        assert test_size % 2 == 0
        assert (df_before['post_id'] == df_sorted['post_id']).all()
        assert df_before.equals(df_sorted)
        assert test[test.isin(train.pop('post_id'))].dropna().shape == (0, 4)

    def test_split_in_train_test(self):
        datahandler = gl_datahandler
        df_before = datahandler.get_data_df(deep_copy=True)
        datahandler.split_in_train_test()
        test = datahandler.get_test_df(False)
        train = datahandler.get_train_df(False)
        self.__shuffle_split_test__(datahandler, df_before, test, train)

        datahandler.split_in_train_test(ordered_pairwise=False)
        test = datahandler.get_test_df(False)
        train = datahandler.get_train_df(False)
        self.__shuffle_split_test__(datahandler, df_before, test, train)

    def test_save_load_train_test_to_csv(self):
        datahandler = prepare_data_loading_test('comments_cleaned.txt', 'annotation.txt')
        data_df = datahandler.get_data_df(deep_copy=False)
        dir_path = create_data_path()
        datahandler.split_in_train_test()
        test = datahandler.get_test_df(True)
        train = datahandler.get_train_df(True)
        assert datahandler.save_train_test_to_csv(dir_path)

        test_path = os.path.join(dir_path, 'test.csv')
        train_path = os.path.join(dir_path, 'train.csv')
        datahandler.load_train_test(dir_path)
        test_test = datahandler.get_test_df(False)
        train_test = datahandler.get_train_df(False)
        os.remove(test_path)
        os.remove(train_path)
        test.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)
        assert test_test.equals(test) and train_test.equals(train)

    def test_getter_train_test_df(self):
        datahandler = gl_datahandler
        df_before = datahandler.get_data_df(deep_copy=True)
        datahandler.set_train_df(None, deep_copy=False)
        train = datahandler.get_train_df()
        datahandler.set_test_df(None, deep_copy=False)
        test = datahandler.get_test_df()
        self.__shuffle_split_test__(datahandler, df_before, test, train)

    def test_shuffle_post_pairs(self):
        datahandle = gl_datahandler
        df_before = datahandle.get_data_df(deep_copy=True)
        df_after = datahandle.shuffle_post_pairs(df_before.copy(deep=True)).copy(deep=True)
        values_after = df_after['post_id'].values
        values_before = df_before['post_id'].values
        values_compared = (values_after == values_before)
        assert df_before.shape == df_after.shape
        assert values_compared.all()

        compare_df = df_before[df_before.isin(df_after.pop('post_id'))]
        assert compare_df.dropna().shape == (0, 4)

    def test_load_word_vectors(self):
        datahandle = gl_datahandler
        vector_file = os.path.join(create_data_path(), "test_vecs.txt")
        vector_file_ft = os.path.join(create_data_path(), "test_ft_vecs.txt")
        np.random.seed(1337)
        vectors = np.random.normal(0, 1, (1000, 300)).astype('float32').round(4)
        words = np.random.randint(int(1e4), int(1e6), (1000, 1)).astype(int)
        if not Path(vector_file).is_file():
            array = np.concatenate([words, vectors], axis=1)
            np.savetxt(vector_file, array, fmt='%.4f')
        if not Path(vector_file_ft).is_file():
            header = "1000 300"
            array_ft = np.concatenate([words, vectors], axis=1)
            np.savetxt(vector_file_ft, array_ft, fmt='%.4f')

            with open(vector_file_ft, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header.rstrip('\r\n') + '\n' + content)
        words_std, vectors_std = datahandle.load_word_vectors(vector_file, 1000, 300)
        words_ft, vectors_ft = datahandle.load_word_vectors(vector_file_ft, 1000, 300, ft_format=True)
        words_str = np.zeros_like(words_std)
        words_str[:] = words_std
        words_std = np.reshape(words_std.astype(float).astype(int), (1000, 1))
        words_ft = np.reshape(words_ft.astype(float).astype(int), (1000, 1))
        assert vectors_std.dtype == 'float32'
        assert words_str.dtype == np.dtype('<U32')
        assert (words == words_std).all()
        assert (vectors == vectors_std).all()
        assert (words == words_ft).all()
        assert (vectors == vectors_ft).all()

        assert not np.asarray([(regex.search(r'\s', str(w))) for w in words_str]).all()

    def test_convert_np_to_tensor(self):
        datahandle = gl_datahandler
        array = np.random.random((9999, 300)).astype('float32')
        tensors = datahandle.conv_inner_to_tensor(array)
        checkarray = np.asarray([(a == t.numpy().astype('float32')).all()
                                 for a, t in zip(array, tensors)])
        assert checkarray.all()

        array = np.random.randint(-1000, 1000, (9973, 117)).astype(np.int64)
        tensors = datahandle.conv_inner_to_tensor(array)
        checkarray = np.asarray([(a == t.numpy().astype(np.int64)).all()
                                 for a, t in zip(array, tensors)])
        assert checkarray.all()

    def test_split_np_data_ii(self):
        iters = 100
        randint = np.random.randint
        array_size = randint(97, 10003, size=(iters,)).tolist()
        percentage = np.random.random(iters).tolist()
        for s, p in zip(array_size, percentage):
            posts = randint(0, 100, size=s)
            replies = randint(0, 100, size=int(2 * s))
            post_s, post_b = DH.split_np_data(posts, percentage=p, pairs=False)
            reply_s, reply_b = DH.split_np_data(replies, percentage=p, pairs=True)
            assert len(post_s) * 2 == len(reply_s)
            assert len(post_b) * 2 == len(reply_b)
            assert len(post_s) + len(post_b) == len(posts)
            assert len(reply_s) + len(reply_b) == len(replies)
            assert (np.concatenate([post_s, post_b]) == posts).all()
            assert (np.concatenate([reply_s, reply_b]) == replies).all()
