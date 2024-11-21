import numpy as np
from pathlib import Path
import pandas as pd

from src.preprocessing.datahandler import DataHandler
from src.tools.config import Config


class TestRewriteWordVectors:
    root_path = Path(Config.path.test_data_folder)

    def test_rewrite_i(self):
        wv_path = self.root_path / "test_vector.csv"
        wv_write_path = self.root_path / "test_vector_write.csv"
        dh = DataHandler()
        words, word_vectors = dh.load_word_vectors_row_wise(str(wv_path), 6, 300)
        df = pd.DataFrame(data=word_vectors, index=words)
        df.to_csv(str(wv_write_path), sep=' ', header=False, float_format='%5.4f'
                  , quotechar="", quoting=3)
        assert True

    def test_rewrite_ii(self):
        words = np.random.randint(0, 10000, size=(1000,)).astype(dtype=str)
        vectors = np.random.rand(1000, 300).astype(dtype='float32')
        wv_write_path = self.root_path / "test_vector_write_ii.csv"
        dh = DataHandler()

        df = pd.DataFrame(data=vectors, index=words, copy=True)
        df.to_csv(str(wv_write_path), sep=' ', header=False, float_format='%5.4f'
                  , quotechar="", quoting=3)
        words_ld, word_vectors_ld = dh.load_word_vectors_row_wise(str(wv_write_path)
                                                                  , 1000, 300, ft_format=False)
        assert (words == words_ld).all()
        assert (np.abs(vectors - word_vectors_ld) <= 6e-5).all()
