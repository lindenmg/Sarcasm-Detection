import os
import sys

project_root_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(project_root_path)

import argparse
from pathlib import Path

import pandas as pd
import src.tools.helpers as helpers
from src.tools.config import Config
from src.preprocessing.datahandler import DataHandler


def get_attribute(obj, name):
    attr = getattr(obj, name)
    return attr[0] if isinstance(attr, list) else attr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word vector rewrite without header')
    parser.add_argument('-w', '--original_vectors', type=str, nargs=1
                        , default="word_vectors/fastText/crawl-300d-2M.vec",
                        help='The relative path to the word vector file '
                             'from the project data folder')
    parser.add_argument('-r', '--rewrite_destination', type=str, nargs=1
                        , default="word_vectors/fastText/ft_2M_300.csv",
                        help='The relative path to the rewritten word vector '
                             'file from the project data folder')
    parser.add_argument('-c', '--word_count', type=int, nargs=1
                        , default=1999995,
                        help='The count of the words in the word vector file - '
                             'without probably existing header')
    parser.add_argument('-d', '--vector_dim', type=int, nargs=1
                        , default=300,
                        help='The dimension (amount of numbers in line after word) '
                             'of the vectors in the word vector file')
    parser.add_argument('-f', '--ft_format', type=bool, nargs=1
                        , default=True,
                        help='Choose ", if it has a header line and 0, else')

    helpers.section_text("START of rewrite")
    root_path = Path(Config.path.data_folder)
    args = parser.parse_args()
    wv_file = get_attribute(args, 'original_vectors')
    rewrite_file = get_attribute(args, 'rewrite_destination')
    word_count = get_attribute(args, 'word_count')
    wv_dim = get_attribute(args, 'vector_dim')
    ft_format = get_attribute(args, 'ft_format')

    wv_path = root_path / wv_file
    wv_write_path = root_path / rewrite_file
    dh = DataHandler()

    print("Load word vector file row per row...")
    words, word_vectors = dh.load_word_vectors_row_wise(str(wv_path), word_count
                                                        , wv_dim, ft_format)
    print("Start rewriting, this may take a very long time...")
    df = pd.DataFrame(data=word_vectors, index=words)
    df.to_csv(str(wv_write_path), sep=' ', header=False, float_format='%5.4f'
              , quotechar="", quoting=3)
    helpers.section_text("FINISHED")
