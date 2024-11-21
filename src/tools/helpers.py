import collections
import hashlib
import json
import os
import pickle
import shutil
import time
from functools import reduce
from pathlib import Path

import numpy as np
import torch

from src.tools.config import Config


def save_json_to_disk(dictionary, file_path, ensure_ascii=True, sort_keys=True):
    """
    Saves dict to disk. Creates the path to the file,
    if path doesn't exist.

    Parameters
    ----------
    dictionary : dict
    file_path : str
        Where you want to save the json
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, ensure_ascii=ensure_ascii,
                  sort_keys=sort_keys)


def save_to_disk(serializable, file_path):
    """
    Saves objects to disk that are compatible with pickle.

    Compatible are among others: (list of) SpaCy Docs,
    (list of) vanilla python data types, Numpy arrays

    Parameters
    ----------
    serializable : object
    file_path : str
        Where you want to save the serializable object(s)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(serializable, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_disk(file_path):
    """
    Loads a with pickle serialized object from disk

    Parameters
    ----------
    file_path : str
        Form where you want to load the serialized object(s)
    Returns
    -------
    list of spacy.tokens.doc.Doc
        The loaded object(s) from the given file path
    """
    with open(file_path, 'rb') as file:
        deserializable = pickle.load(file)
    return deserializable


def nested_list_test(list_of_lists, tests=5, test_for_str_items=False):
    """
    Tests if the list contains lists and no further third layer/dimension of lists.

    Parameters
    ----------
    list_of_lists : list of list
    tests : int
        Number of randomly picked indexes to test for
    test_for_str_items : bool
        True - the list gets tested if it contains lists of strings (str)
    Raises
    -------
    TypeError
        If it doesn't match the criteria
    """

    def list_contains_lists(list_):
        return np.asarray([isinstance(element, list) for element in list_]).all()

    def list_contains_strings(list_):
        return np.asarray([isinstance(element, str) for element in list_]).all()

    length = len(list_of_lists)
    additional_msg = ""

    if test_for_str_items:
        additional_msg = " And there should be lists of strings in the main list!"

    # Not entirely fail-proof of course
    for _ in range(tests):
        idx = np.random.randint(0, length)
        if not isinstance(list_of_lists[idx], list) \
                or list_contains_lists(list_of_lists[idx]) \
                or (test_for_str_items and not list_contains_strings(list_of_lists[idx])):
            raise TypeError("The rows of the target column must be lists of lists. "
                            "There also shouldn't be a list of lists of lists." + additional_msg)


def flatten(list_of_lists):
    """
    Flattens a list which contains lists to a list

    Parameters
    ----------
    list_of_lists : list of list
    Returns
    -------
    list
        contains the Variables that have been in the nested inner lists
    """
    return [element for list_ in list_of_lists for element in list_]


def idx_lookup_from_list(list_, default_dict=False):
    """
    Creates a dict with the list elements as key and their index as value

    Parameters
    ----------
    list_: list
    default_dict:bool
        If it shall construct a dict which returns 0
        if the key does not exist in the dictionary

    Returns
    -------
    dict or collections.defaultdict

    Raises
    ------
    Warning, when the values in list_ are not unique
    """
    if default_dict:
        look_up = collections.defaultdict(_default_unk_index)
    else:
        look_up = {}

    len_list = len(list_)
    len_unique = len(list(set(list_)))
    if len_list != len_unique:
        raise Warning("The values in list_ are not unique!")

    for i, w in enumerate(list_):
        look_up[w] = i
    return look_up


def _default_unk_index():
    return 0


def create_length_tensor(data_list):
    """
    Creates a torch.LongTensor with the lengths of objects in ``data_list``

    For example [np.arange(10), np.random.randint(size=(50,4))]
    returns tensor with values [10, 50]

    Parameters
    ----------
    data_list: list
        Should be list of np.ndarray or torch.Tensor or list-like
    Returns
    -------
    torch.LongTensor
        Containing the lengths of the data examples in data_list
    """
    if not (isinstance(data_list, list)
            or isinstance(data_list, tuple)):
        raise TypeError("data_list should be list or tuple, "
                        "but is {}".format(type(data_list)))
    tensor = torch.LongTensor(len(data_list))

    for i, example in enumerate(data_list):
        tensor[i] = len(example)
    return tensor


def get_best_batch_size(data_length, batch_size, window=0.1, residual=None):
    """
    Calculates the batch size with the smallest left over batch

    When you choose a certain batch size you get data-length // batch-size
    batches per epoch. Plus this one extra batch if
    data-length % batch-size != 0 - if you don't drop it per default.
    To get a batch-size near the favorite one you have to look out for
    the one which gives you the smallest residual value after an integer
    division. This method does exactly that.

    Parameters
    ----------
    data_length: int
        The length of the data you want to find a batch size for
    batch_size: int
        The initial batch-size you prefer (the most)
    window: float
        The window around ``batch_size`` you want to search in.
        In percent of ``batch_size``
    residual: float or None
        If you don't want to use this, don't set it.
        The minimum allowed size of the residual batch in percent
        of ``batch_size``
    Returns
    -------
    int, bool
        A tuple of (best-batch-size, is-bigger-or-equal-then-residual)
        Second part of the tuple determines if the found size is within
        the set criteria for the smallest allowed residual batch
    """
    # CHECK (sanity checks)
    if not isinstance(data_length, int) or not isinstance(batch_size, int):
        raise TypeError("type(data_length) == {} and type(batch_size) == {} "
                        "but should both be int".format(type(data_length)
                                                        , type(batch_size)))
    if residual is not None and (residual > 1 or residual < 0):
        raise ValueError("residual should be between 1.0 "
                         "and 0 but is {}".format(residual))
    if window is None:
        raise ValueError("window is None, but should be between 0 & 1!")
    if window < 0:
        raise ValueError("window is not allow to be negative!")
    if window >= 1:
        raise ValueError("window is >= 1. That makes no sense, "
                         "just choose another base batch_size!")
    if data_length // batch_size < 2:
        raise ValueError("You get currently with {} not even 2 full"
                         "batches per epoch".format(batch_size))
    if data_length < 0:
        raise ValueError("data_length should be greater then 0!, "
                         "but is {}".format(data_length))

    # CALCULATE best batch size
    found = True
    win = int(round(window * batch_size))
    test_range = np.arange(batch_size - win, batch_size + win + 1)
    rest_range = data_length % test_range
    best_batch = batch_size - win + rest_range.argmin()
    if residual is not None:
        residual = 1 - residual
        rest = int(round(residual * batch_size))
        found = False if rest_range.min() < rest else True
    return int(best_batch), found


def pad_tensor(list_of_tensors, data_dim):
    """
    Pads a list of tensors to same length

    Parameters
    ----------
    list_of_tensors: list of torch.LongTensor
        Contains tensors with different length
    data_dim: int
        The length the data shall be padded to

    Returns
    -------
    torch.LongTensor
        Two-dimensional tensor with 1st dim == sentences,
        2nd dim = embedding dim of indices. Padding value is 0
    """
    tensor = torch.LongTensor(len(list_of_tensors), data_dim).zero_()
    for i, idx_t in enumerate(list_of_tensors):
        end = len(idx_t)
        if end > 0:
            tensor[i][0:end] = idx_t[0:data_dim]
    return tensor


def stop_time(start_time):
    """
    Stops the time between start_time and the current time.

    Parameters
    ----------
    start_time : posix.times_result
        The start time gotten from os.times()
    Returns
    -------
    float, posix.times_result
        - time_elapsed: The seconds plus hundredth seconds since time of start_time
        - start_time: The new start time for the time of the next section
    """
    t_end = os.times()
    time_elapsed = t_end.elapsed - start_time.elapsed
    start_time = os.times()
    return time_elapsed, start_time


def section_text(text, upper=True, show_time=True):
    """
    Displays a heading with optional display of last sections runtime.

    Before the execution a global variable named t_start needs to be
    initialised with os.times().

    Parameters
    ----------
    text : str
        The text which will be displayed in the heading
    upper : bool
        Determines, if the heading shall be converted to upper case
        letters. True by default.
    show_time : bool
        Whether the time of last section shall be shown.
    Examples
    --------
    >>> import os
    >>> t_start = os.times()
    >>> text = "This is a new section"
    >>> section_text(text)
          Runtime of last section: 0.0s
    =====  THIS IS A NEW SECTION  ==============================================
     *(The heading would have 4 more '=' at the end)*
    """
    if Config.debug_mode:
        if 't_start' not in globals().keys():
            global t_start
            t_start = os.times()
        time_elapsed, t_start = stop_time(t_start)

        if show_time:
            print("\n\n\tRuntime of last section: {:,.1f}s".format(time_elapsed))
        text = text.upper() if upper else text
        right_delimiter = "=" * (80 - len(text) - 9)
        print("======  " + text + "  " + right_delimiter + "\n")


def get_time_duration():
    """
    Creates a string with the seconds between the time of the first execution
    of this function, and the current time. (Introduces the global variable 'time_start')
    Returns
    -------
    str:
        a string with the format '%.1f secs' for the duration
    """
    if 'time_start' not in globals().keys():
        global time_start
        time_start = time.time()
    return "%.1f secs" % (time.time() - time_start)


def scipy_csr_to_sparse_tensor(csr):
    """
    Not used yet, torch.sparse doesn't seem to be applicable yet...
    Parameters
    ----------
    csr: scipy.csr.csr_matrix
        matrix that shall be created
    Returns
    -------
    torch.sparse.FloatTensor
        A sparse torch vector with the data from the parameter 'csr'
    """
    csr = csr.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((csr.row, csr.col))).long()
    values = torch.from_numpy(csr.data)
    shape = torch.Size(csr.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def eventually_load_params(net, path):
    """
    Eventually loads pretrained net-parameters into a pytorch net.
    If the parameters are not provided in 'path' nothing happens (without error).
    Parameters
    ----------
    net: torch.nn.Module
    path

    Returns
    -------

    """
    if isinstance(path, str) and Path(path).is_file():
        print('load params from: %s' % path)
        net.load_state_dict(torch.load(path))
    return net


def tensorboard_log(writer, epoch, fold, learning_rate, train_loss
                    , val_loss, train_acc, val_acc, should_print=True):
    """
        Adds the scalar values (epoch, fold, learning_rate etc.) to the tensorboard writer.
        Eventually also prints to the console
    Parameters
    ----------
    writer: tensorboardX.SummaryWriter
        The writer to which the values shall be added
    should_print: bool
        True to also log the values (epoch, fold, etc.) to console
    """
    if should_print:
        print(
            '[time: %5s | fold: %2d | epoch: %2d], lr: %1.2e, train_loss: %.4f, '
            'train_acc: %.4f, val_loss: %.4f, val_acc: %.4f' % (
                get_time_duration(), fold, epoch, learning_rate, train_loss
                , train_acc, val_loss, val_acc))

    writer.add_scalars('learning_rate', {'learning_rate': learning_rate}, epoch)
    writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
    writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)


def longest_doc(*datasets):
    """
    From a list of documents, get the document with the greatest length
    Parameters
    ----------
    datasets: list
        the list of documents from which to choose the longest one

    Returns
    -------
        the longest document
    """
    return reduce(lambda acc, doc: doc if doc.shape[0] > acc.shape[0] else acc, datasets)


def clean_folder(path, ignore=['.gitfolder']):
    """
    Removes all files in the given 'path' parameter.

    Ignores the file with the name 'ignore'

    Parameters
    ----------
    path: str
        the path in which all files shall be removed
    ignore: list of strings
        the files that shall not be removed
    """
    ignore_paths = [os.path.join(path, f) for f in ignore]
    is_dir = os.path.isdir(path)
    is_file = os.path.isfile(path)

    # Start deleting only, if the given path exists at all
    if is_dir or is_file:
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            if f_path not in ignore_paths:
                if os.path.isdir(f_path):
                    shutil.rmtree(f_path)
                elif os.path.isfile(f_path):
                    os.remove(f_path)


def filter_dict(dictionary, keys):
    """
    Filters out all key-value pairs from a dictionary that are not in the 'keys' list.
    Parameters
    ----------
    dictionary: dict
        the dict that shall be filtered
    keys: list of strings
        the keys, that shall remain in the dict

    Returns
    -------
    dict
        the dictionary with only the keys provided in the 'keys' parameter
    """
    return {k: v for k, v in filter(lambda t: t[0] in keys, dictionary.items())}


def hash_dict(dictionary):
    """
    Creates a hash of a dict
    Parameters
    ----------
    dictionary: dict
        the dict for which a hash shall be created
    Returns
    -------
    str
        the hash  created from the parameter 'dictionary'
    """
    j_args = json.dumps(dictionary, ensure_ascii=True, sort_keys=True)
    sha1 = hashlib.sha1()
    sha1.update(j_args.encode(encoding='utf-8'))
    return sha1.hexdigest()
