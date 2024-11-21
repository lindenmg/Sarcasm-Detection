"""
This script contains operations, that are used by various modules in the
hyperparameter package. It contains operations, with which parameter
can be converted from a dictionary representation (e.g. for training.learning_session)
to a vector representation (e.g. necessary in bayesian optimization).
It also contains functions, that are used for adding values to normal
and nested dictionaries.
"""

import collections
import warnings
from copy import deepcopy

import numpy as np

key_separator = '_keysep_'
path_separator = '_pathsep_'
val_type_separator = '_valtypesep_'
val_type_quant_s = 'quantitative_singleton'
val_type_quant_g = 'quantitative_group'
val_type_quali_s = 'qualitative_singleton'
val_type_quali_g = 'qualitative_group'


def move_col_in_df(df, col_name, new_idx):
    """
    moves the column (col_name) to the index (new_idx)
    in df

    Parameters
    ----------
    df: DataFrame
    col_name: str
    new_idx: int
    Returns
    -------
    DataFrame
        the df with new col sequence
    """
    cols = list(df)
    cols.insert(new_idx, cols.pop(cols.index(col_name)))
    df = df.ix[:, cols]
    return df


def flatten_complete(list_of_lists):
    """
    Recursively flattens a list with nested lists to
    a single hierarchy

    Parameters
    ----------
    list_of_lists : list of list
    Returns
    -------
    list
        contains the Variables that have been in the nested inner lists
    """
    res = []
    for el in list_of_lists:
        if isinstance(el, list) or isinstance(el, tuple) or isinstance(el, set):
            res += flatten_complete(el)
        else:
            res += [el]
    return res


def flatten_dict(dictionary, keys):
    """
    Creates a new dict with all the keys from the `keys` parameter in the first hierarchy.
    The original dictionary will not be modified, a new generated dict is returned instead.

    Parameters
    ----------
    dictionary: dict
        The dict which shall be flattened
    keys: list of tuples
        First element of every tuple: the key for the value, which shall be filtered
        Second element of every tuple: the key which shall be used instead

    Returns
    -------
    dict
        the flattened dict with the keys, specified in the keys' argument.

    """

    def _recurse(subdict, keys, acc):
        k_match = [k for k in keys if k[0] in subdict.keys()]
        k_rest = [k for k in keys if k not in k_match]
        for k in k_match:
            acc[k[1]] = subdict[k[0]]
        for k in [k for k in subdict.keys() if k not in k_match]:
            if isinstance(subdict[k], dict):
                _recurse(subdict[k], k_rest, acc)

    acc = dict()
    _recurse(deepcopy(dictionary), deepcopy(keys), acc)
    return acc


def flatten_dict_auto(dictionary, sep='_'):
    """
    Works like flatten_dict, but instead of specifying the keynames
    manually, the user just passes in the dict. The resulting keys
    are the path in the tree.
    The original dictionary will not be modified, a new generated dict is returned instead.
    Parameters
    ----------
    dictionary
        the dict, which shall be flattened
    sep
        the separator which is used in the resulting keys between each node
    Returns
    -------
    dict
        the flattened dict with the path to each value as key.
    """

    def _recurse(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(_recurse(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return _recurse(dictionary, '')


def params_vector_to_dict(defaults_dict, variation_list, params_vector):
    """
    Converts a parameter vector (e.g. created by params_dict_to_vector) into
    a dictionary representation, which can be used for a training.learning_session
    object. Therefore, the function uses a dictionary of default values (defaults_dict)
    and a list of variation rules (variation_list), that are normally used in:
    • hyperparameters.bayesian_search.bayesian_param_iterator
    • hyperparameters.random_search.random_parameter_iterator
    • hyperparameters.grid_search.grid_param_iterator

    The function ensures the following things:
    • single (qualitative) values, that are represented with an index (in the vector), are
    translated to their actual value
    • continuous (quantitative) single values, that are represented by a float are
    added to the dict without any manipulation
    • discrete single values, that have to be passed to a training.learning_session,
    are rounded and converted to an int
    • quantitative values that are optimized in a group (represented with tuples in the variation list),
    that were aggregated to a single value, are reproduced proportionally to the value in the
    parameter vector. Discrete values (int) are rounded, while continuous values (floats)
    are written into the resulting parameter dict as they are.
    • qualitative values that are optimized in a group (represented with tuples in the variation list),
    that were aggregated to a single value, are reproduced by using the index from the parameter vector.

    Parameters
    ----------
    defaults_dict: dict
        dictionary which represents the default structure of the
        hyperparameter setting.
    variation_list: list of dicts
        list with all values for which variations shall be generated in
        hyperparameter search. The values from which variations shall be
        in a list and the corresponding key must be equal to one key
        in the default_dict.
    params_vector: list
        a vector of values, from which a parameter dict shall be created.
    Returns
    -------
    dict
        dict which can be used for
        src.training.learning_session.

    """
    params_dict = deepcopy(defaults_dict)
    var_list_sorted = sort_variation_list(variation_list)

    def __deserialize_keys(k, add_val_type=False):
        s = k.split(val_type_separator)
        val_type = s[-1]
        paths = s[:-1][0].split(path_separator)
        key_paths = [p.split(key_separator) for p in paths]
        if add_val_type:
            return key_paths, val_type
        else:
            return key_paths

    def __add_qualitative_value(vec_val, var_dict, param_dict):
        """Looks up the value for the index in the variation list"""
        traverse_add(param_dict, var_dict['keys'], var_dict['val'][int(vec_val)])

    def __add_quantitative_value(vec_val, var_dict, param_dict):
        if 'quantity_type' in var_dict.keys() and var_dict['quantity_type'] == 'float':
            v = vec_val
        else:  # var_dict['quantity_type'] == 'int'
            v = int(np.round(vec_val))
        traverse_add(param_dict, var_dict['keys'], v)

    def __add_qualitative_group(vec_val, var_dict, param_dict):
        """Looks up every value for the index in the variation list by every path"""
        for k, v in zip(var_dict['keys'], var_dict['val']):
            traverse_add(param_dict, k, v[int(vec_val)])

    def __add_quantitative_group(vec_val, var_dict, param_dict):
        min_max = list(zip(var_dict['min'], var_dict['max']))
        percent = (vec_val - min_max[0][0]) / (min_max[0][1] - min_max[0][0])
        values = [percent * max - percent * min + min for min, max in min_max]
        if 'quantity_type' in var_dict:
            if len(var_dict['quantity_type']) != len(var_dict['keys']):
                raise IndexError('The number of elements in "quantity_type" must be equal to ' +
                                 'the number of all other tuples in a variation list')
            # now the values are rounded
            values = [int(np.round(v)) if qt == 'int' else v for qt, v in zip(var_dict['quantity_type'], values)]
        for k, v in zip(var_dict['keys'], values):
            traverse_add(param_dict, k, v)

    aggregated_variation_list = __aggregate_variation_list(var_list_sorted, defaults_dict)
    keys_serialized = [t[0] for t in aggregated_variation_list]
    key_paths = [__deserialize_keys(k, add_val_type=True) for k in keys_serialized]
    for i, k_path in enumerate(key_paths):
        val_type = k_path[-1]
        vec_val = params_vector[i]
        var_dict = var_list_sorted[i]
        if val_type == val_type_quali_s:
            __add_qualitative_value(vec_val, var_dict, params_dict)
        elif val_type == val_type_quant_s:
            __add_quantitative_value(vec_val, var_dict, params_dict)
        elif val_type == val_type_quali_g:
            __add_qualitative_group(vec_val, var_dict, params_dict)
        else:  # val_type == val_type_quant_g
            __add_quantitative_group(vec_val, var_dict, params_dict)
    return params_dict


def params_dict_to_vector(variation_list, params_dict):
    """
    Converts a parameter dict for a learning session into a vector. This
    is done in such a way, that it can be used for bayesian optimization.
    Key aspects:
    • quantitative singleton parameters values are adopted as they are
    • qualitative or nominal singleton parameters are converted into the index.
    • quantitative group parameters are represented by only the first value. This is crucial,
    as the proportionality between the values must be preserved after the optimization.
    • qualitative or nominal group parameters are represented by the index. This is crucial as the index
    must be the same after optimization.
    • Default values (that are not represented in the variation_list) are
    omitted, as they shall not be optimized.
    • The resulting values are sorted alphabetically by their corresponding keys

    Parameters
    ----------
    variation_list: list of dicts
        list with all values for which variations shall be generated.
        The values from which variations shall be in a list and the
        corresponding key must be equal to one key in the default_dict.
    params_dict: dict
        the format of the dict is according to the format for
        src.training.LearningSession.

    Returns
    -------
    np.array
        A vector which represents the values from the params_dict.
        See the function description.
    """
    aggregated_variation_list = __aggregate_variation_list(variation_list, params_dict)
    return [t[1] for t in aggregated_variation_list]


def sort_variation_list(variation_list):
    """
    Sorts a list with one of the following formats by 'keys':
    • [{'keys': list of strings, any other key-value paris...}]
    • [{'keys': tuple of lists of strings, any other key-value paris...}]

    Variation lists are used for defining rules, with which hyperparameters
    are varied in a hyperparameter optimization. Such rules may be min/max values
    of a hyperparameter, or nominal values of a hyperparameter.
    Parameters
    ----------
    variation_list: list of dicts
        list with all values for which variations shall be generated in
        hyperparameter search. The values from which variations shall be
        in a list and the corresponding key must be equal to one key
        in the default_dict.

    Returns
    -------
    list of dicts
        the sorted variation list.
    """

    def _keyf(d):
        k = d['keys']
        if isinstance(k, tuple):
            return path_separator.join([key_separator.join(p) for p in k])
        else:
            return key_separator.join(k)

    return sorted(variation_list, key=_keyf)


def __aggregate_variation_list(variation_list, params_dict):
    def __get_singleton_params(singleton_keys, singleton_params):
        # combine a singleton params
        acc = []
        for key in singleton_keys:
            for d in v_aux:
                if key in d['keys'] and 'max' in d.keys():
                    key_serialization = key + val_type_separator + val_type_quant_s
                    # val, min and max must be proportional. So we can use the percentage
                    actual_value = [p[1] for p in singleton_params if key in p][0]
                    acc.append((key_serialization, actual_value))
                elif key in d['keys'] and 'val' in d.keys():
                    key_serialization = key + val_type_separator + val_type_quali_s
                    actual_value = [p[1] for p in singleton_params if key in p][0]
                    try:
                        idx = [i for i, v in enumerate(d['val']) if actual_value == v][0]
                    except IndexError:
                        idx = 0
                        warnings.warn('Value ' + str(actual_value) + ' for the key ' + str(key) +
                                      ' is not a valid variation. ' + ' valid variations are ' + str(d['val']) +
                                      '. Will take 0 for indexing.')
                    acc.append((key_serialization, idx))
        return acc

    def __get_group_params(group_keys, group_params):
        # combine quantitative group params
        acc = []
        for keys in group_keys:
            for d in v_aux:
                if keys[0] in d['keys'] and 'max' in d.keys():
                    key_serialization = path_separator.join(keys) + val_type_separator + val_type_quant_g
                    # All values in the group are proportional. So we can take the first
                    actual_value = [p[1] for p in group_params if keys[0] in p][0]
                    acc.append((key_serialization, actual_value))
                elif keys[0] in d['keys'] and 'val' in d.keys():
                    key_serialization = path_separator.join(keys) + val_type_separator + val_type_quali_g
                    actual_value = [p[1] for p in group_params if keys[0] in p][0]
                    val = d['val'][0]
                    try:
                        idx = [i for i, v in enumerate(val) if v == actual_value][0]
                    except IndexError:
                        idx = 0
                        warnings.warn('Value ' + str(actual_value) + ' for the key ' + str(keys) +
                                      ' is not a valid variation. ' + ' valid variations are ' + str(d['val']) +
                                      '. Will take 0 for indexing.')
                    acc.append((key_serialization, idx))
        return acc

    v_aux = deepcopy(variation_list)
    for d in v_aux:
        if isinstance(d['keys'], tuple):
            d['keys'] = [key_separator.join(path) if isinstance(path, list) else path for path in d['keys']]
        else:
            d['keys'] = [key_separator.join(d['keys'])]
    flat_params = sorted([(k, v) for k, v in flatten_dict_auto(params_dict, sep=key_separator).items()])
    v_list_keys = [d['keys'] for d in v_aux]
    group_keys = [g for g in v_list_keys if len(g) > 1]
    singleton_params = [g for g in flat_params if g[0] not in flatten_complete(group_keys)]

    singleton_keys = [t[0] for t in singleton_params]
    group_params = [p for p in flat_params if p[0] not in singleton_keys]

    aux = sorted(__get_singleton_params(singleton_keys, singleton_params) +
                 __get_group_params(group_keys, group_params))
    return aux


def traverse_get(dictionary, keyslist):
    """
    Get a value in a dict with nested dict
    Parameters
    ----------
    dictionary: dict
        a dict with nested dicts
    keyslist: list of str
        the keys that represent the path to the value.
        The first key is in the first hierarchy,
        the second in the second hierarchy, and so on
    Returns
    -------
    Any value
    """

    def _recurse(subdict, klist):
        k = klist.pop(0)
        if len(klist) <= 0:
            return subdict.get(k)
        elif isinstance(subdict.get(k), dict):
            return _recurse(subdict.get(k), klist)
        else:
            return None

    return _recurse(dictionary, deepcopy(keyslist))


def filter_by_keys_lists(dictionary, keyslists):
    """
    Filters out all key-value pairs in a nested dictionary,
    where the corresponding key is not referenced in keyslists
    Parameters
    ----------
    dictionary
        A nested dictionary that shall be filtered
    keyslists
        list of lists of strings
        every sublist in a list of strings, which references
        a value in a nested dict
    Returns
    -------
    dict
        the filtered dict
    """
    res_dict = dict()
    for keyslist in keyslists:
        val = traverse_get(dictionary, keyslist)
        traverse_add(res_dict, keyslist, val, parents=True)
    return res_dict


def traverse_add(dictionary, keyslist, val, parents=False):
    """
    Adds a value to a keys-path in a nested dict.
    >>> Example
    >>> d = {
    >>>     'nested': {
    >>>         'val':0
    >>>     }
    >>> }
    >>> traverse_add(d, ['nested', 'val'], 23)
    >>> print(d['nested']['val'])
    >>> Out[0] 23
    Parameters
    ----------
    dictionary: dict
        the dict to which a value shall be added
    keyslist: list
        list of keys representing the path to the value
    val: whatever
        the value, that shall be added
    parents: bool
        make parents if path does not exist
    """

    def _recurse(subdict, klist, val):
        k = klist.pop(0)
        if len(klist) <= 0:
            subdict[k] = val
        elif isinstance(subdict.get(k), dict):
            _recurse(subdict.get(k), klist, val)
        elif parents:
            subdict[k] = dict()
            _recurse(subdict[k], klist, val)
        else:
            raise KeyError('the path does not exist: ' + str(keyslist))

    return _recurse(dictionary, deepcopy(keyslist), val)


def add_single_key_value(variation, params_list):
    v = variation['val'].pop(0)
    k = variation['keys']
    for par in params_list:
        traverse_add(par, k, v)
    p_list = params_list
    for val in variation['val']:
        for params in p_list:
            p = deepcopy(params)
            traverse_add(p, variation['keys'], val)
            params_list = params_list + [p]
    return params_list


def add_tuple_key_value(variation, params_list):
    for k, v in zip(variation['keys'], variation['val']):
        val = v.pop(0)
        for par in params_list:
            traverse_add(par, k, val)
    p_list = params_list
    for vals in list(zip(*variation['val'])):
        for params in p_list:
            p = deepcopy(params)
            for k, v in zip(variation['keys'], vals):
                traverse_add(p, k, v)
            params_list = params_list + [p]
    return params_list


def sort_dict(dictionary):
    result = {}
    for k, v in sorted(dictionary.items()):
        if isinstance(v, dict):
            result[k] = sort_dict(v)
        else:
            result[k] = v
    return result


def same_dict_format(dict1, dict2):
    def _recurse(d1, d2):
        for k, v in d1.items():
            if k not in d2.keys():
                return False
            if isinstance(v, dict):
                if not _recurse(v, d2[k]):
                    return False
        return True

    if _recurse(dict1, dict2):
        return _recurse(dict2, dict1)
    else:
        return False


def assert_param_dicts(param_dicts, variation_list):
    def _test_quantitative_val(res, var):

        if 'quantity_type' in var.keys():
            if var['quantity_type'] == 'int':
                assert isinstance(res, int)
            else:
                assert isinstance(res, float)

        assert res <= var['max']
        assert res >= var['min']

    for param in param_dicts:
        for variation in variation_list:
            if isinstance(variation['keys'], tuple):

                # case qualitative group
                if 'val' in variation.keys():
                    for t in zip(variation['keys'], variation['val']):
                        assert traverse_get(param, t[0]) in t[1]
                # case quantitative group
                else:
                    for i in range(len(variation['keys'])):
                        _test_quantitative_val(
                            traverse_get(param, variation['keys'][i]),
                            {k: v[i] for (k, v) in variation.items()})
            else:

                # case: qualitative single
                if 'val' in variation.keys():
                    res_val = traverse_get(param, variation['keys'])
                    assert res_val in variation['val']

                # case: quantitative single
                else:
                    _test_quantitative_val(
                        traverse_get(param, variation['keys']),
                        variation)
