"""
• It should be applied with search_executor
"""

import json
import os
from copy import deepcopy

from src.hyperparameters.abstract_param_iterator import AbstractParameterIterator
from src.hyperparameters.param_operation import traverse_add, add_tuple_key_value, add_single_key_value, traverse_get, \
    sort_dict


class GridParameterIterator(AbstractParameterIterator):
    """
    A class that can be used for generating hyperparameters for
    `src.training.learning_session.LearningSession`.
    Example
    """

    def __init__(self, default_dict, variation_list, prefix='',
                 enumerate_digits=3, keys=None, filename_path=None, iterations=None):
        """
        default dict: dict
            dict with all default values
        variation_list: list of dicts
            list with all values for which variations shall be generated.
            The values from which variations shall be in a list and the
            corresponding key must be equal to one key in the default_dict
        enumerate_digits: int
            The number of digits for the enumeration of the resulting
            hyperparameter  sets.
        keys: list of list of str
            The keys, for which the values shall be added to the generated
            filenames. Pass `None` to add the values of all keys, that are
            specified in the init parameter `variation_list`
        filename_path: (list of strings, optional)
            the keys, where the filename shall be stored in the resulting json.
            None, if the filename shall not be stored in the resulting json
        """
        super().__init__(default_dict, variation_list, iterations)
        self.hyperparameters_dict = dict()
        self._filename_path = filename_path
        self._prefix = prefix
        self._enumerate_digits = enumerate_digits
        self._keys = keys
        self._generate_filenames(
            self._generate_params_list())
        self.hyperparameters_list = sorted(
            list(self.get_hyperparameters().values()),
            key=lambda d: json.dumps(d, sort_keys=True))
        if iterations is None:
            self._iterations = len(self.hyperparameters_dict)

    def update(self, loss):
        pass

    def get_next_hyperparameter_dict(self, i):
        return self.hyperparameters_list[i]

    def __next__(self):
        if self._i < self._iterations:
            params = self.hyperparameters_list[self._i]
            self._i += 1
            return params
        else:
            raise StopIteration()

    def get_filenames(self):
        """
        Creates the filename of every hyperparameter set
        Returns
        -------
        list of str
            The filename of every hyperparameter set.
        """
        return [k + '.json' for k in self.hyperparameters_dict.keys()]

    def get_hyperparameters(self):
        """
        The generated hyperparameters
        Returns
        -------
        dict
            dict with the hyperparameters
        """
        return self.hyperparameters_dict

    def dump_hyperparameters(self, directory_path):
        """
        Creates a json for every hyperparameter set in `directory_path`
        Parameters
        ----------
        directory_path
            The path where the hyperparameters shall be created.
        """
        for k, v in self.hyperparameters_dict.items():
            path = os.path.join(directory_path, k + '.json')
            os.makedirs(directory_path, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(v, f, sort_keys=True)

    def _generate_params_list(self):
        """
        Generates the hyperparameter variations
        Returns
        -------
        list of dict
            All hyperparameter variations, generated from init parameters
            `default_dict` and `variation_dict`.
        """
        parameter_list = [deepcopy(self._default_dict)]
        for var in self._variation_list:
            var = deepcopy(var)
            if isinstance(var['val'], tuple):
                parameter_list = add_tuple_key_value(var, parameter_list)
            else:
                parameter_list = add_single_key_value(var, parameter_list)
        parameter_list = [sort_dict(d) for d in parameter_list]
        return parameter_list

    def _generate_filenames(self, params_list):
        d = dict()

        def _enumeration(f):
            if f not in d.keys():
                d[f] = -1
            d[f] = d[f] + 1
            filename = ('%.' + str(self._enumerate_digits) + 'd') % d[f]
            if f != '':
                filename = '_'.join([f, filename])
            return filename

        for params in params_list:
            values = [self._prefix]
            if self._keys is not None:
                for key in self._keys:
                    values.append(str(traverse_get(params, key)))
            fname = '_'.join(values)
            fname = _enumeration(fname)
            fname = fname.replace(' ', '')
            if self._filename_path is not None:
                traverse_add(params, self._filename_path, fname)
            self.hyperparameters_dict[fname] = params


"""
the following dict can be used as default parameters
for GridParameterIterator
"""

default_ffn_params = {
    "learning_session": {
        "session_tag": "ffn_test_00",
        "epochs": 20,
        "mode": "train",
        "save_interval": 5,
        "cnn_mode": False,
        "cache_prefix": "fnn_fact",
        "seed": 1337
    },
    "metrics": {
        "output_activation": "log_softmax"
    },
    "model": {
        "module": "src.models.ffn_model",
        "class_name": "FFN00",
        "args": {
            "in_post": 10,
            "in_reply": 10,
            "input_post": 3,
            "input_reply": 5,
            "post_dropout": 0.5,
            "reply_dropout": 0.5,
            "hl1_dropout": 0.5,
            "hidden_layer_1": 3,
            "output_size": 2
        }
    },
    "criterion": {
        "module": "torch.nn",
        "class_name": "NLLLoss",
        "args": {
            "size_average": True
        }
    },
    "optimizer": {
        "module": "torch.optim",
        "class_name": "ASGD",
        "args": {
            "lr": 0.005
        }
    },
    "lr_scheduler": {
        "module": "torch.optim.lr_scheduler",
        "class_name": "ReduceLROnPlateau",
        "args": {
            "mode": "min",
            "factor": 0.7,
            "patience": 2,
            "threshold_mode": "abs",
            "threshold": 0.002
        }
    },
    "cv_iterator_factory": {
        "module": "src.training.base_cv_it_factory",
        "class_name": "BaseCvIteratorFactory",
        "args": {
            "n_splits": 1,
            "split_ratio": 0.9,
            "batch_size": 50,
            "shuffle": False,
            "num_workers": 4,
            "dataset_module": "src.strategies.ffn.ffn_dataset",
            "dataset_classname": "FFNDataset"
        }
    },
    "data_factory": {
        "module": "src.strategies.ffn.ffn_data_factory",
        "class_name": "FFNDataFactory",
        "args": {
            "pp_params": {
                "sw_cut_filename": "stop_words_cut_ultra.txt",
                "sw_full_filename": "stop_words_full_ultra.txt",
                "raw_data": {
                    "n_replies": 10
                },
                "tokenization": {
                    "spacy_model": "en_core_web_md"
                },
                "filter": {
                    "no_stop_words": True,
                    "no_punctuation": False
                },
                "conversion": {
                    "token_kind": "lower_",
                    "transform_specials": True
                },
                "vectorization": {
                    "tfidf": False,
                    "min_df": 1,
                    "ngram_range": [
                        1,
                        2
                    ],
                    "analyzer": "word",
                    "max_features": 10
                }
            }
        }
    }
}
