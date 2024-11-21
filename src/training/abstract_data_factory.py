import os
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from pathlib import Path

from src.tools.helpers import load_from_disk, save_to_disk, save_json_to_disk, hash_dict


class AbstractDataFactory(ABC):
    """
    A subclass of the AbstractDataFactory shall provide the preprocessed
    data with the user-defined ``get_data`` method. This way, the preprocessing
    can be automated by ``LearningSession``. It is the responsibility of the user
    to implement 'get_data' in such a way, that the preprocessing can be
    executed efficiently. Therefore it is suggested to implement efficient caching
    for every preprocessing step (the abstract class provides some convenience
    functions for this)

    Subclass Requirements
    ---------------------
    The output ``get_data`` shall fulfill the following requirements:
    * the first dict contains the data for training
    * the second dict contains the data for testing
    * the keys of both dicts are equal to the init-parameters
      of the targeted Dataset

    Example
    -------
    >>> from strategies.dummy.dummy_ffn import DummyDataset
    >>> from strategies.dummy.dummy_data_factory import DummyDataFactory
    >>>
    >>> # The parameters with which the data is generated
    >>> pp_params = {
    >>>     'seed': 44,
    >>>     'train_examples': 10,
    >>>     'test_examples': 2,
    >>>     'dim': 10}
    >>>
    >>> # the user defined subclass of AbstractDataFactory
    >>> data_factory = DummyDataFactory(pp_params=pp_params, train=True)
    >>>
    >>> # the user defined 'get_data' method
    >>> result_dict = data_factory.get_data()
    >>> train_dict = result_dict['train_data']
    >>> test_dict = result_dict['test_data']
    >>>
    >>> # now the dicts can be used for the targeted dataset
    >>> train_dataset = DummyDataset(**train_dict)
    >>> test_dataset  = DummyDataset(**test_dict)

    For more examples, check out:
    * Easy example: ``src.strategies.dummy.dummy_data_factory`` 
    * More advanced example: ``src.strategies.tfidf.tfidf_data_factory``
    * The test: ``test.pytest.training.abstract_data_factory_test``
    * The test: ``test.pytest.strategies.tfidf_data_factory_test``

    *caching should be done using a subclass of training.abstract_cacher
    """

    def __init__(self, pp_params={}, cache_dir=None, train=False, test=False
                 , session_tag='sarcasm_detection', cache_prefix='ffn_fact'):
        self.pp_params = pp_params
        self.cache_dir = cache_dir
        self.session_tag = session_tag
        self.train = train
        self.test = test
        self.cache_prefix = cache_prefix

    @abstractmethod
    def get_data(self):
        """
        Must be implemented (in a subclass) by the user. It is supposed to
        execute the preprocessing, based on the preprocessing parameters provided
        in 'self.pp_params'.

        Returns
        -------
        dict
            The dict shall fulfill the following requirements:
            * Key 'test_data' which returns dict with data for testing
            * Key 'train_data' which returns dict with data for training
            * The keys of both dicts are equal to the init-parameters
              of the targeted Dataset
            * Optional key 'word_vectors' which returns torch.FloatTensor
              for pretrained embedding layer
            * Optional key 'reply_lengths' which returns torch.LongTensor according to
              helpers.create_length_tensor for the replies of the training set.
              This is for the initialization of a data_science.samplers.BucketRandomSampler
            * Optional key 'embedding_size' which returns the (int) amount of word-vectors
              in the value of the key 'word_vectors'
            * Further optional keys possible...
        """
        raise NotImplementedError

    #                 _     _
    #   ___ __ _  ___| |__ (_)_ __   __ _
    #  / __/ _` |/ __| '_ \| | '_ \ / _` |
    # | (_| (_| | (__| | | | | | | | (_| |
    #  \___\__,_|\___|_| |_|_|_| |_|\__, |
    #                               |___/

    # The following functions can be used from a subclass of AbstractDataFactory
    # to implement efficient logging

    def eventually_load_cache(self, pp_step_params, step_name='pp_step'):
        """
        Convenience function for eventually loading cached data of
        a specific preprocessing step. The path to the cache file
        is conceived from the ``pp_step_params`` and the ``prefix``
        parameter, if it was previously created with the
        ``create_cache`` function.
        Parameters
        ----------
        pp_step_params: dict
            dict all the preprocessing parameters, that where relevant
            at the corresponding preprocessing step.
        step_name: str
            a tag that was appended to the name of the output file.
        Returns
        -------
        any format
            The cached data. If there is no cache, it returns ``None``

        """
        cache = None
        file_path = self._get_data_path(step_name, pp_step_params)
        if self.cache_dir is not None:
            if Path(file_path).is_file():
                print('load cache: %s' % file_path)
                try:
                    cache = load_from_disk(file_path)
                except Exception as e:
                    print(e)
            else:
                print('no cache for: %s' % file_path)
        return cache

    def create_cache(self, data, pp_step_params, step_name='pp_step'):
        """
        Caches the provided data in the path ``self.cache_dir`` (init-parameter).
        Adds a hash of ``pp_step_params`` to the filename.
        Stores the parameters from ``pp_step_params`` as JSON with the respective
        hash (so the parameters of the cache can be looked up manually).
        Stores a mapping from the cached data to the session_tag.
        Parameters
        ----------
        data: any kind
            the data, that shall be cached
        pp_step_params: dict
            the parameters that are relevant at the corresponding
            preprocessing step.
        step_name: str
             a tag that is appended to the name of the output file.
            Suggestion: The name of the preprocessing step.

        Returns
        -------
        tuple: str, str
            First element of the tuple is the path to the cached data.
            Second element of the tuple is the path to the cached parameters (JSON)
        """
        if self.cache_dir is not None:
            data_path = self._get_data_path(step_name, pp_step_params)
            print('store cache: %s' % data_path)
            save_to_disk(data, data_path)
            params_path = self._get_params_path(step_name, pp_step_params)
            print('store params: %s' % params_path)
            save_json_to_disk(pp_step_params, params_path, ensure_ascii=True, sort_keys=True)
            mapping_path = self._get_params_mapping_path(step_name, pp_step_params)
            try:
                os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
                with open(mapping_path, 'a+') as file:
                    file.write(self.session_tag + '\n')
            except Exception as e:
                print(e)
            print('store session_tag: %s' + mapping_path)
            return data_path, params_path

    def create_data(self, step_name, pp_step_params, __create_data, cache=True):
        data = self.eventually_load_cache(pp_step_params, step_name)
        if data is None:
            data = __create_data()
            if cache:
                self.create_cache(data, pp_step_params, step_name)
        return data

    def create_step_name(self, step, tag=None):
        if tag is None:
            return self.cache_prefix + '_' + step
        else:
            return self.cache_prefix + '_' + step + '_' + tag

    def combine_params(self, keys):
        """
        Parameters
        ----------
        keys: list of str
            keys of self.pp_params
        Returns
        -------
        dict
            Converts dicts behind the ´´keys´´ to single dict
        """
        params = deepcopy(self.pp_params)

        def rec(acc, el):
            if acc is None:
                return params[el]
            else:
                acc.update(params[el])
                return acc

        return reduce(rec, keys, None)

    def _get_params_mapping_path(self, step_name, pp_step_params):
        hash_ = hash_dict(pp_step_params)
        data_name = step_name + '_' + hash_ + '.pp_step_mapping'
        return os.path.join(self.cache_dir, 'preprocessing', 'pp_step_mapping', data_name)

    def _get_data_path(self, step_name, pp_step_params):
        hash_ = hash_dict(pp_step_params)
        data_name = step_name + '_' + hash_ + '.pp_cache'
        return os.path.join(self.cache_dir, 'preprocessing', 'pp_data', data_name)

    def _get_params_path(self, step_name, pp_step_params):
        hash_ = hash_dict(pp_step_params)
        data_name = step_name + '_' + hash_ + '.json'
        return os.path.join(self.cache_dir, 'preprocessing', 'pp_step_params', data_name)
