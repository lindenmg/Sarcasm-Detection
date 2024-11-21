import os
from pathlib import Path

from src.tools.config import Config
from src.tools.helpers import load_from_disk
from src.tools.helpers import save_json_to_disk
from src.tools.helpers import save_to_disk
from src.training.abstract_cacher import AbstractCacher


class BaseCacher(AbstractCacher):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def eventually_load_cache(self, params, tag):
        """
        Convenience function for eventually loading cached data of
        a specific preprocessing step. The path to the cache file
        is conceived from the ``pp_step_params`` and the ``prefix``
        parameter, if it was previously created with the
        ``create_cache`` function.
        Parameters
        ----------
        params: dict
            dict all the preprocessing parameters, that where relevant
            at the corresponding preprocessing step.
        tag: str
            a tag that was appended to the name of the output file.
        Returns
        -------
        any format
            The cached data. If there is no cache, it returns ``None``

        """
        cache = None
        file_path = self._get_data_path(tag, params)
        if self.cache_dir is not None:
            if Path(file_path).is_file():
                if Config.debug_mode:
                    print('load cache: %s' % file_path)
                try:
                    cache = load_from_disk(file_path)
                except Exception as e:
                    print(e)
            else:
                if Config.debug_mode:
                    print('no cache for: %s' % file_path)
        return cache

    def create_cache(self, data, params, tag):
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
        params: dict
            the parameters that are relevant at the corresponding
            preprocessing step.
        tag: str
             a tag that is appended to the name of the output file.
            Suggestion: The name of the preprocessing step.

        Returns
        -------
        tuple: str, str
            First element of the tuple is the path to the cached data.
            Second element of the tuple is the path to the cached parameters (JSON)
        """
        if self.cache_dir is not None:
            data_path = self._get_data_path(tag, params)
            if Config.debug_mode:
                print('store cache: %s' % data_path)
            save_to_disk(data, data_path)

            params_path = self._get_params_path(tag, params)
            if Config.debug_mode:
                print('store cache params: %s' % params_path)
            save_json_to_disk(params, params_path, ensure_ascii=True, sort_keys=True)
            return data_path

    def _get_data_path(self, tag, params):
        hash_ = self.hash_dict(params)
        data_name = tag + '_' + hash_ + '.pp_cache'
        return os.path.join(self.cache_dir, 'hyperparam_searching', 'summaries', data_name)

    def _get_params_path(self, tag, params):
        hash_ = self.hash_dict(params)
        data_name = tag + '_' + hash_ + '.json'
        return os.path.join(self.cache_dir, 'hyperparam_searching', 'params', data_name)
