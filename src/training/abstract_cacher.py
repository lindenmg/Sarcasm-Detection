import hashlib
import json
from abc import ABC
from abc import abstractmethod


class AbstractCacher(ABC):

    @abstractmethod
    def eventually_load_cache(self, params, tag):
        pass

    @abstractmethod
    def create_cache(self, data, params, tag):
        pass

    @staticmethod
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
