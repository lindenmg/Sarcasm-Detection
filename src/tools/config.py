import json
import os
import warnings


class _SubConf:
    pass


class _Config(type):
    """
    Parent class for Config (See child at the bottom of the file)
    """

    _instances = {}

    def __init__(self, name, bases, attrs, path=None):
        super().__init__(name, bases, attrs)
        if path is None:
            path = os.path.realpath(os.path.join(__file__, '..', '..', '..', 'config.json'))
        self._instance = None
        self.debug_mode = True
        self.config_dict = {}
        self.load_json(path=path)

    def load_json(self, path):
        """
        For manually loading a config file, which must be in json format.
        Every key in the json is becoming a member of Config with the corresponding value.
        It raises a warning, if the json has duplicate keys.
        Parameters
        ----------
        path: str
            The path to the config json
        """
        self.__remove_config_members()
        with open(path, mode='r') as f:
            self.config_dict = json.load(f, object_pairs_hook=self.__dict_raise_on_duplicates)
        _Config.__add_config_members(self.config_dict, self)

    def data_path(self, file_name):
        """
        Convenience function for quickly getting the full path to a file in the data folder.
        Precondition is that 'data_folder' is defined in your config.json
        Parameters
        ----------
        file_name: str
            name of the file in the data folder, to which you want to have the path

        Returns
        -------
        str
            the path of the respective file

        """
        if file_name is None:
            return None
        else:
            return os.path.join(self.path.data_folder, file_name)

    def cache_path(self, file_name):
        """
        Convenience function for quickly getting the full path to a file in the cache folder.
        Precondition is that 'cache_folder' is defined in your config.json
        Parameters
        ----------
        file_name: str
            name of the file in the cache folder, to which you want to have the path
        Returns
        -------
        str
            the path of the respective file

        """
        if file_name is None:
            return None
        else:
            return os.path.join(self.path.cache_folder, file_name)

    def __remove_config_members(self):
        for k, _ in self.config_dict.items():
            if k in self.__dict__.keys():
                delattr(self, k)

    @staticmethod
    def __add_config_members(dictionary, obj):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = _Config.__add_config_members(v, _SubConf())
            setattr(obj, k, v)
        return obj

    @staticmethod
    def __dict_raise_on_duplicates(ordered_pairs):
        message = '\n\n\x1b[1;35m' + "There is a duplicate key {} " \
                                     "in the configuration file." + '\x1b[0m\n\n'
        d = {}
        for k, v in ordered_pairs:
            if k in d:
                print(message.format(k))
                warnings.warn(Warning(message))
            else:
                d[k] = v
        return d


class Config(metaclass=_Config):
    """
    Singleton class for configuration files. Access it
    A configuration file must be provided in the working director, which should be the project root.
    Examples for usage:
        from src.tools.config import Config

        # access directly by member
        Config.n_cpu

        # alternatively
        Config.config_dict['n_cpu']

        # use alternative config file
        Config.load_json(path='test/test_data/config_test.json'
        param_1 = Config.param_1

        # Get the path to a file in the data folder
        sw_full = Config.data_path("stop_words_full.txt")
    """
    pass
