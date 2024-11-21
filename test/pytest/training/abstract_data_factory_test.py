import json
import os

from src.tools.config import Config
from src.tools.helpers import load_from_disk, clean_folder
from src.training.abstract_data_factory import AbstractDataFactory


class Mock(AbstractDataFactory):
    def get_data(self):
        return None


pp_params = {
    'pp_step_1': {
        'spacy_model': 'en_core_web_md',
    },
    'pp_step_2': {
        'first_param_for_pp_step': 'stop_words_txt_ultra.txt',
        'second_param_for_pp_step': 'stop_words_full_ultra.txt',
    },
    'pp_step_3': {
        'token_kind': None,
        'transform_specials': True
    }
}

list_str = [
    'this is some test preprocessing',
    'for testing'
]


class TestAbstractDataFactory:
    @staticmethod
    def _clean():
        base_folder = os.path.join(Config.path.test_cache_folder, 'preprocessing')
        data_folder = os.path.join(base_folder, 'pp_data')
        params_folder = os.path.join(base_folder, 'pp_step_params')
        mapping_folder = os.path.join(base_folder, 'pp_step_mapping')
        ignore = ['.gitfolder']
        clean_folder(data_folder, ignore)
        clean_folder(params_folder, ignore)
        clean_folder(mapping_folder, ignore)

    def test_create_cache(self):
        self._clean()
        mock = Mock(pp_params, Config.path.test_cache_folder, session_tag='test_session_00')
        datapath, jsonpath = mock.create_cache(list_str, pp_params, 'test_step')
        data_aux = load_from_disk(datapath)
        assert data_aux == list_str
        with open(jsonpath) as f:
            args_aux = json.load(f)
        assert args_aux == pp_params
        self._clean()

    def test_eventually_load_cache(self):
        self._clean()
        mock = Mock()
        mock.cache_dir = Config.path.test_cache_folder
        _, _ = AbstractDataFactory.create_cache(mock, list_str, pp_params, 'test_cache_')
        data_aux = AbstractDataFactory.eventually_load_cache(mock, pp_params, 'test_cache_')
        assert list_str == data_aux
        self._clean()
