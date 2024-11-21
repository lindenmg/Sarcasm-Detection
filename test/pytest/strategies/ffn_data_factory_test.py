from os import listdir
from pathlib import Path

from src.tools.config import Config
from src.tools.helpers import clean_folder
from src.strategies.ffn.ffn_data_factory import FFNDataFactory

args = {
    'cache_dir': Config.path.test_cache_folder,
    'train': False,
    'test': True,
    'sw_cut_filename': 'stop_words_cut_ultra.txt',
    'sw_full_filename': 'stop_words_full_ultra.txt',
    'raw_data': {
        'n_replies': 10,
    },
    'tokenization': {
        'spacy_model': 'en',
    },
    'filter': {
        'no_stop_words': True,
        'no_punctuation': True
    },
    'conversion': {
        'token_kind': 'lower_',
        'transform_specials': True
    },
    'vectorization': {
        'min_df': 1,
        'tfidf': True,
        'ngram_range': (2, 4),
        'analyzer': 'char',
        'max_features': 10
    }
}

pp_data_dir = Path(Config.path.test_cache_folder) / 'preprocessing' / 'pp_data'
pp_step_params_dir = Path(Config.path.test_cache_folder) / 'preprocessing' / 'pp_step_params'


class TestFFNDataFactory:

    @staticmethod
    def _clean():
        if pp_data_dir.is_dir():
            clean_folder(str(pp_data_dir))
        if pp_step_params_dir.is_dir():
            clean_folder(str(pp_step_params_dir))

    def test_get_data(self):
        def _filter_len(str_list, str_filter):
            return len(list(filter(lambda f: f.startswith(str_filter), str_list)))

        self._clean()
        session_tag = 'test_ffn_session'
        factory_params = {'session_tag': session_tag, 'pp_params': args,
                          'cache_dir': Config.path.test_cache_folder, 'test': False, 'train': True}
        fact = FFNDataFactory(**factory_params)
        result_dict = fact.get_data()
        train_dict = result_dict['train_data']
        test_dict = result_dict['test_data']
        assert all(map(lambda el: el is None, test_dict.values()))
        assert train_dict['posts'].shape == (
            int(args['raw_data']['n_replies'] / 2), args['vectorization']['max_features'])
        assert train_dict['replies'].shape == (
            int(args['raw_data']['n_replies']), args['vectorization']['max_features'])

        cache_files = listdir(str(pp_data_dir))
        assert _filter_len(cache_files, 'ffn_fact_vectorization') == 2
        assert _filter_len(cache_files, 'ffn_fact_vocab') == 1
        assert _filter_len(cache_files, 'ffn_fact_conversion') == 4
        assert _filter_len(cache_files, 'ffn_fact_tokenization') == 4

        params_files = listdir(str(pp_step_params_dir))
        assert _filter_len(params_files, 'ffn_fact_vectorization') == 2
        assert _filter_len(params_files, 'ffn_fact_vocab') == 1
        assert _filter_len(params_files, 'ffn_fact_conversion') == 4
        assert _filter_len(params_files, 'ffn_fact_tokenization') == 4

        # Now 'tfidf' is set to False. When get_data is executed,
        # the factory should create new cache files for fnn_fact_vectorization,
        # (but not for the other preprocessing steps)
        factory_params['pp_params']['vectorization']['tfidf'] = False
        fact_2 = FFNDataFactory(**factory_params)
        fact_2.get_data()
        cache_files = listdir(str(pp_data_dir))
        assert _filter_len(cache_files, 'ffn_fact_vectorization') == 4
        assert _filter_len(cache_files, 'ffn_fact_vocab') == 1
        assert _filter_len(params_files, 'ffn_fact_conversion') == 4
        assert _filter_len(params_files, 'ffn_fact_tokenization') == 4

        self._clean()
