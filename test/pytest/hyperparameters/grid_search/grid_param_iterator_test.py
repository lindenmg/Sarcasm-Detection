import json
from os import rmdir, mkdir, listdir
from os.path import join

from src.tools.config import Config
from src.tools.helpers import clean_folder
from src.hyperparameters.grid_search.grid_param_iterator import GridParameterIterator

default_dict = {
    'param0': 0,
    'param1': {
        'param10': 0,
        'param11': {
            'param110': 0
        },
        'param12': 0
    }
}

variation_list = [{
    'keys': ['param0'],
    'val': ['aloha', 'servus']
}, {
    'keys': (['param1', 'param10'], ['param1', 'param12']),
    'val': ([1, 2, 3], ['one', 'two', 'three'])
}, {
    'keys': ['param1', 'param11', 'param110'],
    'val': ['WAT', 'DAT']
}
]

hyperparam_path = join(Config.path.test_data_folder, 'tmp_hyperparams')


class TestGridParameterIterator:

    def test_get_params_list(self):
        generator = GridParameterIterator(default_dict, variation_list)
        params = generator.get_hyperparameters()

        # ensure that the correct number of variations is created
        assert len(params) == 12 and len(params) == generator._iterations

        # ensure that the values for the keys [param1],[param10] and [param1][param12]
        # are correctly combined in every element of params
        assert isinstance(params, dict)
        assert all([(p['param1']['param10'] == 1 and p['param1']['param12'] == 'one') or
                    (p['param1']['param10'] == 2 and p['param1']['param12'] == 'two') or
                    (p['param1']['param10'] == 3 and p['param1']['param12'] == 'three')
                    for _, p in params.items()])

    def test_next_iteration(self):
        grid_iterator = GridParameterIterator(default_dict, variation_list)
        parameters = [p for p in grid_iterator]
        assert len(parameters) == 12
        assert all([(p['param1']['param10'] == 1 and p['param1']['param12'] == 'one') or
                    (p['param1']['param10'] == 2 and p['param1']['param12'] == 'two') or
                    (p['param1']['param10'] == 3 and p['param1']['param12'] == 'three')
                    for p in parameters])

    def test_get_filenames(self):
        prefix = 'test'
        generator = GridParameterIterator(default_dict, variation_list, prefix=prefix,
                                          enumerate_digits=3,
                                          keys=[['param0'], ['param1', 'param11', 'param110']])

        filenames = generator.get_filenames()

        assert all([f.startswith(prefix) for f in filenames])
        assert len([_ for _ in filenames if _.endswith('.json') > 0]) == 12
        assert len([_ for _ in filenames if _.startswith('test_aloha')]) == 6
        assert len([_ for _ in filenames if _.startswith('test_servus')]) == 6
        assert len([_ for _ in filenames if _.find('WAT') > 0]) == 6
        assert len([_ for _ in filenames if _.find('DAT') > 0]) == 6
        assert len([_ for _ in filenames if _.find('000') > 0]) == 4
        assert len([_ for _ in filenames if _.find('001') > 0]) == 4
        assert len([_ for _ in filenames if _.find('002') > 0]) == 4

    def test_dump_hyperparameters(self):
        self._clean()
        mkdir(hyperparam_path)

        prefix = 'test'
        generator = GridParameterIterator(default_dict, variation_list, prefix=prefix,
                                          enumerate_digits=3,
                                          keys=[['param0'], ['param1', 'param11', 'param110']])
        hyperparams = generator.get_hyperparameters()
        generator.dump_hyperparameters(hyperparam_path)
        hyperparam_files = listdir(hyperparam_path)
        assert len(hyperparam_files) == 12
        params_load = []
        for f in hyperparam_files:
            path = join(hyperparam_path, f)
            with open(path) as file:
                params_load.append(json.load(file))

        # Ensure that the json files are equal to the
        # hyperparam dicts from the generator
        assert all([v in params_load for v in hyperparams.values()])

        # Ensure that the json filenames match the keys
        # in the hyperparam dict
        assert all([fname in [k + '.json' for k in hyperparams.keys()] for fname in hyperparam_files])
        self._clean()

    @staticmethod
    def _clean():
        try:
            clean_folder(hyperparam_path)
            rmdir(hyperparam_path)
        except FileNotFoundError:
            pass
