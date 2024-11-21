from src.strategies.ffn.main_hyperparam_generation_00 import variation_list, default_ffn_params, filename_keys
from src.hyperparameters.grid_search.grid_param_iterator import GridParameterIterator
from src.hyperparameters.param_operation import traverse_get


class TestMainHyperparamGenerator00:
    def test(self):
        generator = GridParameterIterator(
            default_dict=default_ffn_params, variation_list=variation_list, prefix="ffn00",
            enumerate_digits=3, keys=filename_keys)
        hyperparams = list(generator.get_hyperparameters().values())

        for var in variation_list:
            if not isinstance(var['keys'], tuple):
                for v in var['val']:
                    assert len(hyperparams) / len(var['val']) == \
                           len([p for p in hyperparams if traverse_get(p, var['keys']) == v])
            else:
                for i in range(len(var['keys'])):
                    for v in var['val'][i]:
                        assert len(hyperparams) / len(var['val'][i]) == \
                               len([p for p in hyperparams if traverse_get(p, var['keys'][i]) == v])
