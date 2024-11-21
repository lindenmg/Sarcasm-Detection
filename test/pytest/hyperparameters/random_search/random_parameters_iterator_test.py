from src.hyperparameters.random_search.random_parameter_iterator import RandomParameterIterator
from src.hyperparameters.param_operation import assert_param_dicts
import numpy as np

default_dict = {
    'param0': 'aloha',
    'param1': {
        'param10': 1,
        'param11': {
            'param110': 0,
            'param111': 111,
        },
        'param12': 'one'
    },
    'param2': 3,
    'param3': 0,
    'param4': 0,
    'param5': 10
}

single_qualitative = {
    'keys': ['param0'],
    'val': ['aloha', 'servus']
}

single_quantitative_discrete_0 = {
    'keys': ['param5'],
    'min': 4,
    'max': 17,
    'quantity_type': 'int'
}

single_quantitative_discrete_1 = {
    'keys': ['param1', 'param11', 'param111'],
    'min': 11,
    'max': 1111,
    'quantity_type': 'int'
}

single_quantitative_continuous = {
    'keys': ['param2'],
    'min': 0.5,
    'max': 5.5,
    'quantity_type': 'float'
}

group_qualitative = {
    'keys': (['param1', 'param10'], ['param1', 'param12']),
    'val': ([1, 2, 3], ['one', 'two', 'three'])
}

group_quantitative = {
    'keys': (['param3'], ['param4']),
    'min': (0, 0),
    'max': (5, 100),
    'quantity_type': ('float', 'int')
}


class TestRandomParametersIterator:

    def test_i(self):
        np.random.seed(1234)
        var_list = [single_qualitative, single_quantitative_continuous, single_quantitative_discrete_0,
                    group_qualitative, group_quantitative]
        iterations = 4
        random_param_it = RandomParameterIterator(default_dict, var_list, iterations)
        assert len(random_param_it.hyperparameter_vectors) == iterations
        assert len(random_param_it.hyperparameter_vectors[0]) == len(var_list)
        assert_param_dicts(random_param_it.hyperparameter_dicts, var_list)

    def test_ii(self):
        np.random.seed(1234)
        var_list = [single_quantitative_discrete_1]
        iterations = 4
        random_param_it = RandomParameterIterator(default_dict, var_list, iterations)
        assert len(random_param_it.hyperparameter_vectors) == iterations
        assert len(random_param_it.hyperparameter_vectors[0]) == len(var_list)
        assert_param_dicts(random_param_it.hyperparameter_dicts, var_list)

    def test_iii(self):
        np.random.seed(1234)
        var_list = [single_quantitative_discrete_1, group_quantitative]
        iterations = 4
        random_param_it = RandomParameterIterator(default_dict, var_list, iterations)
        assert len(random_param_it.hyperparameter_vectors) == iterations
        assert len(random_param_it.hyperparameter_vectors[0]) == len(var_list)
        assert_param_dicts(random_param_it.hyperparameter_dicts, var_list)

    def test_iv(self):
        np.random.seed(1234)
        var_list = [single_qualitative, single_quantitative_continuous, single_quantitative_discrete_0,
                    group_qualitative, group_quantitative]
        iterations = 200
        random_param_it = RandomParameterIterator(default_dict, var_list, iterations)
        assert len(random_param_it.hyperparameter_vectors) == iterations
        assert len(random_param_it.hyperparameter_vectors[0]) == len(var_list)
        assert_param_dicts(random_param_it.hyperparameter_dicts, var_list)
