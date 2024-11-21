import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from src.hyperparameters.bayesian_search.bayesian_param_iterator import BayesianParameterIterator
from src.hyperparameters.param_operation import same_dict_format, assert_param_dicts
from src.hyperparameters.random_search.random_parameter_iterator import RandomParameterIterator

np.random.seed(1234)

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

var_list_0 = [single_quantitative_continuous]
var_list_1 = [single_quantitative_continuous, group_quantitative]
start_parameter_dicts_1 = RandomParameterIterator(default_dict, var_list_0, 1).hyperparameter_dicts
start_parameter_dicts_2 = RandomParameterIterator(default_dict, var_list_1, 2).hyperparameter_dicts


class TestBayesianParameterIterator:

    def test_sample_additional_start_params(self):
        bayesian_param_it_0 = BayesianParameterIterator(default_dict, var_list_0, n_presamples=0)
        bayesian_param_it_1 = BayesianParameterIterator(default_dict, var_list_0, n_presamples=10)
        bayesian_param_it_2 = BayesianParameterIterator(
            default_dict, var_list_0, n_presamples=10, start_hyperparameter_dicts=start_parameter_dicts_1)
        bayesian_param_it_3 = BayesianParameterIterator(
            default_dict, var_list_1, n_presamples=3, start_hyperparameter_dicts=start_parameter_dicts_2)
        for p in bayesian_param_it_0.hyperparameter_vectors: assert len(p) == 1
        for p in bayesian_param_it_1.hyperparameter_vectors: assert len(p) == 1
        for p in bayesian_param_it_2.hyperparameter_vectors: assert len(p) == 1
        for p in bayesian_param_it_3.hyperparameter_vectors: assert len(p) == 2
        assert len(bayesian_param_it_0.hyperparameter_vectors) == 0
        assert len(bayesian_param_it_1.hyperparameter_vectors) == len(bayesian_param_it_2.hyperparameter_vectors) == 10
        assert len(bayesian_param_it_3.hyperparameter_vectors) == 3
        assert_param_dicts(bayesian_param_it_0.hyperparameter_dicts, var_list_0)
        assert_param_dicts(bayesian_param_it_1.hyperparameter_dicts, var_list_0)
        assert_param_dicts(bayesian_param_it_2.hyperparameter_dicts, var_list_0)
        assert_param_dicts(bayesian_param_it_3.hyperparameter_dicts, var_list_1)

    def test_next_i(self):
        # as n_presamples is higher than iterations this test will not use any bayesian optimisation
        np.random.seed(1234)
        bayesian_param_it = BayesianParameterIterator(default_dict, var_list_1, n_presamples=10,
                                                      start_hyperparameter_dicts=start_parameter_dicts_2, iterations=8)
        i = 0
        for param in bayesian_param_it:
            i += 1
            assert same_dict_format(param, default_dict)
        assert i == 8
        assert_param_dicts(bayesian_param_it.hyperparameter_dicts, var_list_1)

    def test_next_ii(self):
        # as n_presamples is smaller than iterations this test will use any bayesian optimisation
        np.random.seed(1234)
        iterations = 10
        bayesian_param_it = BayesianParameterIterator(
            default_dict, var_list_1, n_presamples=5, start_hyperparameter_dicts=start_parameter_dicts_2,
            iterations=iterations, n_random_trials=100)
        i = 0
        for param in bayesian_param_it:
            i += 1
            random_loss = np.random.rand()
            bayesian_param_it.update(random_loss)
            assert same_dict_format(param, default_dict)
        assert i == iterations
        assert_param_dicts(bayesian_param_it.hyperparameter_dicts, var_list_1)

    def test_svm(self):
        def sample_loss(params, data, target):
            return cross_val_score(
                SVC(
                    C=10 ** params[0], gamma=10 ** params[1],
                    random_state=12345),
                X=data, y=target, scoring='roc_auc', cv=3).mean()

        np.random.seed(1234)
        data, target = make_classification(n_samples=2500, n_features=45, n_informative=15, n_redundant=5,
                                           random_state=1234)

        # real_loss, param_grid = grid_search(lambdas, gammas, load_from_disk=True,
        #                                    path_to_cache=Config.path.test_data_folder)
        # max_loss = param_grid[np.array(real_loss).argmax(), :]
        default_dict = {
            'penalty_c': 0,
            'gamma': 0
        }
        var_list = [
            {'keys': ['penalty_c'], 'min': -4, 'max': 1, 'quantity_type': 'float'},
            {'keys': ['gamma'], 'min': -4, 'max': 1, 'quantity_type': 'float'}
        ]
        iterations = 100
        bayesian_it = BayesianParameterIterator(default_dict, var_list, iterations,
                                                n_presamples=5, n_random_trials=10000, return_dicts=False)
        for i, params in enumerate(bayesian_it):
            loss = sample_loss(params, data, target)
            bayesian_it.update(loss)
            print('%i/%i - %.3f' % (i, iterations, loss))
        pass
