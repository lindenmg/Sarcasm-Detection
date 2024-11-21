import numpy as np
import sklearn.gaussian_process as gp

from src.hyperparameters.abstract_param_iterator import AbstractParameterIterator
from src.hyperparameters.bayesian_search.acquisition_function import expected_improvement
from src.hyperparameters.param_operation import params_dict_to_vector, params_vector_to_dict
from src.hyperparameters.random_search.random_parameter_iterator import RandomParameterIterator


class BayesianParameterIterator(AbstractParameterIterator):
    """
    * add a logger, which is provided by the user or by default value in __init__
    """

    def __init__(self, default_dict, variation_list, iterations=5, gp_params=None,
                 start_hyperparameter_dicts=None, n_presamples=5, n_random_trials=10000,
                 acquisition_function=expected_improvement, return_dicts=True, maximize_performance=True):
        super().__init__(default_dict, variation_list, iterations)
        self._acquisition_function = acquisition_function
        self._n_presamples = n_presamples
        self._n_random_trials = n_random_trials
        self._performance_list = []
        self._gp_model = None
        self._maximize_performance = maximize_performance
        self._last_param_conf = None
        self._return_dicts = return_dicts
        self._generate_param_dicts(start_hyperparameter_dicts)
        self._generate_param_vectors()
        self._init_gp_model(gp_params)

    def _generate_param_vectors(self):
        self.hyperparameter_vectors = [params_dict_to_vector(self._variation_list, d)
                                       for d in self.hyperparameter_dicts]

    def _generate_param_dicts(self, start_hyperparameter_dicts):
        """
        The function will generate parameters with RandomParameterIterator
        and add them to self.hyperparameter_vectors.
        """
        h_dicts = [] if start_hyperparameter_dicts is None else start_hyperparameter_dicts
        n = self._n_presamples
        # random samples are added if number of vectors in self.start_vectors is smaller than self.n_presamples
        n = np.max((0, n - len(h_dicts)))
        random_param_it = RandomParameterIterator(self._default_dict, self._variation_list, n)
        h_dicts.extend(random_param_it.hyperparameter_dicts)
        # if start_vectors has too many elements (bigger than n_presamples)
        if len(h_dicts) > self._n_presamples:
            h_dicts = h_dicts[:self._n_presamples]
        self.hyperparameter_dicts = h_dicts

    def _init_gp_model(self, gp_params):
        if gp_params is not None:
            self._gp_model = gp.GaussianProcessRegressor(**gp_params)
        else:
            kernel = gp.kernels.Matern()
            self._gp_model = gp.GaussianProcessRegressor(
                kernel=kernel, alpha=1e-5, n_restarts_optimizer=10, normalize_y=True)

    def get_next_hyperparameter_dict(self, i):
        if i < self._n_presamples:
            # It uses the start hyperparameter dicts, until all of them were used....
            if self._return_dicts:
                params = self.hyperparameter_dicts[i]
            else:
                params = self.hyperparameter_vectors[i]
        else:
            # ... afterward it will use bayesian optimisation
            self._sample_next_hyperparam()
            if self._return_dicts:
                params = self.hyperparameter_dicts[-1]
            else:
                params = self.hyperparameter_vectors[-1]
        return params

    def _sample_next_hyperparam(self):
        """
        The procedure was adopted from
        https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
        """
        self._gp_model.fit(self.hyperparameter_vectors, self._performance_list)
        x_random = RandomParameterIterator(self._default_dict, self._variation_list,
                                           self._n_random_trials).hyperparameter_vectors
        ei = self._acquisition_function(x_random, self._gp_model, self._performance_list, self._maximize_performance)
        next_hyperparameter_vector = x_random[np.argmax(ei)]
        next_hyperparameter_dict = params_vector_to_dict(self._default_dict, self._variation_list,
                                                         next_hyperparameter_vector)
        self.hyperparameter_vectors.append(next_hyperparameter_vector)
        self.hyperparameter_dicts.append(next_hyperparameter_dict)
        return next_hyperparameter_dict

    def update(self, performance):
        self._performance_list.append(performance)
