import numpy as np

from src.hyperparameters.abstract_param_iterator import AbstractParameterIterator
from src.hyperparameters.param_operation import params_vector_to_dict


class RandomParameterIterator(AbstractParameterIterator):

    def __init__(self, default_dict, variation_list, iterations):
        """
        Parameters
        ----------
        default dict: dict
            dict with all default values
        variation_dict: dict
            dict with all values for which variations shall be generated.
            The values from which variations shall be in a list and the
            corresponding key must be equal to one key in the default_dict
        iterations: int
            the number of parameter settings the iterator should return in an iteration.
        """
        super().__init__(default_dict, variation_list, iterations)
        self._generate_param_vectors()
        self._generate_param_dicts()

    def update(self, loss):
        pass

    def get_next_hyperparameter_dict(self, i):
        return self.hyperparameter_dicts[i]

    def _generate_param_vectors(self):
        """
        Generates the hyperparameter variations
        Returns
        -------
        list of dict
            All hyperparameter variations, generated from init parameters
            `default_dict` and `variation_dict`.
        """

        def _sample_variation(var):
            is_group = isinstance(var['keys'], tuple)

            # case: value is qualitative
            if 'val' in var.keys():
                if is_group:
                    max_idx = len(var['val'][0])
                else:
                    max_idx = len(var['val'])
                rand = np.random.randint(0, max_idx)

            # case: value is quantitative
            else:
                if is_group:
                    # The values in a group are all proportional to max/min.
                    # Hence, only the first value is sampled.
                    # The other values are recreated using proportionality.
                    min_val = var['min'][0]
                    max_val = var['max'][0]
                else:
                    min_val = var['min']
                    max_val = var['max']
                rand = np.random.uniform(min_val, max_val)
            return rand

        self.hyperparameter_vectors = [[_sample_variation(var) for var in self._variation_list]
                                       for _ in range(self._iterations)]

    def _generate_param_dicts(self):
        self.hyperparameter_dicts = [params_vector_to_dict(self._default_dict, self._variation_list, v)
                                     for v in self.hyperparameter_vectors]
