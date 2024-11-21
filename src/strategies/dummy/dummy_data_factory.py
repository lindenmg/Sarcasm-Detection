import numpy as np
from sklearn.datasets import make_classification

from src.training.abstract_data_factory import AbstractDataFactory


class DummyDataFactory(AbstractDataFactory):
    """
    Factory class for generating test preprocessing.
    """

    def __init__(self, **args):
        """

        Parameters
        ----------
        **args: dict
            The init parameters as defined in AbstractDataFactory.__init__
        """
        super().__init__(**args)

    def get_data(self):
        """
        Returns random preprocessing for testing
        Implementation of abstractmethod
        Returns
        -------
        dict, dict
            First dict contains preprocessing, labels for training.
            Second dict contains preprocessing, labels, for testing
        """

        train_data = None
        test_data = None
        np.random.seed(self.pp_params['seed'])
        dim = self.pp_params['dim']
        n_train_examples = self.pp_params['train_examples']
        n_test_examples = self.pp_params['test_examples']
        if self.train:
            pp_step_params = {'n_samples': n_train_examples, 'n_features': dim}
            train_data = self.eventually_load_cache(pp_step_params, 'dummy_train_data_')
            if train_data is None:
                train_data = self._generate_data(pp_step_params, 'dummy_train_data_')
        if self.test:
            pp_step_params = {'n_samples': n_test_examples, 'n_features': dim}
            test_data = self.eventually_load_cache(pp_step_params, 'dummy_test_data_')
            if test_data is None:
                test_data = self._generate_data(pp_step_params, 'dummy_test_data_')

        return {'train_data': train_data, 'test_data': test_data}

    def _generate_data(self, pp_step_params, tag):
        d = make_classification(**pp_step_params)
        data = {'data': d[0], 'labels': d[1]}
        self.create_cache(data, pp_step_params, tag)
        return data
