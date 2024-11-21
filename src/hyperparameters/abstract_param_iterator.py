from abc import ABC, abstractmethod

from src.hyperparameters.param_operation import sort_variation_list


class AbstractParameterIterator(ABC):
    def __init__(self, default_dict, variation_list, iterations):
        self._variation_list = sort_variation_list(variation_list)
        self._default_dict = default_dict
        self._iterations = iterations
        self.hyperparameter_dicts = dict()
        self.hyperparameter_vectors = []
        self._i = 0

    @abstractmethod
    def update(self, loss):
        pass

    @abstractmethod
    def get_next_hyperparameter_dict(self, i):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._iterations:
            params = self.get_next_hyperparameter_dict(self._i)
            self._i += 1
            return params
        else:
            raise StopIteration()
