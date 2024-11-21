from abc import ABC
from abc import abstractmethod


class AbstractExecutorLogger(ABC):

    @abstractmethod
    def log_start_hyperparam_test(self, param):
        pass

    @abstractmethod
    def log_end_hyperparam_test(self, summary):
        pass

    @abstractmethod
    def log_end_hyperparam_search(self):
        pass
