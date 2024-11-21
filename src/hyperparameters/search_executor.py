from copy import deepcopy

from src.hyperparameters.param_operation import traverse_get
from src.training.learning_session import LearningSession


class SearchExecutor:
    def __init__(self, param_iterator, logger, cacher, tag, performance_key=['validation', 'mean_last_epoch_loss'],
                 performance_threshold=None):
        """

        Parameters
        ----------
        param_iterator: subclass of AbstractParameterIterator
            provides the hyperparameters for every iteration
        logger: subclass of AbstractExecutorLogger
            the logger provides functions for logging at the beginning
            and the end of a parameter test. The executor executes
            these functions
        cacher: subclass of AbstractCacher
            the cacher provides functions for caching the results of
            a training session with certain parameters
        tag: str
            a tag for identifying the search session
        performance_threshold: float
            The threshold (e.g. loss), at which the parameter search is aborted.
            If None, it will search until the maximum number of
            iterations is reached.
        """
        self.tag = tag
        self.logger = logger
        self.logger.set_cache_dir(tag)
        self.cacher = cacher
        self.performance_key = performance_key
        self.performance_threshold = performance_threshold
        self.param_iterator = param_iterator

    def run(self):
        """
        Starts the hyperparameter search
        """
        for param in self.param_iterator:
            self.logger.log_start_hyperparam_test(param)
            summary = self.cacher.eventually_load_cache(param, self.tag)
            if summary is None:
                session = LearningSession(deepcopy(param))
                summary = session.run()
                self.cacher.create_cache(summary, param, self.tag)
            performance = traverse_get(summary, self.performance_key)
            self.param_iterator.update(performance)
            self.logger.log_end_hyperparam_test(summary)
        self.logger.log_end_hyperparam_search()
