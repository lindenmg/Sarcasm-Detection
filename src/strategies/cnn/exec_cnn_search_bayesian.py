import os
import sys

project_root_path = os.path.realpath(os.path.join(__file__, "..", "..", "..", ".."))
sys.path.append(project_root_path)

import json
from pathlib import Path
import numpy as np

from src.tools.config import Config
from src.hyperparameters.search_executor import SearchExecutor
from src.hyperparameters.bayesian_search.bayesian_param_iterator import BayesianParameterIterator
from src.hyperparameters.base_executor_logger import BaseExecutorLogger
from src.training.base_cacher import BaseCacher


def __main__():
    """
    the main function for a random hyperparameter search.
    It will test a certain number of hyperparameter settings
    (as specified in `iterations`).
    """
    default_dict = get_default_dict()
    variation_list = get_variation_list()
    np.random.seed(1234)
    iterations = 500
    executor_logger = BaseExecutorLogger(variation_list, iterations)
    executor_cacher = BaseCacher(Config.path.cache_folder)
    param_iterator = BayesianParameterIterator(default_dict, variation_list, iterations=iterations,
                                               n_presamples=5, maximize_performance=True)
    executor = SearchExecutor(param_iterator, executor_logger, executor_cacher, tag='bayesian_search',
                              performance_key=['validation', 'mean_last_epoch_acc'])
    executor.run()


def get_default_dict():
    """
    This function provides a dict with all the default parameters,
    that are used the learning session.
    Returns
    -------
    dict
        the default arguments for every learning session
    """
    train_pipe_config_path = Path(Config.path.project_root_folder) / 'src' / 'strategies'
    train_pipe_config_path = train_pipe_config_path / 'cnn' / 'cnn_default_config.json'

    with open(str(train_pipe_config_path)) as file:
        default_dict = json.load(file)

    logger_args = {
        'tensorboard_log_dir': Config.path.log_folder,
        'mongo_host': 'localhost',
        'mongo_port': Config.logging.port
    }
    default_dict['logger']['args'].update(logger_args)
    default_dict['learning_session'].update({'cache_folder': Config.path.cache_folder})
    return default_dict


def get_variation_list():
    """
    The variation list contains all values, that are varied in the hyperparameter search.
    Returns
    -------
    list of dicts
        every dict in the list specifies key for the hyperparameter, which shall be varied,
        as well as the constraints for the variation (e.g. min and max).
    """

    x = np.linspace(0, 10, 100)
    y = (0.5558215 + (2.886274e-8 - 0.5558215) / (1 + (x / 9.857313) ** 11.08463)).tolist()

    return [
        {'keys': ['optimizer', 'args', 'lr'], 'val': y},
        {'keys': ['cv_iterator_factory', 'args', 'batch_size'], 'min': 10, 'max': 1000, 'quantity_type': 'int'},
        {'keys': ['cv_iterator_factory', 'args', 'batch_search_window'], 'min': 2, 'max': 10, 'quantity_type': 'int'},
        {'keys': ['model', 'args', 'hl1_kernel_num'], 'min': 5, 'max': 1000, 'quantity_type': 'int'},
        {'keys': ['model', 'args', 'hl1_kernels'], 'val': [[[i, 300]] for i in range(1, 21)]}
    ]


if __name__ == '__main__':
    __main__()
