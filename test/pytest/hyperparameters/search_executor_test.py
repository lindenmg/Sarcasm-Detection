from src.hyperparameters.random_search.random_parameter_iterator import RandomParameterIterator
from src.hyperparameters.grid_search.grid_param_iterator import GridParameterIterator
from src.hyperparameters.search_executor import SearchExecutor
from src.hyperparameters.base_executor_logger import BaseExecutorLogger
from src.training.base_cacher import BaseCacher
from src.tools.config import Config
import numpy as np

# In this setting, the data is created by the data_factory itself. Normally this makes
# no sense, but this is a test/example, and it shows the flexibility of the framework.

default_dict = {"data_factory": {"module": "src.strategies.dummy.dummy_data_factory", "class_name": "DummyDataFactory",
                                 "args": {"pp_params": {"train_examples": 2000, "test_examples": 200, "dim": 50,
                                                        "seed": 44}}},
                "optimizer": {"module": "torch.optim", "class_name": "ASGD", "args": {"lr": 0.05}},
                "metrics": {"output_activation": "log_softmax"},
                "criterion": {"module": "torch.nn", "class_name": "NLLLoss", "args": {"size_average": True}},
                "lr_scheduler": {"module": "torch.optim.lr_scheduler", "class_name": "ReduceLROnPlateau",
                                 "args": {"threshold": 0.007, "mode": "min", "patience": 20, "threshold_mode": "abs",
                                          "factor": 0.99}},
                "cv_iterator_factory": {"module": "src.strategies.dummy.dummy_cv_iterator_factory",
                                        "class_name": "DummyCVIteratorFactory", "args": {"shuffle": False,
                                                                                         "dataset_module": "src.strategies.dummy.dummy_dataset",
                                                                                         "n_splits": 2,
                                                                                         "split_ratio": 0.9,
                                                                                         "dataset_classname": "DummyDataset",
                                                                                         "batch_size": 50}},
                "model": {"module": "src.strategies.dummy.dummy_ffn", "class_name": "DummyFFN",
                          "args": {"hidden_size_1": 60, "hidden_size_2": 5, "output_size": 2, "input_size": 50}},
                "logger": {"module": "src.strategies.dummy.dummy_logger", "class_name": "DummyLogger",
                           "args": {"foo": 42, 'tensorboard_log_dir': Config.path.log_folder}},
                "learning_session": {"epochs": 10, "mode": "train", "mode_to_load": "train", "fold_to_load": 9,
                                     "save_interval": -1, "cnn_mode": False, "seed": 1234,
                                     "session_tag": "learning_session_test", 'cache_folder': Config.path.cache_folder}}


class TestSearchExecutor:

    def test_run_grid_search(self):
        # the parameter iterators will vary values according to the specifications in the variation list.
        # Any parameter - nominal, ordinal, discrete, continuous, etc. - could be varied
        variation_list = [
            {'keys': ['model', 'args', 'hidden_size_1'], 'val': [15, 20]},
            {'keys': ['cv_iterator_factory', 'args', 'batch_size'], 'val': [30, 70]}
        ]
        np.random.seed(1234)
        iterations = 3
        param_iterator = GridParameterIterator(default_dict, variation_list, iterations=iterations)
        logger = BaseExecutorLogger(variation_list, iterations)
        cacher = BaseCacher(Config.path.cache_folder)
        executor = SearchExecutor(param_iterator=param_iterator, logger=logger,
                                  cacher=cacher, tag='search_executor_test')
        executor.run()

        summary_with_most_overfitting = self.__most_overfitting(logger.summary_list)
        summary_with_highest_val_acc_mean = self.__highest_val_acc_mean(logger.summary_list)
        summary_with_lowest_val_loss_mean = self.__lowest_val_loss_mean(logger.summary_list)
        assert np.isclose(summary_with_most_overfitting['validation']['sum_of_loss_change']['pos_change'][0], 0)
        assert np.isclose(summary_with_highest_val_acc_mean['validation']['mean_last_epoch_acc'], 0.8875)
        assert np.isclose(summary_with_lowest_val_loss_mean['validation']['mean_last_epoch_loss'], 0.30726281801859534)
        assert len(logger.summary_list) == param_iterator._iterations

    def test_run_random_search(self):
        # the parameter iterators will vary values according to the specifications in the variation list.
        # Any parameter - nominal, ordinal, discrete, continuous, etc. - could be varied
        variation_list = [
            {'keys': ['model', 'args', 'hidden_size_1'], 'min': 15, 'max': 20, 'quantity_type': 'int'},
            {'keys': ['model', 'args', 'hidden_size_2'], 'min': 5, 'max': 10, 'quantity_type': 'int'},
            {'keys': ['cv_iterator_factory', 'args', 'batch_size'], 'min': 10, 'max': 100, 'quantity_type': 'int'}
        ]
        np.random.seed(1234)
        iterations = 3
        param_iterator = RandomParameterIterator(default_dict, variation_list, iterations=iterations)
        logger = BaseExecutorLogger(variation_list, iterations)
        cacher = BaseCacher(Config.path.cache_folder)
        executor = SearchExecutor(param_iterator=param_iterator, logger=logger,
                                  cacher=cacher, tag='search_executor_test')
        executor.run()

        summary_with_most_overfitting = self.__most_overfitting(logger.summary_list)
        summary_with_highest_val_acc_mean = self.__highest_val_acc_mean(logger.summary_list)
        summary_with_lowest_val_loss_mean = self.__lowest_val_loss_mean(logger.summary_list)

        assert np.isclose(summary_with_most_overfitting['validation']['sum_of_loss_change']['pos_change'][0],
                          0.015183528264363588)
        assert np.isclose(summary_with_highest_val_acc_mean['validation']['mean_last_epoch_acc'], 0.895)
        assert np.isclose(summary_with_lowest_val_loss_mean['validation']['mean_last_epoch_loss'], 0.2676094273726145)
        assert len(logger.summary_list) == iterations

# def test_run_bayesian_search(self):
#     pass
#  _          _
# | |__   ___| |_ __   ___ _ __ ___
# | '_ \ / _ \ | '_ \ / _ \ '__/ __|
# | | | |  __/ | |_) |  __/ |  \__ \
# |_| |_|\___|_| .__/ \___|_|  |___/
#              |_|

    @staticmethod
    def __highest_val_acc_mean(summary_list):
        summary = summary_list[0]
        for s in summary_list:
            if s['validation']['mean_last_epoch_acc'] > \
                    summary['validation']['mean_last_epoch_acc']:
                summary = s
        return summary

    @staticmethod
    def __lowest_val_loss_mean(summary_list):
        summary = summary_list[0]
        for s in summary_list:
            if s['validation']['mean_last_epoch_loss'] < \
                    summary['validation']['mean_last_epoch_loss']:
                summary = s
        return s

    @staticmethod
    def __most_overfitting(summary_list):
        summary = summary_list[0]
        for s in summary_list:
            if max(s['validation']['sum_of_loss_change']['pos_change']) > \
                    max(summary['validation']['sum_of_loss_change']['pos_change']):
                summary = s
        return s
