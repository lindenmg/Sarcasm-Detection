from copy import deepcopy
from os import listdir
from os.path import join

import torch

from src.tools.config import Config
from src.tools.helpers import clean_folder
from src.strategies.dummy.dummy_cv_iterator_factory import DummyCVIteratorFactory
from src.strategies.dummy.dummy_data_factory import DummyDataFactory
from src.strategies.dummy.dummy_ffn import DummyFFN, DummyDataset
from src.training.learning_session import LearningSession

json_args = {
    'learning_session': {
        'session_tag': 'ffn00_FFN00_lemma__False_5_word_[1, 2]_5000_50_000',
        'epochs': 5,
        "mode": "train",
        "mode_to_load": "train",
        "fold_to_load": 9,
        "save_interval": 4,
        "cnn_mode": False,
        "cache_prefix": "fnn_fact",
        "seed": 1337
    },
    'logger': {
        'module': 'src.strategies.dummy.dummy_logger',
        'class_name': 'DummyLogger',
        'args': {
            'foo': 42
        }
    },
    'metrics': {
        'output_activation': 'log_softmax',
    },
    'model': {
        'module': 'src.strategies.dummy.dummy_ffn',
        'class_name': 'DummyFFN',
        'args': {
            'input_size': 100,
            'hidden_size_1': 60,
            'hidden_size_2': 30,
            'output_size': 2
        }
    },
    'criterion': {
        'module': 'torch.nn',
        'class_name': 'NLLLoss',
        'args': {
            'size_average': True
        }
    },
    'optimizer': {
        'module': 'torch.optim',
        'class_name': 'ASGD',
        'args': {
            'lr': 0.005
        }
    },
    'lr_scheduler': {
        'module': 'torch.optim.lr_scheduler',
        'class_name': 'ReduceLROnPlateau',
        'args': {
            'mode': 'min',
            'factor': 0.99,
            'patience': 20,
            'threshold_mode': 'abs',
            'threshold': 0.007
        }
    },
    'data_factory': {
        'module': 'src.strategies.dummy.dummy_data_factory',
        'class_name': 'DummyDataFactory',
        'args': {
            'pp_params': {
                'seed': 44,
                'train_examples': 2000,
                'test_examples': 200,
                'dim': 100
            }
        }
    },
    'cv_iterator_factory': {
        'module': 'src.strategies.dummy.dummy_cv_iterator_factory',
        'class_name': 'DummyCVIteratorFactory',
        'args': {
            'n_splits': 4,
            'split_ratio': 0.9,
            'batch_size': 50,
            'shuffle': False,
            'dataset_module': 'src.strategies.dummy.dummy_dataset',
            'dataset_classname': 'DummyDataset'
        }
    }
}

log_dir = Config.path.test_log_folder
cache_dir = Config.path.test_cache_folder
pp_data = join(cache_dir, 'preprocessing', 'pp_data')
pp_step_params = join(cache_dir, 'preprocessing', 'pp_step_params')
model_data = join(cache_dir, 'models', 'data')
model_params = join(cache_dir, 'models', 'parameters')
ignore_files = ['.gitfolder']


class TestLearningSession:

    @staticmethod
    def _clean():
        for path in [pp_data, pp_step_params, model_data, model_params, log_dir]:
            clean_folder(path, ignore_files)

    @staticmethod
    def _count_files_with_prefix(path, prefix):
        return len([_ for _ in listdir(path) if _.startswith(prefix)])

    def test_init(self):
        """
        This test ensures that every member object from ``LearningSession``
        is initialized according to the parameters given from ``json_args``.
        """

        args_temp = deepcopy(json_args)
        args_temp['logger']['args'].update({
            'tensorboard_log_dir': log_dir
        })
        args_temp['learning_session'].update({
            'cache_folder': cache_dir
        })
        ts = LearningSession(args_temp)

        # We have to run the run() method to initialize everything
        try:
            ts.run()
        except:
            pass

        # learning_session attributes
        d_ls = json_args['learning_session']
        assert ts.epochs == d_ls['epochs']
        assert ts.session_tag == d_ls['session_tag']
        assert ts.save_interval == 4
        assert ts.test_mode is False
        assert ts.train_mode is True
        assert ts.cache_folder == cache_dir

        # model attributes
        d_m = json_args['model']['args']
        assert isinstance(ts.model, DummyFFN)
        assert ts.model.h1.in_features == d_m['input_size']
        assert ts.model.h1.out_features == d_m['hidden_size_1']
        assert ts.model.h2.in_features == d_m['hidden_size_1']
        assert ts.model.h2.out_features == d_m['hidden_size_2']
        assert ts.model.h3.in_features == d_m['hidden_size_2']
        assert ts.model.h3.out_features == d_m['output_size']

        # criterion attributes
        d_cr = json_args['criterion']['args']
        assert isinstance(ts.criterion, torch.nn.NLLLoss)
        assert ts.criterion.size_average == d_cr['size_average']

        # optimizer attributes
        d_o = json_args['optimizer']['args']
        assert isinstance(ts.optimizer, torch.optim.ASGD)
        assert ts.optimizer.defaults['lr'] == d_o['lr']

        # lr_scheduler attributes
        d_lr = json_args['lr_scheduler']['args']
        assert isinstance(ts.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert ts.lr_scheduler.mode == d_lr['mode']
        assert ts.lr_scheduler.factor == d_lr['factor']
        assert ts.lr_scheduler.patience == d_lr['patience']
        assert ts.lr_scheduler.threshold_mode == d_lr['threshold_mode']
        assert ts.lr_scheduler.threshold == d_lr['threshold']

        # data_factory attributes
        d_df = json_args['data_factory']['args']
        assert isinstance(ts.data_factory, DummyDataFactory)
        assert ts.data_factory.train is True
        assert ts.data_factory.test is False
        assert ts.data_factory.cache_dir == cache_dir
        assert ts.data_factory.pp_params == d_df['pp_params']

        # cv_iterator_factory attributes
        d_if = json_args['cv_iterator_factory']['args']
        assert isinstance(ts.cv_iterator_factory, DummyCVIteratorFactory)
        assert isinstance(ts.cv_iterator_factory.pin_memory, bool)
        assert ts.cv_iterator_factory.datadict is not None
        assert ts.cv_iterator_factory.cv_split_indices is not None
        assert ts.cv_iterator_factory.shuffle == d_if['shuffle']
        assert ts.cv_iterator_factory.batch_size == d_if['batch_size']
        assert ts.cv_iterator_factory.split_ratio == d_if['split_ratio']
        assert ts.cv_iterator_factory.dataset_class == DummyDataset

    def test_run(self):
        """
        This test will execute a learning session according to the given
        parameters from ``json_args``. It then ensures that all resulting
        values (e.g. logs) are generated appropriately.
        """
        self._clean()
        args_temp = deepcopy(json_args)
        args_temp['logger']['args'].update({
            'tensorboard_log_dir': log_dir
        })
        args_temp['learning_session'].update({
            'cache_folder': cache_dir
        })
        ls = LearningSession(args_temp)
        ls.run()

        # Ensure logs are created
        fold_logs = ['dummy_fold_%d' % (f + 1) for f in range(json_args['cv_iterator_factory']['args']['n_splits'])]
        assert len(fold_logs) > 0
        for f in fold_logs:
            assert f in listdir(log_dir)

        # Ensure that preprocessing cache was created
        prefix = 'dummy_train_data'
        assert self._count_files_with_prefix(pp_data, prefix) > 0
        assert self._count_files_with_prefix(pp_step_params, prefix) > 0

        # Ensure that model parameters were cached
        prefix = json_args['learning_session']['session_tag']
        assert self._count_files_with_prefix(model_data, prefix) > 0
        assert self._count_files_with_prefix(model_params, prefix) > 0

        # Ensure that the logging functions were executed
        assert ls.logger.foo == json_args['logger']['args']['foo']
        assert ls.logger.session_start is True
        assert ls.logger.session_finished is True
        assert ls.logger.training_start is True
        assert ls.logger.training_finished is True
        assert ls.logger.testing_start is False
        assert ls.logger.testing_finished is False
        assert ls.logger.fold_started is True
        assert ls.logger.fold_finished is True
        assert ls.logger.epoch_finished is True
        self._clean()

    def test_early_stopping(self):
        self._clean()
        args_temp = deepcopy(json_args)
        args_temp['logger']['args'].update({
            'tensorboard_log_dir': log_dir
        })
        args_temp['learning_session'].update({
            'cache_folder': cache_dir
        })
        # the following settings will lead to overfitting
        args_temp['cv_iterator_factory']['args'].update({
            'n_splits': 1
        })
        args_temp['optimizer']['args'].update({
            'lr': 0.1
        })
        args_temp['learning_session'].update({
            'epochs': 10
        })

        ls = LearningSession(args_temp)
        summary = ls.run()

        # assures, that the loss actually went up. This is a sign for overfitting
        assert summary['validation']['sum_of_loss_change']['pos_change'][0] > 0

        # now let's add an early stopper, to stop training before overfitting starts
        args_temp.update({
            'early_stopper': {
                'args': {
                    'max_loss_increase_sum': 0.02
                },
                'module': 'src.training.base_early_stopper',
                'class_name': 'BaseEarlyStopper'
            }
        })

        ls = LearningSession(args_temp)
        _ = ls.run()
        assert len(ls.early_stopper.loss_list) < 10
        assert ls.early_stopper.max_loss_increase_sum < ls.early_stopper.sum_pos_loss_change
