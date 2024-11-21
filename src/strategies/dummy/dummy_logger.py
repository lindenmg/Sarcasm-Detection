import os

from tensorboardX import SummaryWriter

from src.tools.helpers import get_time_duration
from src.training.abstract_logger import AbstractLogger


class DummyLogger(AbstractLogger):
    def __init__(self, foo, tensorboard_log_dir):
        self.foo = foo
        self.tensorboard_log_dir = tensorboard_log_dir
        self.tensorboard_writer = None
        self.session_start = False
        self.session_finished = False
        self.training_start = False
        self.training_finished = False
        self.testing_start = False
        self.testing_finished = False
        self.fold_started = False
        self.fold_finished = False
        self.epoch_finished = False
        self.fold = 0

    def log_session_start(self, session_args):
        self.session_start = True

    def log_session_finished(self):
        self.session_finished = True

    def log_training_start(self):
        self.training_start = True

    def log_training_finished(self):
        self.training_finished = True

    def log_training_aborted(self):
        print('========= training aborted. reached max loss increase ============')

    def log_testing_start(self):
        self.testing_start = True

    def log_testing_finished(self):
        self.testing_finished = True

    def log_fold_start(self, fold):
        self.fold_started = True
        self.fold = fold
        self.tensorboard_writer = self._create_tensorboard_logger(fold)

    def log_fold_finished(self, fold):
        self.fold_finished = True

    def log_epoch_finished(self, epoch, train_acc, train_loss, val_acc, val_loss, lr):
        self.epoch_finished = True
        print(
            ('[time: %5s | fold: %2d | epoch: %2d], lr: %1.2e, train_loss: %.4f, ' +
             'train_acc: %.4f, val_loss: %.4f, val_acc: %.4f') % (
                get_time_duration(), self.fold, epoch, lr, train_loss, train_acc, val_loss, val_acc))
        self.tensorboard_writer.add_scalars('learning_rate', {'learning_rate': lr}, epoch)
        self.tensorboard_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        self.tensorboard_writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

    def _create_tensorboard_logger(self, fold):
        writer = None
        if self.tensorboard_log_dir is not None:
            path = os.path.join(self.tensorboard_log_dir, 'dummy_fold_%d' % fold)
            writer = SummaryWriter(log_dir=path)
        return writer
