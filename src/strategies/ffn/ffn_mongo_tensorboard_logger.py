import os

from pymongo import MongoClient
from tensorboardX import SummaryWriter

from src.tools.helpers import get_time_duration
from src.tools.helpers import hash_dict
from src.training.abstract_logger import AbstractLogger


class FFNMongoTensorboardLogger(AbstractLogger):
    def __init__(self, mongo_host, mongo_port, tensorboard_log_dir,
                 connect_timeout_ms, server_selection_timeout_ms):
        self.tensorboard_writer = None
        self.session_tag = None
        self.session_id = None
        self.client = None
        self.fold = None
        self.log = None
        self.db = None

        self._args_hash = None

        if tensorboard_log_dir is not None:
            self.tensorboard_log_dir = tensorboard_log_dir

        if mongo_host is not None:
            self.client = MongoClient(mongo_host, mongo_port,
                                      connectTimeoutMS=connect_timeout_ms,
                                      serverSelectionTimeoutMS=server_selection_timeout_ms)
            self.db = self.client.session_db
            self.log = self.db.log

    def log_session_start(self, session_args):
        self.session_tag = session_args['learning_session']['session_tag']
        print('START SESSION %s' % self.session_tag)
        if self.db is not None:
            self._args_hash = hash_dict(session_args)
            self.session_id = self.db.session_args.insert_one(session_args).inserted_id

    def log_session_finished(self):
        print('FINISHED SESSION %s' % self.session_tag)
        if self.client is not None:
            self.client.close()

    def log_training_start(self):
        print('START TRAINING')

    def log_training_finished(self):
        print('FINISHED TRAINING')

    def log_training_aborted(self):
        print('TRAINING ABORTED. reached max loss increase')

    def log_testing_start(self):
        print('START TESTING')

    def log_testing_finished(self):
        print('FINISHED TESTING')

    def log_fold_start(self, fold):
        print('START FOLD %d' % fold)
        self.fold = fold
        if self.tensorboard_writer is not None:
            self.tensorboard_writer = self._create_tensorboard_logger(fold)

    def log_fold_finished(self, fold):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def log_epoch_finished(self, epoch, train_acc, train_loss, val_acc, val_loss, lr):
        if self.log is not None:
            self.log.insert_one({
                '_session_id': self.session_id,
                'fold': self.fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': lr
            })
        print(
            ('[time: %5s | fold: %2d | epoch: %2d], lr: %1.2e, train_loss: %.4f, ' +
             'train_acc: %.4f, val_loss: %.4f, val_acc: %.4f') % (
                get_time_duration(), self.fold, epoch, lr, train_loss, train_acc, val_loss, val_acc))

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalars('learning_rate', {'learning_rate': lr}, epoch)
            self.tensorboard_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            self.tensorboard_writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

    def _create_tensorboard_logger(self, fold):
        writer = None
        if self.tensorboard_log_dir is not None:
            path = os.path.join(self.tensorboard_log_dir, 'fold_%d' % fold,
                                self._args_hash + '_' + self.session_tag)
            writer = SummaryWriter(log_dir=path)
        return writer
