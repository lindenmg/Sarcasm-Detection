from abc import ABC, abstractmethod


class AbstractLogger(ABC):

    @abstractmethod
    def log_session_start(self, session_args):
        pass

    @abstractmethod
    def log_session_finished(self):
        pass

    @abstractmethod
    def log_training_start(self):
        pass

    @abstractmethod
    def log_training_finished(self):
        pass

    @abstractmethod
    def log_training_aborted(self):
        pass

    @abstractmethod
    def log_testing_start(self):
        pass

    @abstractmethod
    def log_testing_finished(self):
        pass

    @abstractmethod
    def log_fold_start(self, fold):
        pass

    @abstractmethod
    def log_fold_finished(self, fold):
        pass

    @abstractmethod
    def log_epoch_finished(self, epoch, train_acc, train_loss, val_acc, val_loss, lr):
        pass
