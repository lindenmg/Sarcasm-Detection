from abc import ABC
from abc import abstractmethod


class AbstractEarlyStopper(ABC):

    @abstractmethod
    def epoch_finished(self, val_loss, val_acc):
        pass

    @abstractmethod
    def fold_finished(self):
        pass

    @abstractmethod
    def should_stop(self):
        pass
