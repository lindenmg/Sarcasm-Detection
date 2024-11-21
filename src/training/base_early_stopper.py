from src.training.abstract_early_stopper import AbstractEarlyStopper


class BaseEarlyStopper(AbstractEarlyStopper):
    def __init__(self, max_loss_increase_sum):
        self.max_loss_increase_sum = max_loss_increase_sum
        self._reset()

    def epoch_finished(self, val_loss, val_acc):
        if len(self.loss_list) > 0:
            d_loss = val_loss - self.loss_list[-1]
            if d_loss > 0:
                self.sum_pos_loss_change += d_loss
        self.loss_list.append(val_loss)

    def should_stop(self):
        return self.sum_pos_loss_change > self.max_loss_increase_sum

    def fold_finished(self):
        self._reset()

    def _reset(self):
        self.sum_pos_loss_change = 0
        self.loss_list = []
