from src.training.base_early_stopper import BaseEarlyStopper
import numpy as np


class TestBaseEarlyStopper:
    def test_i(self):
        x = np.linspace(0, 9, 10)
        loss = (x - 5) ** 2
        es = BaseEarlyStopper(max_loss_increase_sum=13)
        n_should_stop = 0
        for l in loss:
            es.epoch_finished(val_loss=l, val_acc=0)
            n_should_stop += es.should_stop()
        assert n_should_stop == 1
