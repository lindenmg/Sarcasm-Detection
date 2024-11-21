import numpy as np
import torch


class Metrics:

    def __init__(self, output_activation):
        self.reset_cumulation()
        self.loss_list = None
        self.acc_list = None
        self.loss_list_fold = np.array([])
        self.acc_list_fold = np.array([])
        if output_activation == 'log_softmax':
            self.accuracy = self._accuracy_log_softmax
        elif output_activation == 'softmax':
            self.accuracy = self._accuracy_softmax
        else:
            raise NotImplementedError('Please implement the accuracy function for your output layer')

    def update(self, ypred, labels, loss):
        self.acc_cum += self.accuracy(ypred, labels)
        self.loss_cum += loss

    def get_metrics(self, batch_size, n_batches, last_batch_size):
        loss = self.loss_cum[0] / n_batches
        acc = self.acc_cum[0] / (batch_size * (n_batches - 1) + last_batch_size)
        self.acc_list_fold = np.append(self.acc_list_fold, acc)
        self.loss_list_fold = np.append(self.loss_list_fold, loss)
        self.reset_cumulation()
        return loss, acc

    @staticmethod
    def _accuracy_log_softmax(y_pred, labels):
        class_pred = torch.exp(y_pred).max(dim=1)[1]
        return (class_pred == labels).sum()

    @staticmethod
    def _accuracy_softmax(y_pred, labels):
        raise NotImplementedError

    def fold_finished(self):
        if self.loss_list is None:
            self.loss_list = self.loss_list_fold.copy()
        else:
            self.loss_list = np.vstack((self.loss_list, self.loss_list_fold))
        if self.acc_list is None:
            self.acc_list = self.acc_list_fold.copy()
        else:
            self.acc_list = np.vstack((self.acc_list, self.acc_list_fold))
        self.loss_list_fold = np.array([])
        self.acc_list_fold = np.array([])

    def get_summary(self):
        def get_epoch_difference(m):
            return (np.hstack((m[:, 1:], np.zeros(m.shape[0]).reshape(-1, 1))) - m)[:, :-1]

        if len(self.acc_list.shape) == 1:
            self.acc_list = self.acc_list[np.newaxis, :]
        if len(self.loss_list.shape) == 1:
            self.loss_list = self.loss_list[np.newaxis, :]
        # change_diff_acc = get_epoch_difference(self.acc_list)
        change_diff_loss = get_epoch_difference(self.loss_list)
        change_diff_sum_loss_neg = np.sum((change_diff_loss < 0) * change_diff_loss, axis=1)
        change_diff_sum_loss_pos = np.sum((change_diff_loss > 0) * change_diff_loss, axis=1)
        return {
            'mean_last_epoch_acc': self.acc_list[:, -1].mean(),
            'mean_last_epoch_loss': self.loss_list[:, -1].mean(),
            'last_epoch_acc': self.acc_list[:, -1].tolist(),
            'last_epoch_loss': self.loss_list[:, -1].tolist(),
            'sum_of_loss_change': {
                'pos_change': change_diff_sum_loss_pos.tolist(),
                'neg_change': change_diff_sum_loss_neg.tolist(),
            }
        }

    def reset_cumulation(self):
        if torch.cuda.is_available():
            self.acc_cum = torch.FloatTensor(1).zero_().cuda()
            self.loss_cum = torch.FloatTensor(1).zero_().cuda()
        else:
            self.acc_cum = torch.FloatTensor(1).zero_()
            self.loss_cum = torch.FloatTensor(1).zero_()

    acc_cum = None
    loss_cum = None
