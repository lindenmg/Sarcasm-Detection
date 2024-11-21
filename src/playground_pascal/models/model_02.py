import inspect
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from src.playground_pascal.preprocessing.pp_02 import get_tfidf, get_labels
from src.preprocessing.datahandler import DataHandler
from src.tools.config import Config
from src.tools.helpers import eventually_load_params, tensorboard_log

#           _   _   _
#  ___  ___| |_| |_(_)_ __   __ _ ___
# / __|/ _ \ __| __| | '_ \ / _` / __|
# \__ \  __/ |_| |_| | | | | (_| \__ \
# |___/\___|\__|\__|_|_| |_|\__, |___/
#                           |___/

# Sadly, using td-idf every epoch would be a bit to slow... Stay therefore at 1x Val and 1x Train for some time†
data_size = 1000
tfidf_size = None  # take full vector length of preprocessing
input_post = 20
input_reply = 20
hidden_layer_1 = int(input_reply + input_post)
batch_size = 1000
learning_rate = 2e-1
post_dropout = 0.5
reply_dropout = 0.5
hl1_dropout = 0.5
save_interval = 5
n_epoch = 10
cv_folds = 1
cv_split = 1 - (1 / 9)
session = '02_01'
params_path = None


#                               _
#  _ __  _ __ ___   ___ ___  __| |_   _ _ __ ___
# | '_ \| '__/ _ \ / __/ _ \/ _` | | | | '__/ _ \
# | |_) | | | (_) | (_|  __/ (_| | |_| | | |  __/
# | .__/|_|  \___/ \___\___|\__,_|\__,_|_|  \___|
# |_|

# Das eventually_cuda() war Overhead, jetzt nicht mehr. Wir trainieren letztendlich sowieso nur auf der GPU
# Da ginge noch mehr hinsichtlich speed, aber das kann noch warten.

def run(data_size, tfidf_size, input_post, input_reply, hidden_layer_1, batch_size, learning_rate, post_dropout,
        reply_dropout, hl1_dropout, save_interval, n_epoch, cv_folds, cv_split, params_path):
    global net, optimizer, criterion, lr_scheduler
    param_s = param_str(frame=inspect.currentframe())
    torch.manual_seed(1337)

    data, labels, _ = load_data(train=True, test=False, data_size=data_size, tfidf_size=tfidf_size)

    net = Model00(in_post=tfidf_size, in_reply=tfidf_size, input_post=input_post,
                  input_reply=input_reply, hidden_layer_1=hidden_layer_1,
                  post_dropout=post_dropout, reply_dropout=reply_dropout, hl1_dropout=hl1_dropout)
    net = eventually_load_params(net, Config.data_path(params_path))
    print("Network has {:,.0f} models".format(net.n_parameters_))

    criterion = torch.nn.NLLLoss(size_average=True)
    optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate)  # try LBGFS
    dl_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': Config.hardware.n_cpu}

    # A Scheduler which would reduce the learning rate also in case
    # of a certain threshold reached by the loss would be cool

    # The StepLR is for finding out the start lr value, ReduceLROnPlateau for the real training
    # Start with the biggest lr - found out through StepLR -, which does still decrease the loss significantly.
    # Maybe increase this biggest lr in addition with a constant factor like 2, 3 or 4
    # !!! Don't forget to change the lr_scheduler.step() function parameter further down !!!
    # lr_scheduler = StepLR(optimizer, step_size=3, gamma=1.5)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold_mode='abs',
                                     threshold=0.002)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        net = net.cuda()
        dl_args['num_workers'] = 1
        dl_args['pin_memory'] = True

    dataloaders = cv_dataloaders(posts=data['train_post'], replies=data['train_reply'], labels=labels,
                                 n_splits=cv_folds, split_ratio=cv_split, args=dl_args)

    print('starting training')
    for fold, (dl_train, dl_val) in enumerate(dataloaders, 1):
        log_dir = os.path.join(Config.path.log_folder, 'session_%s_fold_%d' % (session, fold), param_s)
        print(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(1, (n_epoch + 1)):
            train_loss, train_acc = train(dl_train)
            val_loss, val_acc = evaluate(dl_val)

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

            tensorboard_log(writer, epoch, fold, optimizer.param_groups[0]['lr'],
                            train_loss, val_loss, train_acc, val_acc)

            if epoch % save_interval == 0 and params_path is not None:
                torch.save(net.state_dict(), Config.data_path(params_path))
        writer.close()


def calc_correct_predictions(y_pred, labels):
    class_pred = torch.exp(y_pred).max(dim=1)[1]
    return (class_pred == labels).sum()


def train(dl):
    def _train(posts, replies, labels, loss_cum, acc_cum):
        y_pred = net(posts, replies)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cum += loss.data
        acc_cum += calc_correct_predictions(y_pred.data, labels.data)

    net.train()
    loss_cum = torch.FloatTensor(1).zero_()
    acc_cum = torch.LongTensor(1).zero_()
    n_batches = math.ceil(dl.dataset.__len__() / dl.batch_size)
    if torch.cuda.is_available():
        loss_cum = loss_cum.cuda()
        acc_cum = acc_cum.cuda()
        for posts, replies, labels in dl:
            _train(posts=Variable(posts, requires_grad=True).cuda(),
                   replies=Variable(replies, requires_grad=True).cuda(),
                   labels=Variable(labels, requires_grad=False).cuda(),
                   loss_cum=loss_cum,
                   acc_cum=acc_cum)
    else:
        for posts, replies, labels in dl:
            _train(posts=Variable(posts, requires_grad=True),
                   replies=Variable(replies, requires_grad=True),
                   labels=Variable(labels, requires_grad=False),
                   loss_cum=loss_cum,
                   acc_cum=acc_cum)
    loss_val = loss_cum[0] / n_batches
    acc_val = acc_cum[0] / (dl.batch_size * n_batches)
    return loss_val, acc_val


def evaluate(dl):
    def _evaluate(posts, replies, labels, loss_cum, acc_cum):
        y_pred = net(posts, replies)
        loss_cum += criterion(y_pred, labels).data
        acc_cum += calc_correct_predictions(y_pred.data, labels.data)
        return loss_cum, acc_cum

    net.eval()
    loss_cum = torch.FloatTensor(1).zero_()
    acc_cum = torch.LongTensor(1).zero_()
    n_batches = math.ceil(dl.dataset.__len__() / dl.batch_size)
    if torch.cuda.is_available():
        loss_cum = loss_cum.cuda()
        acc_cum = acc_cum.cuda()
        for posts, replies, labels in dl:
            loss_cum, acc_cum = _evaluate(
                posts=Variable(posts, volatile=True, requires_grad=False).cuda(),
                replies=Variable(replies, volatile=True, requires_grad=False).cuda(),
                labels=Variable(labels, volatile=True, requires_grad=False).cuda(),
                loss_cum=loss_cum,
                acc_cum=acc_cum)
    else:
        for posts, replies, labels in dl:
            loss_cum, acc_cum = _evaluate(
                posts=Variable(posts, volatile=True, requires_grad=False),
                replies=Variable(replies, volatile=True, requires_grad=False),
                labels=Variable(labels, volatile=True, requires_grad=False),
                loss_cum=loss_cum,
                acc_cum=acc_cum)

    loss_val = loss_cum[0] / n_batches
    acc_val = acc_cum[0] / (dl.batch_size * n_batches)
    return loss_val, acc_val


#                      _      _
#  _ __ ___   ___   __| | ___| |
# | '_ ` _ \ / _ \ / _` |/ _ \ |
# | | | | | | (_) | (_| |  __/ |
# |_| |_| |_|\___/ \__,_|\___|_|


class DataSet00(Dataset):
    def __init__(self, posts, replies, labels):
        self.replies = replies
        self.posts = posts
        self.labels = labels

    def __len__(self):
        return self.replies.shape[0]

    # ==> Problem: Just creating a tensor with torch.FloatTensor([posts, replies, labels]) does not work  !!!
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.posts[int(idx / 2),].todense()).squeeze_(),
            torch.FloatTensor(self.replies[idx,].todense()).squeeze_(),

            # That should be already a list of floats
            self.labels[idx,])


class Model00(nn.Module):
    def __init__(self, in_post, in_reply, input_post, input_reply, post_dropout, reply_dropout, hl1_dropout
                 , hidden_layer_1, output_size=2):
        super().__init__()
        # layer initialization
        init_func = torch.nn.init.kaiming_normal  # <== Probiere am Ende auch mal Xavier und so aus

        # We want to normalize our input, that's it
        self.post_norm = nn.BatchNorm1d(in_post, affine=False)
        self.reply_norm = nn.BatchNorm1d(in_reply, affine=False)
        self.in_post = nn.Linear(in_post, input_post, bias=True)
        init_func(self.in_post.weight, a=0.4678478066834172)
        #                                          PyTorch hat zwar nicht die besten Methoden
        #                                          ever dafür, aber besser als gar nichts
        self.in_reply = nn.Linear(in_reply, input_reply, bias=True)
        init_func(self.in_reply.weight)
        self.in_p_dropout = nn.AlphaDropout(post_dropout)
        self.in_r_dropout = nn.AlphaDropout(reply_dropout)

        self.in_r_activation = nn.SELU()
        self.in_p_activation = nn.SELU()
        self.hl1 = nn.Linear(input_post + input_reply, hidden_layer_1, bias=True)
        self.hl1_activation = nn.SELU()
        self.hl1_dropout = nn.AlphaDropout(hl1_dropout)

        init_func(self.hl1.weight, a=0.4678478066834172)

        self.out = nn.Linear(hidden_layer_1, output_size, bias=True)
        init_func(self.out.weight, a=0.4678478066834172)

        self.n_parameters_ = (((in_post + 1) * input_post)
                              + ((in_reply + 1) * input_reply)
                              + (input_post + input_reply + 1) * hidden_layer_1
                              + (hidden_layer_1 + 1) * output_size)

    def forward(self, in_p, in_r):
        p_in = self.post_norm(in_p)
        r_in = self.reply_norm(in_r)
        p_in = self.in_p_dropout(self.in_post(p_in))
        r_in = self.in_r_dropout(self.in_reply(r_in))
        p_in = self.in_p_activation(p_in)
        r_in = self.in_r_activation(r_in)

        hh1_in = torch.cat([p_in, r_in], dim=1)
        hh1 = self.hl1(self.hl1_dropout(hh1_in))
        hh1 = self.hl1_activation(hh1)
        out = self.out(hh1)
        return F.log_softmax(out, dim=1)


def cv_dataloaders(posts, replies, labels, args={}, n_splits=10, split_ratio=0.9, seed=42):
    p, r, l = DataHandler.shuffle_pair_matrix(posts, replies, labels, seed=seed)
    idx_pairs = DataHandler.cv_train_val_indices(r, pairs=True, split_ratio=split_ratio)
    idx_single = DataHandler.conv_cv_idx_to_single(idx_pairs)
    for _ in range(n_splits if n_splits is not None and n_splits < len(idx_pairs) else len(idx_pairs)):
        (single_train_idx, single_val_idx), (pair_train_idx, pair_val_idx) = next(zip(idx_single, idx_pairs))
        ds_train = DataSet00(posts=p[single_train_idx], replies=r[pair_train_idx], labels=l[pair_train_idx.tolist()])
        ds_val = DataSet00(posts=p[single_val_idx], replies=r[pair_val_idx], labels=l[pair_val_idx.tolist()])
        dl_train = DataLoader(ds_train, **args)
        dl_val = DataLoader(ds_val, **args)
        yield dl_train, dl_val


def load_data(train=True, test=False, data_size=None, tfidf_size=1e4):
    data = get_tfidf(train=train, test=test, data_size=data_size, max_tfidf_features=tfidf_size)
    train_labels, test_labels = get_labels(data_size=data_size)
    return data, train_labels, test_labels


def param_str(frame, log=True):
    keys, _, _, values = inspect.getargvalues(frame)
    out = ''
    for key in keys:
        s = "%s=%s" % (key, values[key])
        if log: print(s)
        out = out + '_' + s
    return out


if __name__ == '__main__':
    run(data_size, tfidf_size, input_post, input_reply, hidden_layer_1, batch_size, learning_rate, post_dropout,
        reply_dropout, hl1_dropout, save_interval, n_epoch, cv_folds, cv_split, params_path)
