import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.tools.helpers as helpers
from src.data_science.datasets import EmbeddingDataSet
from src.data_science.samplers import BucketRandomSampler, LazyBatchSampler
from src.models.cnn_baseline import CnnSimple
from src.preprocessing.datahandler import DataHandler as DH
from src.tools.config import Config

torch.manual_seed(1337)
np.random.seed(1337)
train_val_split = 1 / 9
lr_scheduler = ReduceLROnPlateau
num_threads = Config.hardware.n_cpu
batch_multi = 3  # For the BucketRandomSampler
learn_rate = 3e-4
batch_size = 256
criterion = nn.NLLLoss(size_average=True)
optimizer = torch.optim.Adam
drop_last = False
epochs = 12
cuda = False

init_func_conv = torch.nn.init.dirac
init_func_fc = torch.nn.init.xavier_normal
fc_drop_out = 0.5

data_length = int(round(196526 * (1 - train_val_split)))
data_length = data_length - data_length % 2
batch_size, _ = helpers.get_best_batch_size(data_length, batch_size, residual=0.9)
batch_size = int(batch_size)


def split_train_val(posts, replies, labels):
    p_val, p_train = DH.split_np_data(posts, percentage=train_val_split, pairs=False)
    r_val, r_train = DH.split_np_data(replies, percentage=train_val_split)
    l_val, l_train = DH.split_np_data(labels, percentage=train_val_split)
    return p_val, p_train, r_val, r_train, l_val, l_train


def train(dataloader, net, criterion, optimizer, num_batches):
    net.train()
    loss_epoch = torch.FloatTensor(1).zero_().cuda()
    acc_epoch = torch.LongTensor(1).zero_().cuda()
    for data, label in tqdm(dataloader):
        y_pred = net(data['posts'].cuda(), data['replies'].cuda())
        y_pred.retain_grad = True

        optimizer.zero_grad()
        label = Variable(label, volatile=True).cuda()
        loss = criterion(y_pred, label)
        acc_epoch += calc_correct_predictions(y_pred.data, label.data)
        loss_epoch += loss.data
        loss.backward()
        optimizer.step()
    loss_epoch = loss_epoch[0] / num_batches
    acc_epoch = acc_epoch[0] / (num_batches * batch_size)
    return loss_epoch, acc_epoch


def validate(dataloader, net, criterion, num_batches):
    net.eval()
    loss_val = torch.FloatTensor(1).zero_().cuda()
    acc_val = torch.LongTensor(1).zero_().cuda()
    for data, label in dataloader:
        y_pred = net(Variable(data['posts'], volatile=True).cuda()
                     , Variable(data['replies'], volatile=True).cuda())
        label = Variable(label, volatile=True).cuda()
        loss = criterion(y_pred, label)
        acc_val += calc_correct_predictions(y_pred.data, label.data)
        loss_val += loss.data
    loss_val = loss_val[0] / num_batches
    acc_val = acc_val[0] / (num_batches * batch_size)
    return loss_val, acc_val


def calc_correct_predictions(y_pred, labels):
    class_pred = torch.exp(y_pred).max(dim=1)[1]
    return (class_pred == labels).sum()


def log_epoch(epoch, val_loss, train_loss, val_acc, train_acc, lr):
    msg = "{:3d}: Train-loss: {:.2e}, Train-acc: {:3.1f} | " \
          "Val-loss: {:.2e}, Val-acc: {:3.1f} | Lr: {:.2e}"
    epoch += 1
    print()
    print(msg.format(epoch, train_loss, train_acc * 100, val_loss, val_acc * 100, lr))


def train_iter(net, val_dataloader, train_dataloader, num_train_batches
               , num_val_batches, optimizer, lr_scheduler):
    net.cuda()
    criterion.cuda()
    params_to_optimize = [p for p in net.parameters() if p.requires_grad]
    optimizer = optimizer(params_to_optimize, lr=learn_rate)
    lr_scheduler = lr_scheduler(optimizer, mode='min', factor=0.2, patience=5)

    for epoch in range(epochs):
        loss_train, acc_train = train(train_dataloader, net, criterion
                                      , optimizer, num_train_batches)
        loss_val, acc_val = validate(val_dataloader, net
                                     , criterion, num_val_batches)
        lr_scheduler.step(loss_val)  # loss_val as parameter in case of ReduceLROnPlateau scheduler
        lr = optimizer.param_groups[0]['lr']
        log_epoch(epoch, loss_val, loss_train, acc_val, acc_train, lr)


def _load_data_factory(args):
    dict_ = {'train': True,
             'test': False,
             'cache_dir': Config.path.cache_folder,
             'session_tag': "cnn_baseline",
             'cache_prefix': "cnn_simple"}
    args['args'].update(dict_)
    class_ = getattr(importlib.import_module(args['module']), args['class_name'])
    return class_(**args['args'])


train_pipe_config_path = Path(Config.path.project_root_folder) / 'src' / 'playground_gabriel'
train_pipe_config_path = train_pipe_config_path / 'baseline_config.json'

with open(str(train_pipe_config_path)) as file:
    params = json.load(file)

data_factory = _load_data_factory(params['data_factory'])
result_dict = data_factory.get_data()
train_datadict = result_dict['train_data']
test_datadict = result_dict['test_data']
word_vectors = result_dict.get('word_vectors', None)
embedding_size = result_dict.get('embedding_size', None)
reply_lengths = result_dict.get('reply_lengths', None)

p_val, p_train, r_val, r_train, l_val, l_train = split_train_val(train_datadict['posts']
                                                                 , train_datadict['replies']
                                                                 , train_datadict['labels'])

train_dataset = EmbeddingDataSet(p_train, r_train, l_train)
val_dataset = EmbeddingDataSet(p_val, r_val, l_val)

# train_len = helpers.create_length_tensor(r_train)
# val_len = helpers.create_length_tensor(r_val)

train_sampler = BucketRandomSampler(train_dataset, reply_lengths[-len(train_dataset):]
                                    , batch_size, batch_multi, cuda)
train_batch_sampler = LazyBatchSampler(train_sampler, batch_size, drop_last)
num_train_batches = len(train_batch_sampler)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler
                              , num_workers=num_threads, pin_memory=True)

val_sampler = BucketRandomSampler(val_dataset, reply_lengths[:len(val_dataset)]
                                  , batch_size, batch_multi, cuda)
val_batch_sampler = LazyBatchSampler(val_sampler, batch_size, drop_last)
num_val_batches = len(val_batch_sampler)
val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler
                            , num_workers=num_threads, pin_memory=True)

net = CnnSimple([(2, 300)], 12, embedding_size, 300, word_vectors)

print("Cuda is available ==", torch.cuda.is_available())
print("Network has {:,.0f} trainable parameters".format(net.parameter_count()))
helpers.section_text("Start DNN training sequence")
train_iter(net, val_dataloader, train_dataloader, num_train_batches
           , num_val_batches, optimizer, lr_scheduler)
helpers.section_text("End of DNN training sequence")
