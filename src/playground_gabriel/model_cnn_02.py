#      _                               _           _
#   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
#  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
# | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
#  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
#            |_|

from pathlib import Path

import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.tools.helpers as helpers
from src.data_science.datasets import EmbeddingDataSet
from src.data_science.networkhelper import NetworkHelper as NH
from src.data_science.samplers import BucketRandomSampler, LazyBatchSampler
from src.models.cnn_baseline import CnnSimple
from src.preprocessing.datahandler import DataHandler as DH
from src.tools.config import Config

# ### Paths Of Data Files

# In[3]:
helpers.section_text("Start DNN training preparation")

test_file = "survey_test"
data_dir = Path(Config.path.data_folder)
f_train_posts = str(data_dir / 'post_emb_train_lower.pkl')
f_train_reply = str(data_dir / 'reply_emb_train_lower.pkl')
f_train_label = str(data_dir / 'train_labels.pkl')
f_test_posts = str(data_dir / ('post_emb_' + test_file + '_lower.pkl'))
f_test_reply = str(data_dir / ('reply_emb_' + test_file + '_lower.pkl'))
f_test_label = str(data_dir / (test_file + '_labels.pkl'))

glove_dir = 'glove'
fast_text_dir = 'fastText'
lexvec_dir = 'LexVec'
word2vec_dir = 'word2vec'
f_word2vec = 'GoogleNews-vectors-negative300.bin'
f_glove = 'glove.twitter.27B.200d.txt'
f_fast_text = 'ft_2M_300.csv'
f_lexvec = 'lexvec.commoncrawl.300d.W+C.pos.neg3.vectors'
f_vectors = str(data_dir / 'word_vectors' / fast_text_dir / f_fast_text)

f_counter = data_dir / 'counter_lower.pkl'
if not f_counter.is_file():
    raise IOError(
        "File for counter object for word-vector loading not "
        "found! {} does not exist!".format(str(f_counter)))
else:
    f_counter = str(f_counter)

# ### Config Values For DNN Training

# Word vectors and data %%%%%%%%%%%%%%%%%%%%%%%%
is_ft_format = False  # deprecated
vector_num = int(2e6 - 1)  # int(1193514)
vector_dim = 300
word2vec = "word2vec" in f_vectors

# data_size = 512  # Use None value for all the data
data_dim = 30  # How many tokens at maximum per reply/post

# General %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
torch.manual_seed(87076143)
train_val_split = 1 / 9

num_threads = Config.hardware.n_cpu
batch_multi = 3  # For the BucketRandomSampler
cuda = False

# Training sequence %%%%%%%%%%%%%%%%%%%%%%%%%%%%
batch_size = 256
drop_last = False
epochs = 4

# Training mechanics %%%%%%%%%%%%%%%%%%%%%%%%%%%
lr_scheduler = ReduceLROnPlateau
criterion = nn.NLLLoss(size_average=True)
optimizer = torch.optim.Adam
learn_rate = 3e-4  # 5e-04 for Adam
train_emb = False

# Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
init_func_conv = torch.nn.init.dirac
init_func_fc = torch.nn.init.xavier_normal
hl1_kernels = [(2, vector_dim)]  # , (3, vector_dim), (4, vector_dim)]
hl1_kernel_num = 12
fc_drop_out = 0.5

data_length = int(round(196526 * (1 - train_val_split)))
data_length = data_length - data_length % 2
batch_size, _ = helpers.get_best_batch_size(data_length, batch_size, residual=0.9)
batch_size = int(batch_size)


# ### Helper Functions

def load_data():
    train_posts = helpers.load_from_disk(f_train_posts)
    train_reply = helpers.load_from_disk(f_train_reply)
    train_label = helpers.load_from_disk(f_train_label)
    test_posts = helpers.load_from_disk(f_test_posts)
    test_reply = helpers.load_from_disk(f_test_reply)
    test_label = helpers.load_from_disk(f_test_label)
    return train_posts, train_reply, train_label, test_posts, test_reply, test_label


def get_length_tensors(posts, replies):
    reply_length = helpers.create_length_tensor(replies)
    post_length = helpers.create_length_tensor(posts)
    return post_length, reply_length


def split_train_val(posts, replies, labels):
    p_val, p_train = DH.split_np_data(posts, percentage=train_val_split, pairs=False)
    r_val, r_train = DH.split_np_data(replies, percentage=train_val_split)
    l_val, l_train = DH.split_np_data(labels, percentage=train_val_split)
    return p_val, p_train, r_val, r_train, l_val, l_train


def load_word_vectors():
    if not word2vec:
        word_list, vectors = DH.load_word_vectors(f_vectors, vector_num, vector_dim, is_ft_format)
        word_idx = helpers.idx_lookup_from_list(word_list)
        vector_t = DH.conv_inner_to_tensor(vectors)
    else:
        model = KeyedVectors.load_word2vec_format(f_vectors, binary=True)
        word_idx = helpers.idx_lookup_from_list(model.index2word)
        vector_t = DH.conv_inner_to_tensor(model.vectors)
    counter = helpers.load_from_disk(f_counter)
    vocab = NH.create_tt_vocab_obj(counter, word_idx, vector_t
                                   , max_size=None, min_freq=1)
    return vocab


def pad_tensor(emb_idx):
    tensor = torch.LongTensor(len(emb_idx), data_dim).zero_()
    for i, idx_t in enumerate(emb_idx):
        end = len(idx_t)
        if end > 0:
            tensor[i][0:end] = idx_t[0:data_dim]
    return tensor


def train(dataloader, net, criterion, optimizer, num_batches):
    net.train()
    loss_epoch = torch.FloatTensor(1).zero_().cuda()
    acc_epoch = torch.LongTensor(1).zero_().cuda()
    for data, label in tqdm(dataloader):
        if train_emb:
            y_pred = net(Variable(data['posts'], requires_grad=False).cuda()
                         , Variable(data['replies'], requires_grad=False).cuda())
        else:
            y_pred = net(data['posts'].cuda(), data['replies'].cuda())
            y_pred.retain_grad = True

        optimizer.zero_grad()
        label = Variable(label, volatile=True).cuda()
        loss = criterion(y_pred, label)
        acc_epoch += calc_correct_predictions(y_pred.data, label.data)
        loss_epoch += loss.data
        loss.backward()
        optimizer.step()
        last_batch_size = len(data['replies'])
    loss_epoch = loss_epoch[0] / num_batches
    acc_epoch = acc_epoch[0] / ((num_batches - 1) * batch_size + last_batch_size)
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
        last_batch_size = len(data['replies'])
    batch_size = dataloader.batch_sampler.batch_size
    loss_val = loss_val[0] / num_batches
    acc_val = acc_val[0] / ((num_batches - 1) * batch_size + last_batch_size)
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
               , num_val_batches, optimizer, lr_scheduler, train_emb):
    net.cuda()
    criterion.cuda()

    if train_emb:
        optimizer = optimizer(net.parameters(), lr=learn_rate)
    else:
        params_to_optimize = [p for p in net.parameters() if p.requires_grad]
        optimizer = optimizer(params_to_optimize, lr=learn_rate)
    lr_scheduler = lr_scheduler(optimizer, mode='min', factor=0.2, patience=5)

    for epoch in range(epochs):
        loss_train, acc_train = train(train_dataloader, net, criterion
                                      , optimizer, num_train_batches)
        loss_val, acc_val = validate(val_dataloader, net, criterion, num_val_batches)
        lr_scheduler.step(loss_val)  # loss_val as parameter in case of ReduceLROnPlateau scheduler
        lr = optimizer.param_groups[0]['lr']
        log_epoch(epoch, loss_val, loss_train, acc_val, acc_train, lr)
    loss_test, acc_test = validate(test_dataloader, net, criterion, num_test_batches)
    print("Test-loss: {:.2e}, Test-acc: {:3.2f}".format(loss_test, acc_test * 100))


data_t = load_data()
train_posts = data_t[0]
train_reply = data_t[1]
train_label = data_t[2]
test_posts = data_t[3]
test_reply = data_t[4]
test_label = data_t[5]

p_val, p_train, r_val, r_train, l_val, l_train = split_train_val(train_posts, train_reply, train_label)

p_train_len, r_train_len = get_length_tensors(p_train, r_train)
p_val_len, r_val_len = get_length_tensors(p_val, r_val)
p_test_len, r_test_len = get_length_tensors(test_posts, test_reply)

p_train = pad_tensor(p_train)
r_train = pad_tensor(r_train)
p_val = pad_tensor(p_val)
r_val = pad_tensor(r_val)
p_test = pad_tensor(test_posts)
r_test = pad_tensor(test_reply)
train_dataset = EmbeddingDataSet(p_train, r_train, l_train)
val_dataset = EmbeddingDataSet(p_val, r_val, l_val)
test_dataset = EmbeddingDataSet(p_test, r_test, test_label)

train_sampler = BucketRandomSampler(train_dataset, r_train_len, batch_size, batch_multi, cuda)
train_batch_sampler = LazyBatchSampler(train_sampler, batch_size, drop_last)
num_train_batches = len(train_batch_sampler)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler
                              , num_workers=num_threads, pin_memory=True)

val_sampler = BucketRandomSampler(val_dataset, r_val_len, batch_size, batch_multi, cuda)
val_batch_sampler = LazyBatchSampler(val_sampler, batch_size, drop_last)
num_val_batches = len(val_batch_sampler)
val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler
                            , num_workers=num_threads, pin_memory=True)

test_batch_size = 1024 if len(r_test) >= 1024 else len(r_test)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=False
                             , num_workers=num_threads, pin_memory=True)
num_test_batches = len(test_dataloader.batch_sampler)

vocab = load_word_vectors()
vectors = vocab.vectors
vocab = None
unkn = -1  # Because of the padding vector which is zero anyway
for vector in vectors:
    if vector.sum() == 0:
        unkn += 1
print("{:d} unknown words that have vectors initialized with zero".format(unkn))

# ### Training
net = CnnSimple(hl1_kernels, hl1_kernel_num, len(vectors)
                , vector_dim, vectors)

print("Cuda is available ==", torch.cuda.is_available())
print("Network has {:,.0f} parameters".format(net.parameter_count()))
helpers.section_text("Start DNN training sequence")
train_iter(net, val_dataloader, train_dataloader, num_train_batches
           , num_val_batches, optimizer, lr_scheduler, train_emb)
helpers.section_text("End of DNN training sequence")
