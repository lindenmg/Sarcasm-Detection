#      _                               _           _
#   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
#  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
# | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
#  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
#            |_|

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('<root_directory_path>')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pathlib import Path
from tqdm import tqdm

from src.data_science.samplers import BucketRandomSampler, LazyBatchSampler
from src.data_science.networkhelper import NetworkHelper as NH
from src.preprocessing.datahandler import DataHandler as DH
from src.data_science.datasets import EmbeddingDataSet
from src.tools.config import Config
import src.tools.helpers as helpers

# ### Paths Of Data Files

# In[3]:
helpers.section_text("Start DNN training preparation")

data_dir = Path(Config.path.data_folder)
f_train_posts = str(data_dir / 'post_emb_train_lower.pkl')
f_train_reply = str(data_dir / 'reply_emb_train_lower.pkl')
f_train_label = str(data_dir / 'train_labels.pkl')

glove_dir = 'glove'
fast_text_dir = 'fastText'
f_glove = 'glove.twitter.27B.200d.txt'
f_fast_text = 'test.vec'
f_vectors = str(data_dir / 'word_vectors' / glove_dir / f_glove)

f_counter = data_dir / 'counter_lower.pkl'
if not f_counter.is_file():
    raise IOError(
        "File for counter object for word-vector loading not "
        "found! {} does not exist!".format(str(f_counter)))
else:
    f_counter = str(f_counter)

# ### Config Values For DNN Training

# In[4]:

# Word vectors and data %%%%%%%%%%%%%%%%%%%%%%%%
is_ft_format = False  # If the word-vector file is in the fastText format (header row)
vector_num = int(1193514)
vector_dim = 200

# data_size = 512  # Use None value for all the data
data_dim = 20  # How many tokens at maximum per reply/post

# General %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
torch.manual_seed(1337)
train_val_split = 1 / 9

num_threads = Config.hardware.n_cpu
batch_multi = 3  # For the BucketRandomSampler
cuda = False

# Training sequence %%%%%%%%%%%%%%%%%%%%%%%%%%%%
batch_size = 512
drop_last = False
epochs = 30

# Training mechanics %%%%%%%%%%%%%%%%%%%%%%%%%%%
lr_scheduler = ReduceLROnPlateau
criterion = nn.NLLLoss(size_average=True)
optimizer = torch.optim.Adam
learn_rate = 1e-3  # 5e-04 for Adam
train_emb = False

# Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
weight_init_func = torch.nn.init.kaiming_normal
reply_kernel_size = (2, vector_dim)
post_kernel_size = (2, vector_dim)
reply_kernels = 20
post_kernels = 20
conv_drop_out = 0.5
fc_drop_out = 0.2

# +3 because of the cosine-similarity and l1 + l2 distance calculations
fco_out = (post_kernels + reply_kernels + 3)
fc1_hid = post_kernels + reply_kernels + 3
# In[5]:


data_length = int(round(196527 * (1 - train_val_split)))
data_length = data_length - data_length % 2
batch_size, _ = helpers.get_best_batch_size(data_length, batch_size, residual=0.9)
batch_size = int(batch_size)


# ### Helper Functions

# In[6]:


def load_data():
    train_posts = helpers.load_from_disk(f_train_posts)
    train_reply = helpers.load_from_disk(f_train_reply)
    train_label = helpers.load_from_disk(f_train_label)
    return train_posts, train_reply, train_label


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
    word_list, vectors = DH.load_word_vectors(f_vectors, vector_num, vector_dim, is_ft_format)
    word_idx = helpers.idx_lookup_from_list(word_list)
    vector_t = DH.conv_inner_to_tensor(vectors)
    counter = helpers.load_from_disk(f_counter)
    vocab = NH.create_tt_vocab_obj(counter, word_idx, vector_t
                                   , max_size=None, min_freq=3)
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
    for post, reply, label in tqdm(dataloader):
        if train_emb:
            y_pred = net(Variable(post, requires_grad=False).cuda()
                         , Variable(reply, requires_grad=False).cuda())
        else:
            y_pred = net(post.cuda(), reply.cuda())
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
    for post, reply, label in dataloader:
        y_pred = net(Variable(post, volatile=True).cuda()
                     , Variable(reply, volatile=True).cuda())
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
    msg = "{:3d}: Train-loss: {:.2e}, Train-acc: {:3.2f} | " \
          "Val-loss: {:.2e}, Val-acc: {:3.2f} | Lr: {:.2e}"
    epoch += 1
    print()
    print(msg.format(epoch, train_loss, train_acc, val_loss, val_acc, lr))


def train_iter(net, val_dataloader, train_dataloader, num_train_batches
               , num_val_batches, optimizer, lr_scheduler, train_emb):
    net.cuda()
    criterion.cuda()

    if train_emb:
        optimizer = optimizer(net.parameters(), lr=learn_rate)
    else:
        params_to_optimize = [p for p in net.parameters() if p.requires_grad]
        optimizer = optimizer(params_to_optimize, lr=learn_rate)
    lr_scheduler = lr_scheduler(optimizer, mode='min', factor=0.5, patience=4)
    # lr_scheduler = StepLR(optimizer, step_size=3, gamma=1.5)

    for epoch in range(epochs):
        loss_train, acc_train = train(train_dataloader, net, criterion
                                      , optimizer, num_train_batches)
        loss_val, acc_val = validate(val_dataloader, net
                                     , criterion, num_val_batches)
        lr_scheduler.step(loss_val)  # loss_val as parameter in case of ReduceLROnPlateau scheduler
        lr = optimizer.param_groups[0]['lr']
        log_epoch(epoch, loss_val, loss_train, acc_val, acc_train, lr)


# ### Setup Before Training

# In[10]:


# %%time
train_posts, train_reply, train_label = load_data()

# In[11]:


p_val, p_train, r_val, r_train, l_val, l_train = split_train_val(train_posts, train_reply, train_label)

# In[12]:


p_train_len, r_train_len = get_length_tensors(p_train, r_train)
p_val_len, r_val_len = get_length_tensors(p_val, r_val)

# In[13]:

p_train = pad_tensor(p_train)
r_train = pad_tensor(r_train)
p_val = pad_tensor(p_val)
r_val = pad_tensor(r_val)
train_dataset = EmbeddingDataSet(p_train, r_train, l_train)
val_dataset = EmbeddingDataSet(p_val, r_val, l_val)

# In[14]:


# %%time
train_sampler = BucketRandomSampler(train_dataset, r_train_len, batch_size, batch_multi, cuda)
train_batch_sampler = LazyBatchSampler(train_sampler, batch_size, drop_last)
num_train_batches = len(train_batch_sampler)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler
                              , num_workers=num_threads, pin_memory=True)

# In[15]:


# %%time
val_sampler = BucketRandomSampler(val_dataset, r_val_len, batch_size, batch_multi, cuda)
val_batch_sampler = LazyBatchSampler(val_sampler, batch_size, drop_last)
num_val_batches = len(val_batch_sampler)
val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler
                            , num_workers=num_threads, pin_memory=True)

# In[16]:


# %%time
vocab = load_word_vectors()
vectors = vocab.vectors
vocab = None
unkn = -1  # Because of the padding vector which is zero anyway
for vector in vectors:
    if vector.sum() == 0:
        unkn += 1
print("{:d} unknown words that have vectors initialized with zero".format(unkn))


# ### Model Definition
class CnnSimple(nn.Module):

    def __init__(self, post_kernel, reply_kernel, p_kernel_num, r_kernel_num
                 , emb_num, emb_dim, pt_vectors, train_embed=False):
        super().__init__()
        init_func_conv = torch.nn.init.dirac
        init_func_fc = torch.nn.init.xavier_normal
        self.embed = nn.Embedding(emb_num, emb_dim)
        self.embed.weight = nn.Parameter(pt_vectors, train_embed)
        self.embed_norm = nn.BatchNorm2d(1, affine=False)

        self.conv_p = nn.Conv2d(1, p_kernel_num, post_kernel, bias=True)
        self.conv_r = nn.Conv2d(1, r_kernel_num, reply_kernel, bias=True)
        self.conv_a_do = nn.AlphaDropout(conv_drop_out)
        init_func_conv(self.conv_p.weight)  # 0.7150094360363474
        init_func_conv(self.conv_r.weight)  # 1.0954825378155515

        self.max_pool_p = nn.AdaptiveMaxPool1d(1)
        self.max_pool_r = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(fc1_hid, fco_out, bias=True)
        self.fc2 = nn.Linear(fco_out, fco_out, bias=True)
        self.fco = nn.Linear(fco_out, 2, bias=True)
        self.fc_do = nn.AlphaDropout(fc_drop_out)
        init_func_fc(self.fco.weight, gain=1.0)  # 0.8797652911446248 0.2574230339124581
        init_func_fc(self.fc1.weight, gain=1.0)  # 0.34818185395576184
        init_func_fc(self.fc2.weight, gain=1.0)  # 0.4678478066834172 1.280956089

        self.activation = nn.SELU()
        self.cos_similarity = F.cosine_similarity
        self.lx_similarity = F.pairwise_distance
        self.cat = torch.cat

    def forward(self, in_p, in_r):
        in_p = self.embed(in_p)
        in_r = self.embed(in_r)
        in_p = self.embed_norm(in_p.unsqueeze(1))
        in_r = self.embed_norm(in_r.unsqueeze(1))

        in_p = self.conv_a_do(self.conv_p(in_p)).squeeze()
        in_r = self.conv_a_do(self.conv_r(in_r)).squeeze()
        in_p = self.activation(in_p)
        in_r = self.activation(in_r)
        in_p = self.max_pool_p(in_p).squeeze()
        in_r = self.max_pool_r(in_r).squeeze()
        cos_sim = self.cos_similarity(in_p, in_r).unsqueeze(1)
        l1_sim = self.lx_similarity(in_p, in_r, p=1)
        l2_sim = self.lx_similarity(in_p, in_r, p=2)

        data = self.cat([in_p, in_r, cos_sim, l1_sim, l2_sim], dim=1)
        data = self.activation(self.fc_do(self.fc1(data)))
        data = self.activation(self.fc_do(self.fc2(data)))
        data = self.fco(data)
        return F.log_softmax(data, dim=1)

    def parameter_count(self):
        params = list(self.parameters())
        return sum([np.prod(list(d.size())) for d in params])


# ### Training
net = CnnSimple(post_kernel_size, reply_kernel_size, post_kernels, reply_kernels
                , len(vectors), vector_dim, vectors, train_emb)

print("Cuda is available ==", torch.cuda.is_available())
print("Network has {:,.0f} parameters".format(net.parameter_count()))
helpers.section_text("Start DNN training sequence")
train_iter(net, val_dataloader, train_dataloader, num_train_batches
           , num_val_batches, optimizer, lr_scheduler, train_emb)
helpers.section_text("End of DNN training sequence")
