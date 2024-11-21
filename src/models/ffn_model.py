import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN00(nn.Module):
    def __init__(self, post_input_size, reply_input_size, post_layer_size, reply_layer_size,
                 post_dropout, reply_dropout, hl1_dropout, hidden_layer_1, output_size=2):
        super().__init__()
        # layer initialization
        init_func = torch.nn.init.kaiming_normal

        # We want to normalize our input, that's it
        self.post_norm = nn.BatchNorm1d(post_input_size, affine=False)
        self.reply_norm = nn.BatchNorm1d(reply_input_size, affine=False)
        self.in_post = nn.Linear(post_input_size, post_layer_size, bias=True)
        init_func(self.in_post.weight, a=0.4678478066834172)

        self.in_reply = nn.Linear(reply_input_size, reply_layer_size, bias=True)
        init_func(self.in_reply.weight)
        self.in_p_dropout = nn.AlphaDropout(post_dropout)
        self.in_r_dropout = nn.AlphaDropout(reply_dropout)

        self.in_r_activation = nn.SELU()
        self.in_p_activation = nn.SELU()
        self.hl1 = nn.Linear(post_layer_size + reply_layer_size, hidden_layer_1, bias=True)
        self.hl1_activation = nn.SELU()
        self.hl1_dropout = nn.AlphaDropout(hl1_dropout)

        init_func(self.hl1.weight, a=0.4678478066834172)

        self.out = nn.Linear(hidden_layer_1, output_size, bias=True)
        init_func(self.out.weight, a=0.4678478066834172)

        self.n_parameters_ = (((post_input_size + 1) * post_layer_size)
                              + ((reply_input_size + 1) * reply_layer_size)
                              + (post_layer_size + reply_layer_size + 1) * hidden_layer_1
                              + (hidden_layer_1 + 1) * output_size)

    def forward(self, posts, replies):
        p_in = self.post_norm(posts)
        r_in = self.reply_norm(replies)
        p_in = self.in_p_dropout(self.in_post(p_in))
        r_in = self.in_r_dropout(self.in_reply(r_in))
        p_in = self.in_p_activation(p_in)
        r_in = self.in_r_activation(r_in)

        hh1_in = torch.cat([p_in, r_in], dim=1)
        hh1 = self.hl1(self.hl1_dropout(hh1_in))
        hh1 = self.hl1_activation(hh1)
        out = self.out(hh1)
        return F.log_softmax(out, dim=1)


# uses RELU instead of SELU
class FFN01(nn.Module):
    def __init__(self, post_input_size, reply_input_size, post_layer_size, reply_layer_size,
                 post_dropout, reply_dropout, hl1_dropout, hidden_layer_1, output_size=2):
        super().__init__()
        # layer initialization
        init_func = torch.nn.init.kaiming_normal

        # We want to normalize our input, that's it
        self.post_norm = nn.BatchNorm1d(post_input_size, affine=False)
        self.reply_norm = nn.BatchNorm1d(reply_input_size, affine=False)
        self.in_post = nn.Linear(post_input_size, post_layer_size, bias=True)
        init_func(self.in_post.weight, a=0.4678478066834172)

        self.in_reply = nn.Linear(reply_input_size, reply_layer_size, bias=True)
        init_func(self.in_reply.weight)
        self.in_p_dropout = nn.AlphaDropout(post_dropout)
        self.in_r_dropout = nn.AlphaDropout(reply_dropout)

        self.in_r_activation = nn.ReLU()
        self.in_p_activation = nn.ReLU()
        self.hl1 = nn.Linear(post_layer_size + reply_layer_size, hidden_layer_1, bias=True)
        self.hl1_activation = nn.ReLU()
        self.hl1_dropout = nn.AlphaDropout(hl1_dropout)

        init_func(self.hl1.weight, a=0.4678478066834172)

        self.out = nn.Linear(hidden_layer_1, output_size, bias=True)
        init_func(self.out.weight, a=0.4678478066834172)

        self.n_parameters_ = (((post_input_size + 1) * post_layer_size)
                              + ((reply_input_size + 1) * reply_layer_size)
                              + (post_layer_size + reply_layer_size + 1) * hidden_layer_1
                              + (hidden_layer_1 + 1) * output_size)

    def forward(self, posts, replies):
        p_in = self.post_norm(posts)
        r_in = self.reply_norm(replies)
        p_in = self.in_p_dropout(self.in_post(p_in))
        r_in = self.in_r_dropout(self.in_reply(r_in))
        p_in = self.in_p_activation(p_in)
        r_in = self.in_r_activation(r_in)

        hh1_in = torch.cat([p_in, r_in], dim=1)
        hh1 = self.hl1(self.hl1_dropout(hh1_in))
        hh1 = self.hl1_activation(hh1)
        out = self.out(hh1)
        return F.log_softmax(out, dim=1)


# xavier normalization + cosine similarity
class FFN02(nn.Module):
    def __init__(self, post_input_size, reply_input_size, post_layer_size, reply_layer_size,
                 post_dropout, reply_dropout, output_size=2):
        super().__init__()
        # layer initialization
        init_func = torch.nn.init.xavier_normal  # <== Probiere am Ende auch mal Xavier und so aus

        # We want to normalize our input, that's it
        self.post_norm = nn.BatchNorm1d(post_input_size, affine=False)
        self.reply_norm = nn.BatchNorm1d(reply_input_size, affine=False)
        self.in_post = nn.Linear(post_input_size, post_layer_size, bias=True)
        init_func(self.in_post.weight, gain=1)

        if reply_layer_size <= post_layer_size:
            raise ValueError("Condition not fulfilled: reply_layer_size <= post_layer_size")
        padding_size = reply_layer_size - post_layer_size
        self.post_padding = nn.ZeroPad2d((0, padding_size))

        self.in_reply = nn.Linear(reply_input_size, reply_layer_size, bias=True)
        init_func(self.in_reply.weight, gain=1)
        self.in_p_dropout = nn.AlphaDropout(post_dropout)
        self.in_r_dropout = nn.AlphaDropout(reply_dropout)

        self.in_r_activation = nn.SELU()
        self.in_p_activation = nn.SELU()

        self.out = nn.Linear(post_layer_size + padding_size + reply_layer_size + 1, output_size, bias=True)
        init_func(self.out.weight, gain=1)

        self.n_parameters_ = (((post_input_size + 1) * post_layer_size)
                              + ((reply_input_size + 1) * reply_layer_size)
                              + (post_layer_size + reply_layer_size + 1) * output_size)

    def forward(self, posts, replies):
        p_in = self.post_norm(posts)
        r_in = self.reply_norm(replies)
        p_in = self.in_p_dropout(self.in_post(p_in))
        r_in = self.in_r_dropout(self.in_reply(r_in))
        p_in = self.in_p_activation(p_in)
        r_in = self.in_r_activation(r_in)

        p_in = self.post_padding(p_in)

        cos_sim = F.cosine_similarity(p_in, r_in).unsqueeze(1)  # TEST THIS
        hh1 = torch.cat([p_in, r_in, cos_sim], dim=1)
        out = self.out(hh1)
        return F.log_softmax(out, dim=1)
