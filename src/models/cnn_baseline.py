import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnSimple(nn.Module):

    def __init__(self, hl1_kernels, hl1_kernel_num, emb_num, emb_dim, pt_vectors,
                 conv_activ=nn.PReLU, conv_activ_param_dict={}, init_func_conv=nn.init.dirac
                 , init_func_fc=nn.init.xavier_normal, init_conv_param_dict={}, init_fc_param_dict={}):
        """
        This is the baseline CNN model for the SARC sarcasm detection task.

        It takes input that represents text in the form of a post and a reply.
        The actual input should be the indices of the word vectors in the
        embedding layer.
        It runs one layer of convolution separately over the word vectors from
        post and reply. The result gets concatenated and put into a final fully
        connected layer on which result log(softmax) is applied.

        Parameters
        ----------
        hl1_kernels: list of tuples
            Definition of the kernels. The size of the kernels is defined as
            a tuple. One tuple per kernel. Which means, if you want to run
            several different kernels over the input you have just to define
            them in the tuple fashion.
        hl1_kernel_num: int
            How many different variates of one kernel you want to have.
            Can also be viewed as the number of features or channel you create
            for a certain kernel size. Parameter is applied to all kernel sizes.
        emb_num: int
            The number of words in the embedding layer
        emb_dim: int
            The size which the word vectors in the embedding layer will have
        pt_vectors: torch.FloatTensor
            The pre-trained content of the embedding layer. Be aware, that the
            embedding layer does not get trained. Use therefore given word vectors.
        conv_activ: torch.nn.Module
            The activation function for the CNN input layer
        conv_activ_param_dict: dict
            The parameter names (key) and their values (value) for the activation
            function of the CNN input layer
        init_func_conv: torch.init
            The initialization function for the weights of the convolution layers
        init_func_fc: torch.init
            The initialization function for the weights of the fully connected layers
        init_conv_param_dict: dict
            The parameter names (key) and their values (value) for the initialization
            function of the convolution layers. Pass {} for standard / no parameters
        init_fc_param_dict: dict
            The parameter names (key) and their values (value) for the initialization
            function of the fully connected layers. Pass {} for standard / no parameters
        """
        super().__init__()

        # I EMBEDDING
        # BTW: we don't train the Embedding Layer - not ever. Overfitting guaranteed
        self.embed = nn.Embedding(emb_num, emb_dim)
        self.embed.weight = nn.Parameter(pt_vectors, requires_grad=False)
        self.embed_norm_posts = nn.BatchNorm2d(1, affine=False)
        self.embed_norm_reply = nn.BatchNorm2d(1, affine=False)

        # II CONVOLUTION
        # ... weight-init, dropout, activation & BatchNorm of the first layer
        self.hl1_posts_convs = nn.ModuleList([nn.Conv2d(1, hl1_kernel_num, kernel)
                                              for kernel in hl1_kernels])
        self.hl1_reply_convs = nn.ModuleList([nn.Conv2d(1, hl1_kernel_num, kernel)
                                              for kernel in hl1_kernels])
        self.hl1_posts_norm = nn.ModuleList([nn.BatchNorm2d(hl1_kernel_num, affine=False)
                                             for _ in range(len(hl1_kernels))])
        self.hl1_reply_norm = nn.ModuleList([nn.BatchNorm2d(hl1_kernel_num, affine=False)
                                             for _ in range(len(hl1_kernels))])
        [init_func_conv(conv.weight, **init_conv_param_dict) for conv in self.hl1_posts_convs]
        [init_func_conv(conv.weight, **init_conv_param_dict) for conv in self.hl1_reply_convs]
        self.activation_conv_p = nn.ModuleList([conv_activ(hl1_kernel_num, **conv_activ_param_dict)
                                                for _ in range(len(hl1_kernels))])
        self.activation_conv_r = nn.ModuleList([conv_activ(hl1_kernel_num, **conv_activ_param_dict)
                                                for _ in range(len(hl1_kernels))])

        # III SIMILARITY
        # Helper functions to calculate feature vector
        # similarities and concatenate for fully connected
        self.cos_similarity = F.cosine_similarity
        self.cat = torch.cat

        # IV FULLY CONNECTED
        # The input size of the fully connected layer
        # after the concatenation of the max pooling results
        fco_out = int(2 * hl1_kernel_num * len(hl1_kernels) + len(hl1_kernels))
        self.fco = nn.Linear(fco_out, 2, bias=True)
        init_func_fc(self.fco.weight, **init_fc_param_dict)

    def forward(self, posts, replies):
        """
        Forward pass of the baseline CNN model for the SARC sarcasm detection task.

        Parameters
        ----------
        posts: torch.FloatTensor
            Tensor which represents a post in the form of the indices for the words
            in the embedding layer. Meaning: The sentence in its original form consists
            of words. You have created a look-up table for word vectors at some point
            (pre-trained embedding layer weights). Now in the posts parameter the words
            got replaced by their corresponding indices in the embedding layer.
        replies: torch.FloatTensor
            Reply of the post-reply pair. For deeper explanation see posts parameter

        Returns
        -------
        torch.FloatTensor
            Result of the log-softmax. First index represents confidence that
            the reply is NOT sarcastic, second index that it is sarcasm
        """
        # I EMBEDDING
        # - get the word vectors that represent the reply & post sentences
        posts = self.embed(posts)
        replies = self.embed(replies)
        posts = self.embed_norm_posts(posts.unsqueeze(1))
        replies = self.embed_norm_reply(replies.unsqueeze(1))

        # II CONVOLUTION layer with activation, batch-norm and max pooling
        posts = [activ(conv(posts)) for conv, activ
                 in zip(self.hl1_posts_convs, self.activation_conv_p)]
        replies = [activ(conv(replies)) for conv, activ
                   in zip(self.hl1_reply_convs, self.activation_conv_r)]
        posts = [norm(posts_features).squeeze()
                 for norm, posts_features in zip(self.hl1_posts_norm, posts)]
        replies = [norm(reply_features).squeeze()
                   for norm, reply_features in zip(self.hl1_reply_norm, replies)]

        posts = [F.adaptive_max_pool1d(posts_feature, 1).squeeze() for posts_feature in posts]
        replies = [F.adaptive_max_pool1d(reply_feature, 1).squeeze() for reply_feature in replies]

        # III SIMILARITY of the post and reply features
        cos_sims = [self.cos_similarity(p, r).unsqueeze(1) for p, r in zip(posts, replies)]
        cos_sims = self.cat(cos_sims, dim=1)
        posts = self.cat(posts, dim=1)
        replies = self.cat(replies, dim=1)

        # IV FULLY CONNECTED layer operation on the concatenated post, reply and similarity features
        data = self.cat([posts, replies, cos_sims], dim=1)
        data = self.fco(data)
        return F.log_softmax(data, dim=1)

    def parameter_count(self):
        """
        Returns
        -------
        int or float
            The count of all trainable weights in the neural network.
        """
        params = list(self.parameters())
        return int(sum([np.prod(list(d.size()))
                        for d in params
                        if d.requires_grad]))
