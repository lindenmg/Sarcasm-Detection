import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_science.attentiveconv import AttentiveLightConv2d


class CnnAttentiveI(nn.Module):

    def __init__(self, hl1_kernel_num, emb_num, emb_dim, pt_vectors
                 , self_attention=True, conv_activ=nn.PReLU
                 , conv_activ_param_dict={}, init_func_conv=nn.init.dirac
                 , init_fc_param_dict={}, init_conv_param_dict={}
                 , init_func_fc=nn.init.xavier_normal, log_features=False):
        """
        This is the 1st attentive CNN model for the SARC sarcasm detection task.

        It takes input that represents text in the form of a post and a reply.
        The actual input should be the indices of the word vectors in the
        embedding layer.
        It runs one layer of convolution with attention over the word vectors
        of the source and focus. Through ``self_attention`` you can decide if
        the replies are the foci or the posts. Source is always the reply.
        The result gets put into a final fully connected layer on which result
        the natural logarithm of it's softmax is applied.

        Parameters
        ----------
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
        self_attention: bool
            True, if the attention focus of the reply shall be the reply itself.
            The posts are not used at all in this case.
            False, else.
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
        log_features: bool
            True, if you want to return the torch.FloatTensor after the 1-max pooling,
            before it gets forwarded to the fully-connected layer. As kind of feature
            vector of the data.
            False, if you want the log-softmax classification output if the actual full
            network in form of a 1-D Tensor with len(tensor) == number of classes
        """
        super().__init__()
        self.log_features = log_features

        # I EMBEDDING
        # BTW: we don't train the Embedding Layer - not ever. Overfitting guaranteed
        self.embed = nn.Embedding(emb_num, emb_dim)
        self.embed.weight = nn.Parameter(pt_vectors, requires_grad=False)
        self.norm_embed_posts = nn.BatchNorm2d(1, affine=False)
        self.norm_embed_reply = nn.BatchNorm2d(1, affine=False)

        # II CONVOLUTION with attention
        # ... init, activation & BatchNorm of the first layer
        self.self_attention = self_attention
        self.attentive_conv = AttentiveLightConv2d(in_channels=1, out_channels=hl1_kernel_num
                                                   , kernel_height=emb_dim, init=init_func_conv
                                                   , init_param_dict=init_conv_param_dict)
        self.activation_conv = conv_activ(hl1_kernel_num, **conv_activ_param_dict)
        self.norm_conv = nn.BatchNorm2d(hl1_kernel_num, affine=False)

        # III FULLY CONNECTED
        self.fco = nn.Linear(hl1_kernel_num, 2, bias=True)
        init_func_fc(self.fco.weight, **init_fc_param_dict)

    def forward(self, posts, replies):
        """
        Forward pass of the 1st attentive CNN model for the SARC sarcasm detection task.

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
        posts = self.norm_embed_posts(posts.unsqueeze(1))
        replies = self.norm_embed_reply(replies.unsqueeze(1))

        # II CONVOLUTION with attention layer
        # - activation, batch-norm and max pooling
        if self.self_attention:
            features = self.attentive_conv(replies, replies)
        else:
            features = self.attentive_conv(replies, posts)
        features = self.norm_conv(self.activation_conv(features)).squeeze()
        features = F.adaptive_max_pool1d(features, 1).squeeze()

        # III OUTPUT CHOICE
        if self.log_features:
            result = features
        else:
            # III.ii FULLY CONNECTED layer operation
            data = self.fco(features)
            result = F.log_softmax(data, dim=1)
        return result

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
