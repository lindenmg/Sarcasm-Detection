from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Conv2d

"""
Overview over the classes in this file:

"""


class BaseAttentiveConv2d(ABC, torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size
                 , energy_func='dot', source_reweight=False, stride=1
                 , init=torch.nn.init.dirac, init_param_dict={}):
        """
        Implementation of Attentive Convolution (convolution with true attention)

        Unlike in the paper this implementation uses a kernel which goes over the
        2nd dimension (e.g. the word vectors).
        If ``energy_func`` is 'biliniear', then the input has to have always
        the same length (``dim_1st``)

        Parameters
        ----------
        in_channels: int
            Number of channels in the input
        out_channels: int
            Number of channels produced by the convolution
        kernel_size: tuple
            Size of the convolving kernel (width x height) for
            all convolutions
        energy_func: 'dot' or 'bilinear':
            Generates a matching-score between the hidden states of the
            focus data and the data on which the attention shall be focused
        source_reweight: bool
            True if the source shall be re-weighted with a convolution
            kernel of size (1, 1) - each scalar data-point gets re-weighted
        stride: int or tuple
            Stride of the convolution. Default: 1
        init: class of torch.nn.init
            The initialization function used for the
            convolutions in this module
        init_param_dict: dict, optional
            Contains the parameters for the init function -
            apart from the actual tensor data.
            Should have the form {'param_name': param_value}
        Notes
        -----
        Shape:
            * Input: (N, in_channels, H_in, W_in)
            * Output: (N, out_channels, H_out, W_out), where
             H_out = floor((H_in − kernel_size[0] + 2) ⁄ stride[0] + 1) and
             W_out =  floor((W_in − kernel_size[1] + 2) ⁄ stride[1] + 1)
        See Also
        --------
        https://arxiv.org/abs/1710.00519
        https://pytorch.org/docs/0.3.1/_modules/torch/nn/modules/conv.html#Conv2d
        """
        super().__init__()
        self.calc_benefit = source_reweight
        conv_init = init
        padding = kernel_size[0] // 2

        if not (energy_func in ['dot', 'bilinear']):
            raise ValueError("Value of energy_func must either be 'dot' or "
                             "'bilinear', but is {:}".format(energy_func))
        else:
            self.energy_func = energy_func

        if source_reweight:
            uni_conv_num = 3
        else:
            uni_conv_num = 2

        uni_conv = {'in_channels': in_channels
            , 'out_channels': 1
            , 'kernel_size': (1, kernel_size[1])
            , 'stride': 1}
        tri_conv = uni_conv.copy()
        tri_conv['kernel_size'] = kernel_size
        tri_conv['padding'] = (padding, 0)

        # ---------------------------------------------------------------------
        # Formula (7) - convolution for the attention functions
        # - Gated Convolution functions for Advanced Attentive Convolution
        #   Order: Source, Focus, Beneficiary - last one has no triple convolution
        self.conv_tri = ModuleList([Conv2d(**tri_conv) for _ in range(2)])
        [conv_init(conv.weight, **init_param_dict) for conv in self.conv_tri]
        for i, module in enumerate(self.conv_tri):
            name = 'conv_tri_' + self.__give_sub_name(i)
            self.add_module(name, module)

        self.conv_uni = ModuleList([Conv2d(**uni_conv)
                                    for _ in range(uni_conv_num)])
        [conv_init(conv.weight, **init_param_dict) for conv in self.conv_uni]
        for i, module in enumerate(self.conv_uni):
            name = 'conv_uni_' + self.__give_sub_name(i)
            self.add_module(name, module)

        # Formula (8) - gate for the convolution
        self.gate_tri = ModuleList([Conv2d(**tri_conv) for _ in range(2)])
        [conv_init(conv.weight, **init_param_dict) for conv in self.gate_tri]
        for i, module in enumerate(self.gate_tri):
            name = 'gate_tri_' + self.__give_sub_name(i)
            self.add_module(name, module)
        self.gate_uni = ModuleList([Conv2d(**uni_conv)
                                    for _ in range(uni_conv_num)])
        [conv_init(conv.weight, **init_param_dict) for conv in self.gate_uni]
        for i, module in enumerate(self.gate_uni):
            name = 'gate_uni_' + self.__give_sub_name(i)
            self.add_module(name, module)

        # ---------------------------------------------------------------------
        # Formula (6) - the actual convolution at the end
        tri_conv['out_channels'] = out_channels
        tri_conv['stride'] = stride
        tri_conv['bias'] = False

        # - We have x2 ‘in_channels’ because of stacked triple & uni convolution results
        uni_conv['out_channels'] = out_channels
        uni_conv['in_channels'] = 2 * in_channels
        uni_conv['kernel_size'] = 1
        uni_conv['stride'] = stride

        self.conv = torch.nn.Conv2d(**tri_conv)
        conv_init(self.conv.weight, **init_param_dict)
        self.add_module('conv', self.conv)

        # - Conv1d because we reduce the 2nd dim (e.g. word vectors)
        #   to a scalar in the attentive weight creation part
        self.context_conv = torch.nn.Conv1d(**uni_conv)
        conv_init(self.context_conv.weight, **init_param_dict)
        self.add_module('context_conv', self.context_conv)

    @staticmethod
    def __give_sub_name(i):
        name = 'beneficiary'
        if i == 0:
            name = 'source'
        elif i == 1:
            name = 'foci'
        return name

    @abstractmethod
    def forward(self, source, focus):
        raise NotImplementedError


class AttentiveLightConv2d(BaseAttentiveConv2d):

    def __init__(self, in_channels, out_channels, kernel_height, energy_func='dot'
                 , stride=1, init=torch.nn.init.dirac, init_param_dict={}):
        """
        Implementation of Attentive Convolution (convolution with true attention)

        Unlike in the paper this implementation uses a kernel which goes over the
        2nd dimension (e.g. the word vectors).
        If ``energy_func`` is 'biliniear', then the input has to have always
        the same length (``dim_1st``)

        Parameters
        ----------
        in_channels: int
            Number of channels in the input
        out_channels: int
            Number of channels produced by the convolution
        kernel_height: int
            Height of the convolving kernel (width x height)
            for all convolutions
        energy_func: 'dot' or 'bilinear':
            Generates a matching-score between the hidden states
            of the focus data and the data on which the attention
            shall be focused
        stride: int or tuple
            Stride of the convolution. Default: 1
        init: class of torch.nn.init
            The initialization function used for the
            convolutions in this module
        init_param_dict: dict, optional
            Contains the parameters for the init function -
            apart from the actual tensor data.
            Should have the form {'param_name': param_value}
        Notes
        -----
        Shape:
            * Input: (N, in_channels, H_in, W_in)
            * Output: (N, out_channels, H_out, W_out), where
             H_out = floor((H_in − 1) ⁄ stride[0] + 1) and
             W_out =  floor((W_in − kernel_size[1] + 2) ⁄ stride[1] + 1)
        See Also
        --------
        https://arxiv.org/abs/1710.00519
        https://pytorch.org/docs/0.3.1/_modules/torch/nn/modules/conv.html#Conv2d
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels
                         , kernel_size=(3, kernel_height), energy_func=energy_func
                         , source_reweight=False, stride=stride, init=init
                         , init_param_dict=init_param_dict)

        # ---------------------------------------------------------------------
        # Formula (2) - intermediate step for (attentive) context weight
        # - source & attention focus get reduced to 1d, therefore (1, 1)
        if energy_func == 'bilinear':
            self.en_bl = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
            self.register_parameter('energy_func_bl', self.en_bl)
            init(self.en_bl, init_param_dict)

    def forward(self, source, focus):
        """
        Executes Advanced Attentive Convolution

        Parameters
        ----------
        source: torch.FloatTensor
            The source text for the convolution
        focus: torch.FloatTensor
            The text where the attention shall be focused on.
            It can be the same as source
        Returns
        -------
        torch.FloatTensor
            Shape: (N, C_out, H_out, 1), where
            H_out = floor((H_in − 1) ⁄ stride[0] + 1)
        See Also
        --------
        https://arxiv.org/abs/1710.00519
        """

        # Formulas (7)-(9): gated convolution
        def gconv(conv, gate, state_hh):
            o = F.leaky_relu(conv(state_hh)).squeeze(dim=3)
            g = F.sigmoid(gate(state_hh)).squeeze(dim=3)

            # Because we have reduced the 2nd input dimension, e.g. word vector
            # to a scalar trough the convolution, we also have to reduce the
            # original data points in the same way for the gating mechanism
            return g * state_hh.norm(dim=3) + (1 - g) * o

        # Formulas (10),(11),(13) - creating attention source and focus
        # - With optional re-weight (13) of the source input which is used for
        #   normal convolution (beneficiary), which “benefits” from attention
        if self.calc_benefit:
            beneficiary = gconv(self.conv_uni[2], self.gate_uni[2], source)
        else:
            beneficiary = source
        src_tri = gconv(self.conv_tri[0], self.gate_tri[0], source)
        src_uni = gconv(self.conv_uni[0], self.gate_uni[0], source)
        focus_tri = gconv(self.conv_tri[1], self.gate_tri[1], focus)
        focus_uni = gconv(self.conv_uni[1], self.gate_uni[1], focus)

        # Formula (12)
        # - Uni- and tri-gated convolution become two different channels / features
        source = torch.cat([src_uni, src_tri], dim=1)
        focus = torch.cat([focus_uni, focus_tri], dim=1)

        # Formulas (2) - the ‘if’ part - and (1) - the ‘else’ part
        if self.energy_func == 'bilinear':
            e = source * self.en_bl * focus
        else:
            # ‘*’ suffices, because each former vector of a data
            # point is now a scalar. So we multiply each new
            # data point of the source with every one of the focus
            e = source.unsqueeze(dim=3) * focus.unsqueeze(dim=2)

        # Formula (4) - calculation of the attentive context
        context = F.softmax(e, dim=3).matmul(focus.unsqueeze(dim=3)).squeeze()

        # Formula (6) - actual convolution with the attentive context
        return self.conv(beneficiary) + self.context_conv(context).unsqueeze(dim=3)
