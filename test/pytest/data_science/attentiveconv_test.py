import torch
from torch.autograd.variable import Variable

from src.data_science.attentiveconv import AttentiveLightConv2d

# Some parameters for the initialization of the Attentive Convolution networks
word_vector_length = 3
param_dict_i = {'in_channels': 1,
                'out_channels': 1,
                'kernel_height': word_vector_length,
                'energy_func': 'dot',
                'stride': 1,
                'init': torch.nn.init.constant,
                'init_param_dict': {'val': 0.5}}


class NetworkLight(torch.nn.Module):
    """
    Class that creates a PyTorch DNN for the local tests
    """

    def __init__(self, params):
        super().__init__()
        self.attentiv_conv = AttentiveLightConv2d(**params)

    def forward(self, source, focus):
        return self.attentiv_conv(source, focus)


class TestAttentiveLightConv2d:

    def test_forward_i(self):
        # Create test data
        torch.manual_seed(1337)
        expander = torch.Tensor([1, 1, 1])
        source = torch.Tensor([0.1, 0.2, 0.3, -0.1, -0.2]).unsqueeze(dim=1)
        focus = torch.Tensor([0.3, 0.4, 0.5, 0.4, 0.3]).unsqueeze(dim=1)
        source = source * expander
        focus = focus * expander

        # Create mini-batch & channel dimensions
        source = source.unsqueeze(dim=0).unsqueeze(dim=0)
        source = torch.cat([source, source], dim=0)
        focus = focus.unsqueeze(dim=0).unsqueeze(dim=0)
        focus = torch.cat([focus, focus], dim=0)
        source = Variable(source)
        focus = Variable(focus)

        net = NetworkLight(param_dict_i)
        result = net(source, focus)
        assert (list(result.size()) == [2, 1, 5, 1])
