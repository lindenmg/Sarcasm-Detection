from src.models.ffn_model import FFN02
from src.strategies.ffn.ffn_dataset import FFNRandomDataset
from torch.autograd.variable import Variable
from torch import manual_seed
from torch.utils.data.dataloader import DataLoader


class TestFFNModel:
    def test_model_002(self):
        post_input_size = 30
        reply_input_size = 30
        post_layer_size = 15
        reply_layer_size = 30
        post_layer_dropout = 0.5
        reply_layer_dropout = 0.5
        output_size = 2
        ds = FFNRandomDataset(reply_input_size, n_samples=50)
        manual_seed(1234)
        model = FFN02(post_input_size, reply_input_size, post_layer_size, reply_layer_size, post_layer_dropout,
                      reply_layer_dropout, output_size)
        model.train()

        dl = DataLoader(ds, batch_size=5)
        for i, (d, labels) in enumerate(dl, 1):
            post = Variable(d['posts'])
            reply = Variable(d['replies'])
            model(post, reply)

        assert True
