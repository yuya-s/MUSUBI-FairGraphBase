import torch
from torch import nn

from models.model import Encoder, Classifier
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class Vanilla_GNN(torch.nn.Module):
    def __init__(
            self,
            encoder,
            num_feature,
            num_hidden,
            gnn_layer_size,
            gnn_hidden,
            cls_layer_size,
            device="cuda:0",
            data=None
    ):
        super(Vanilla_GNN, self).__init__()
        self.device = device

        self.encoder = Encoder(encoder, num_feature, num_hidden,
                               hidden_size=gnn_hidden, num_layers=gnn_layer_size, data=data).to(device)

        self.drop_edge_rate_1 = self.drop_edge_rate_2 = 0.5
        self.drop_feature_rate_1 = self.drop_feature_rate_2 = 0.5

        self.c1 = Classifier(num_hidden, 1, num_layers=cls_layer_size, use_spectral_norm=True)

        for m in self.modules():
            self.weights_init(m)

        self.to(device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self.c1(z)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D_entropy(self, x1, x2):
        x2 = x2.detach()
        return (-torch.max(F.softmax(x2), dim=1)[0] * torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

    def D(self, x1, x2):
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

