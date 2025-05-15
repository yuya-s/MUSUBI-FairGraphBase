import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

from models.model import Encoder, Classifier
import torch.nn.functional as F

class NIFTY_GAT(torch.nn.Module):
    def __init__(self,
                 seed,
                 num_features,
                 num_hidden=16,
                 num_proj_hidden=16,
                 encoder="sage",
                 gnn_layer_size=1,
                 gnn_hidden=16,
                 cls_layer_size=1,
                 sim_coeff=0.5,
                 nclass=1,
                 device="cuda:0",
                 data=None
                 ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        super(NIFTY_GAT, self).__init__()

        self.device = device

        self.encoder_name = encoder

        self.encoder = Encoder(in_channels=num_features, out_channels=num_hidden, base_model=encoder,
                               num_layers=gnn_layer_size, hidden_size=gnn_hidden, data=data).to(device)

        self.sim_coeff = sim_coeff


        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden)
        )

        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        self.c1 = Classifier(num_hidden, nclass, num_layers=cls_layer_size, use_spectral_norm=True)

        for m in self.modules():
            self.weights_init(m)

        self = self.to(device)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def parameters_1(self):
        par_1 = list(self.encoder.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(
            self.fc3.parameters()) + list(self.fc4.parameters())
        return par_1

    def parameters_2(self):
        par_2 = list(self.c1.parameters()) + list(self.encoder.parameters())
        return par_2

    def forward(self, x: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, z):
        return self.c1(z)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D(self, x1, x2):
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()


