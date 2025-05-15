import numpy as np
import torch
from torch import nn

from models.model import Encoder, Classifier


class FairGNN_ALL(nn.Module):

    def __init__(self, encoder, nfeat, num_hidden, gnn_layer_size, gnn_hidden, cls_layer_size, device="cuda:0", data=None):
        print("----- optuna best params -----")

        torch.backends.cudnn.deterministic = True
        super(FairGNN_ALL, self).__init__()

        self.device = device

        self.model = encoder
        self.estimator = Encoder(encoder, nfeat, 1, data=data)
        self.GNN = Encoder(encoder, nfeat, num_hidden, num_layers=gnn_layer_size, hidden_size=gnn_hidden, data=data)
        self.classifier = Classifier(num_hidden, 1, cls_layer_size, False)

        self.G_loss = 0
        self.A_loss = 0

    def get_gparams(self):
        return (
                list(self.GNN.parameters())
                + list(self.classifier.parameters())
                + list(self.estimator.parameters())
        )

    def forward(self, g, x):
        s = self.estimator(x, g)
        z = self.GNN(x, g)
        y = self.classifier(z)
        return s, z, y

