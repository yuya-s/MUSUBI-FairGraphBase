import torch
from torch import nn
from models.model import Encoder


class StructualModel(nn.Module):
    def __init__(self, encoder, nfeat, num_hidden, gnn_layer_size, gnn_hidden, device="cuda:0", data=None):
        super(StructualModel, self).__init__()
        self.encoder = Encoder(encoder, nfeat, num_hidden, num_layers=gnn_layer_size, hidden_size=gnn_hidden, data=data).to(device)
        
        self.free_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(data.x.shape[0], nfeat)))
        self.cl = nn.Linear(num_hidden, 1)       
        
        for m in self.modules():
            self.weights_init(m)

        self.to(device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, edge_index):
        structural_h = self.encoder(self.free_embedding, edge_index)
        cl = self.cl(structural_h)

        return cl, structural_h


class AttributeModel(nn.Module):
    def __init__(self, encoder, nfeat, num_hidden, gnn_layer_size, gnn_hidden, device="cuda:0", data=None):
        super(AttributeModel, self).__init__()
        self.device = device

        self.encoder = Encoder(encoder, nfeat, num_hidden, hidden_size=gnn_hidden, num_layers=gnn_layer_size, data=data).to(device)
        self.cl = nn.Linear(num_hidden, 1) 

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
        attribute_h = self.encoder(x, edge_index)
        cl = self.cl(attribute_h)
        return cl, attribute_h



class InteractionModel(nn.Module):
    def __init__(self, num_hidden, device="cuda:0"):
        super(InteractionModel, self).__init__()
        self.num_hidden = num_hidden
        self.lin1 = nn.Linear(num_hidden, num_hidden // 2)
        self.bn1 = nn.BatchNorm1d(num_hidden // 2)
        self.relu = nn.ReLU()
        self.cl = nn.Linear(num_hidden // 2, 1)

        for m in self.modules():
            self.weights_init(m)

        self.to(device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, concat_emd):
        entangled_h = self.lin1(concat_emd)
        entangled_h = self.relu(self.bn1(entangled_h))
        potential_h = entangled_h - (concat_emd[:, :self.num_hidden // 2] + concat_emd[:, self.num_hidden // 2:])
        cl = self.cl(potential_h)

        return cl, potential_h 
    
    
class WDapproximator(nn.Module):
    def __init__(self, nfeat, device):
        super(WDapproximator, self).__init__()
        self.lin = nn.Linear(nfeat, 1)
        self.to(device)

    def forward(self, x):
        h = self.lin(x)
        return h
    
