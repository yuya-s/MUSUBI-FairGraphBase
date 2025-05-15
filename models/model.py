import scipy
from torch.nn import Linear
from torch.nn.utils import spectral_norm
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from utils.utils import *
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GINConv, SAGEConv, GCNConv, GATConv, JumpingKnowledge

class Encoder(torch.nn.Module):
    def __init__(self, base_model, in_channels: int, out_channels: int, hidden_size = -1, num_layers = 1, data=None):
        super(Encoder, self).__init__()
        print("ENCODER: ", base_model)
        self.base_model = base_model

        if hidden_size == -1:
            hidden_size = out_channels

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_size = hidden_size
            out_size = hidden_size
            if i == 0:
                in_size = in_channels
            if i == num_layers - 1:
                out_size = out_channels
            if self.base_model == "gat":
                conv = GAT_encoder(in_size, out_size)
            elif self.base_model == 'gcn':
                conv = GCN_encoder(in_size, out_size)
            elif self.base_model == 'sage':
                conv = SAGE_encoder(in_size, out_size)
            elif self.base_model == 'h2gcn':
                conv = H2GCN_encoder(in_channels=in_size, \
                                     hidden_channels=hidden_size, out_channels=out_size, \
                                     edge_index=data.edge_index, num_nodes=data.num_nodes)
            self.convs.append(conv)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for i, conv in enumerate(self.convs):
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        return x

class GCN_encoder(nn.Module):
    def __init__(self, num_features, hidden):
        super(GCN_encoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden, normalize=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class GAT_encoder(nn.Module):
    def __init__(self, num_features, hidden, use_transition=True):
        super(GAT_encoder, self).__init__()
        nfeat = num_features
        nhid = hidden
        dropout = 0.5
        nheads = 1
        print("NHID: ", nhid)
        print("NFEAT: ", nfeat)
        self.conv1 = GATConv(nfeat, nhid, heads=nheads, dropout=dropout)
        self.use_transition = use_transition
        if use_transition:
            self.transition = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(nhid * nheads),
                nn.Dropout(p=dropout)
            )
            for m in self.modules():
                self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.use_transition:
            x = x.flatten(start_dim=1)
            x = self.transition(x)
        return x



class SAGE_encoder(nn.Module):
    def __init__(self, num_features, hidden, dropout=0.5):
        super(SAGE_encoder, self).__init__()

        self.conv1 = SAGEConv(num_features, hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=dropout)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
         self.conv1.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, use_spectral_norm=False):
        super(Classifier, self).__init__()

        self.lins = nn.ModuleList()
        for i in range(num_layers):
            in_size = in_features
            out_size = in_features
            if i == num_layers - 1:
                out_size = out_features
            if use_spectral_norm:
                lin = spectral_norm(Linear(in_size, out_size))
            else:
                lin = Linear(in_size, out_size)
            self.lins.append(lin)

    def reset_parameters(self):
        for i, lin in enumerate(self.lins):
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
        return x


class H2GCNConv(nn.Module):

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN_encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN_encoder, self).__init__()

        self.feature_embed = MLP_for_H2GCN(in_channels, hidden_channels,
                                 hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_channels * (2 ** len(self.convs))))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels * (2 ** (num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, x, edge_index):
        n = len(x)

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x

class MLP_for_H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP_for_H2GCN, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
