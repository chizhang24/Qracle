from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool

import torch 

class GCN(torch.nn.Module):
    def __init__(self, n_x, n_y, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_x, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, n_y)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin(h)

        return h

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, n_x, n_y, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(n_x, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, n_y)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h


class DGNN(torch.nn.Module):
  def __init__(self, n_x, n_y, dim_h):
    super(DGNN, self).__init__()
    hidden_channels_gcn = dim_h * 2
    hidden_channels_gat = hidden_channels_gcn * 2
    hidden_channels_gin = hidden_channels_gat * 2

    # Initialize the first GCNConv layer in the forward method
    self.conv1 = GCNConv(n_x, hidden_channels_gcn)
    self.hidden_channels_gcn = hidden_channels_gcn
    self.conv2 = GCNConv(hidden_channels_gcn, hidden_channels_gcn)
    self.gat_conv1 = GATConv(hidden_channels_gcn, hidden_channels_gat)
    self.gat_conv2 = GATConv(hidden_channels_gat, hidden_channels_gat)

    mlp = torch.nn.Sequential(
        Linear(hidden_channels_gat, hidden_channels_gin),
        ReLU(),
        Linear(hidden_channels_gin, hidden_channels_gin)
    )
    self.gin_conv1 = GINConv(mlp)
    self.out = Linear(hidden_channels_gin, n_y)

  def forward(self, x, edge_index, batch):
    x = F.relu(self.conv1(x, edge_index))
    x = F.relu(self.conv2(x, edge_index))
    x = F.relu(self.gat_conv1(x, edge_index))
    x = F.relu(self.gat_conv2(x, edge_index))
    x = F.relu(self.gin_conv1(x, edge_index))
    x = global_mean_pool(x, batch)
    x = self.out(x)
    return x
