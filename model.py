from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import dgl.function as fn
from dgl.nn import SAGEConv

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.linears[0](h))
        return self.linears[1](h)
    
class GCN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, device):
        super(GCN, self).__init__()
        self.convlayers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.convlayers.append(
                    GraphConv(in_feats, h_feats)
                )
            else:
                self.convlayers.append(
                    GraphConv(h_feats, h_feats)
                )
            # Initialize the weights with xavier_uniform
            nn.init.xavier_uniform_(self.convlayers[-1].weight)
        #self.conv_out = GraphConv(h_feats, num_classes)
        
        #self.mlp = MLP(h_feats, 8, num_classes)
        #nn.init.xavier_uniform_(self.mlp.weight)
        

    def forward(self, g):
        h = g.ndata['feature']
        e = g.edata['edge_feature']
    
        for i, layer in enumerate(self.convlayers):
            # h = layer(g, h, edge_weight=e)
            h = layer(g, h)
            h = F.relu(h)
            
        return h
    
    
    
    
# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        
    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
        
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']