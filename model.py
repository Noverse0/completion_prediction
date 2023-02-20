from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import dgl

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class GCN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, num_classes, device):
        super(GCN, self).__init__()
        self.convlayers = nn.ModuleList()
        for layer in range(num_layers - 1):
            if layer == 0:
                self.convlayers.append(
                    GraphConv(in_feats, h_feats)
                )
            else:
                self.convlayers.append(
                    GraphConv(h_feats, h_feats)
                )
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g):
        h = g.ndata['feature']
        #ew = g.edata['edge_feature']
        for i, layer in enumerate(self.convlayers):
            h = layer(g, h)
            h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")
    
    