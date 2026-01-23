import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import torch
 

class MultiAttrGAT(nn.Module):
    # def __init__(self, num_road_type, rt_emb_dim=16,
    #              lanes_emb_dim=8, oneway_emb_dim=4,
    #              cont_dim=2, hidden=32, heads=2, dropout=0.1):
    def __init__(self, num_road_type, rt_emb_dim=16,
                 lanes_emb_dim=8, oneway_emb_dim=4,
                 cont_dim=1, hidden=32, heads=2, dropout=0.1):
        super().__init__()

        # Embeddings for categorical inputs
        self.rt_emb = nn.Embedding(num_road_type, rt_emb_dim)
        # lanes ids: 0,1,2 + MASK(3) + MISSING(4) => size 5
        self.lanes_emb = nn.Embedding(5, lanes_emb_dim)
        # oneway ids: 0,1 + MASK(2) => size 3
        self.oneway_emb = nn.Embedding(3, oneway_emb_dim)
 
        in_dim = cont_dim + rt_emb_dim + lanes_emb_dim + oneway_emb_dim

        self.gat1 = GATv2Conv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=heads, concat=False, dropout=dropout)

        # Heads
        self.head_road_type = nn.Linear(hidden, num_road_type)  # CE
        self.head_lanes   = nn.Linear(hidden, 3)            # CE for 3 classes
        self.head_oneway  = nn.Linear(hidden, 1)            # BCE logit
        self.head_width   = nn.Linear(hidden, 1)            # Huber
        # self.head_length  = nn.Linear(hidden, 1)            # Huber

    def forward(self, x_cont, road_type_in, lanes_in, oneway_in, edge_index):
        hwy = self.rt_emb(road_type_in)
        lan = self.lanes_emb(lanes_in)
        onw = self.oneway_emb(oneway_in)

        x = torch.cat([x_cont, hwy, lan, onw], dim=1)

        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = F.elu(h)

        out = {
            "road_type": self.head_road_type(h),
            "lanes":   self.head_lanes(h),
            "oneway":  self.head_oneway(h).squeeze(-1),
            "width":   self.head_width(h).squeeze(-1),
        }
        return out