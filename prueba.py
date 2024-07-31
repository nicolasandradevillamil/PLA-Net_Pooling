import dgl
import torch as th
from dgl.nn import GlobalAttentionPooling
import pdb

breakpoint()
g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
g1_node_feats = th.rand(3, 5)  # feature size is 5

g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
g2_node_feats = th.rand(4, 5)  # feature size is 5

gate_nn = th.nn.Linear(5, 1)  # the gate layer that maps node feature to scalar
gap = GlobalAttentionPooling(gate_nn)  # create a Global Attention Pooling layer

gap(g1, g1_node_feats)

batch_g = dgl.batch([g1, g2])
batch_f = th.cat([g1_node_feats, g2_node_feats], 0)

gap(batch_g, batch_f)
