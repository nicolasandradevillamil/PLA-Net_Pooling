import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer, MLP, MM_AtomEncoder
from model.model_encoder import AtomEncoder, BondEncoder

import dgl
from dgl.nn import AvgPooling,SortPooling,GlobalAttentionPooling

import logging
import pdb

class DeeperGCN(torch.nn.Module):
    def __init__(self, args, is_prot=False, saliency=False):
        super(DeeperGCN, self).__init__()

        # Set PM configuration
        if is_prot:
            print("isprot")
            self.num_layers = args.num_layers_prot
            mlp_layers = args.mlp_layers_prot
            hidden_channels = args.hidden_channels_prot
            self.msg_norm = args.msg_norm_prot
            learn_msg_scale = args.learn_msg_scale_prot
            self.conv_encode_edge = args.conv_encode_edge_prot

        # Set LM configuration
        else:
            self.num_layers = args.num_layers
            mlp_layers = args.mlp_layers
            hidden_channels = args.hidden_channels
            self.msg_norm = args.msg_norm
            learn_msg_scale = args.learn_msg_scale
            self.conv_encode_edge = args.conv_encode_edge

        # Set overall model configuration
        self.dropout = args.dropout
        self.block = args.block
        self.add_virtual_node = args.add_virtual_node
        self.training = True
        self.args = args

        num_classes = args.nclasses
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p

        norm = args.norm

        graph_pooling = args.graph_pooling

        # Print model parameters
        print(
            "The number of layers {}".format(self.num_layers),
            "Aggr aggregation method {}".format(aggr),
            "block: {}".format(self.block),
        )
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        elif self.block == "res":
            print("GraphConv->LN/BN->ReLU->Res")
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")
        elif self.block == "plain":
            print("GraphConv->LN/BN->ReLU")
        else:
            raise Exception("Unknown block Type")

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3, norm=norm))

        # Set GCN layer configuration
        for layer in range(self.num_layers):
            if conv == "gen":
                gcn = GENConv(
                    hidden_channels,
                    hidden_channels,
                    args,
                    aggr=aggr,
                    t=t,
                    learn_t=self.learn_t,
                    p=p,
                    learn_p=self.learn_p,
                    msg_norm=self.msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    encode_edge=self.conv_encode_edge,
                    bond_encoder=True,
                    norm=norm,
                    mlp_layers=mlp_layers,
                )
            else:
                raise Exception("Unknown Conv Type")
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # Set embbeding layers
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        
        if saliency:
            self.atom_encoder = MM_AtomEncoder(emb_dim=hidden_channels)
        else:
            self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)
        # Set type of pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "avg":
            self.pool = AvgPooling()
        elif graph_pooling == "sort":
            self.pool = SortPooling(k=1)
            print("using sort")
        elif graph_pooling == "gap":
            print("using gap")
            self.gatelayer = torch.nn.Linear(128,1)
            self.pool = GlobalAttentionPooling(self.gatelayer)
        else:
            raise Exception("Unknown Pool Type")

        # Set classification layer
        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_classes)


    def create_batched_dgl_graph(self,h, batch, edge_index):
        #breakpoint()
        src, dst = edge_index #edge_index.shape = (2, num_edges) primero source luego destination
        large_graph = dgl.graph((src, dst)) #crea un grafo grande 
        large_graph.ndata['h'] = h #se agregan features al grafo grande (asume correspondencia indices de nodos)
        _, node_counts = torch.unique_consecutive(batch, return_counts=True) #cuenta cuantos nodos hay en cada grafo
        graph_starts = torch.cat((torch.tensor([0],device=h.device), torch.cumsum(node_counts, dim=0)[:-1])) #obtiene indices iniciales de cada grafo dentro del grafo grande
        graphs = [large_graph.subgraph(torch.arange(start, start + count,device=h.device)) for start, count in zip(graph_starts, node_counts)] #Separa el grafo grande en grafos pequeños
        batched_graph = dgl.batch(graphs) #crea un batch de grafos

        return batched_graph

    def forward(self, input_batch, dropout=True, embeddings=False):
        #breakpoint()
        x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = self.atom_encoder(x)

        if self.add_virtual_node: 
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1)
                .to(edge_index.dtype)
                .to(edge_index.device)
            )
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == "res+":
            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                if dropout:
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = (
                        global_add_pool(h2, batch) + virtualnode_embedding
                    )
                    if dropout:
                        virtualnode_embedding = F.dropout(
                            self.mlp_virtualnode_list[layer - 1](
                                virtualnode_embedding_temp
                            ),
                            self.dropout,
                            training=self.training,
                        )

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h
            #breakpoint()
            h = self.norms[self.num_layers - 1](h)
            if dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "res":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")

        elif self.block == "plain":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception("Unknown block Type")
        
        batch_graph = self.create_batched_dgl_graph(h, batch, edge_index)
        batch_h = batch_graph.ndata['h']
        #breakpoint()
        h_graph = self.pool(batch_graph,batch_h)

        if self.args.use_prot or embeddings:
            return h_graph
        else:
            return self.graph_pred_linear(h_graph)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print("Final t {}".format(ts))
            else:
                logging.info("Epoch {}, t {}".format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print("Final p {}".format(ps))
            else:
                logging.info("Epoch {}, p {}".format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print("Final s {}".format(ss))
            else:
                logging.info("Epoch {}, s {}".format(epoch, ss))
