import os
os.environ['DGLBACKEND'] = 'pytorch'

import torch 
import torch.nn as nn 

import dgl 
from dgl.nn.pytorch import GraphConv, NNConv, WeightAndSum

import utils 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# modified from pre-tasks by removing predictor 

# if allow_zero_in_degree is False, it will temporarily add self loops to graph just for this, so that nodes with zero in degree still have an input 

allow_zero_in_degree = False 

class GCN(nn.Module): 
    def __init__(self, in_feats:int, n_classes:int, hidden_feats:list, afuncs, with_pooling_func:bool=False, device=device):
        
        super(GCN, self).__init__()

        self.allow_zero_in_degree = allow_zero_in_degree 

        self.with_pooling_func = with_pooling_func 
        self.device = device 

        # classification data - i'm making this a general thing that can be extended to classifiers but for now n_classes = 1 is equivalent to regression 
        self.n_classes = n_classes 
        self.layercount = len(hidden_feats)

        # convlayer data 
        self.in_feats=in_feats
        self.hidden_feats=hidden_feats

        # end of each layer data 
        self.activations=afuncs

        # end of each forward pass data 
        self.n_tasks=n_classes
        self.biases = [True for _ in range(self.layercount)]

        # setting convlayers 
        inner_layers = [] 
        in_feats = self.in_feats
        for i in range(self.layercount):
            layer = GraphConv(
                in_feats=in_feats, 
                out_feats=self.hidden_feats[i], 
                bias=self.biases[i], 
                allow_zero_in_degree=self.allow_zero_in_degree,
                )
            layer = layer.to(device)
            inner_layers.append(layer)

            #inner_layers.append(self.activations[i]) 

            in_feats = hidden_feats[i]
        
        self.convlayers = nn.ModuleList(inner_layers).to(device) 
            
        # set variable for pooling function 
        self.pooling_weight_and_sum_func = WeightAndSum(in_feats).to(device)

        '''
        # predictor from graph features 
        final_predictor_in_feats = in_feats*2 # since it has both weighted sum and max 

        predictor_inner_layers = [] 
        prev_hidden_feats = self.predictor_hidden_feats_list[0] 
        for curr_hidden_feats in self.predictor_hidden_feats_list[1:]: 
            predictor_inner_layers += [nn.InstanceNorm1d(prev_hidden_feats), nn.Linear(prev_hidden_feats, curr_hidden_feats), nn.ReLU()]
            prev_hidden_feats = curr_hidden_feats 

        self.final_predictor = nn.Sequential(
            nn.Dropout(self.predictor_dropout), 
            nn.Linear(final_predictor_in_feats, self.predictor_hidden_feats_list[0]), 
            nn.ReLU(), 
            *predictor_inner_layers
        ).to(device)'''

    def pooling_func(self, g, node_feats): 
        # defining pooling function to turn node features into graph features 
        # in this case, weighted sum and then max, concatenated together 
        g_sum = self.pooling_weight_and_sum_func(g, node_feats) 
        with g.local_scope(): 
            g.ndata['features'] = node_feats 
            g_max = dgl.max_nodes(g, 'features') 
        graph_feats = torch.cat([g_sum, g_max], dim=1) 
        return graph_feats 

    def forward(self, g, node_feats):
        if (not self.allow_zero_in_degree): 
            g = dgl.add_self_loop(g) # quick fix for corner case 
        node_feats = node_feats.to(self.device)
        # run through convolution layers 
        for i in range(len(self.convlayers)):
            layer = self.convlayers[i] 
            node_feats = layer(g, node_feats)

            node_feats = self.activations[i](node_feats)
            node_feats = node_feats.to(self.device) 
            #print("LAYER dONE") 
        
        # get graph features (weighted sum and max) from node features 
        if self.with_pooling_func: 
            graph_feats = self.pooling_func(g, node_feats).to(self.device)
            return graph_feats

        else: 
            graph_feats = torch.cat((node_feats[0], node_feats[0]))[None, :] # since 0 is always root in this case. TODO: just in case, make this not only be 0. 
            return graph_feats 
    


# variation with edge features considered, and conditioned as well: 

class GCN_edge_conditional(nn.Module): 
    def __init__(self, in_feats:int, n_classes:int, hidden_feats:list, afuncs, device=device):
        super(GCN_edge_conditional, self).__init__()

        self.allow_zero_in_degree = allow_zero_in_degree 

        self.device = device 

        # classification data - i'm making this a general thing that can be extended to classifiers but for now n_classes = 1 is equivalent to regression 
        self.n_classes = n_classes 
        self.layercount = len(hidden_feats)

        # convlayer data 
        self.in_feats=in_feats
        self.hidden_feats=hidden_feats

        self.conditioning_feats_count = self.hidden_feats[-1]*2 # due to nature of hidden layer 

        # end of each layer data 
        self.activations=afuncs

        # end of each forward pass data 
        self.n_tasks=n_classes
        self.biases = [True for _ in range(self.layercount)]

        # setting convlayers 
        inner_layers = [] 
        in_feats = self.in_feats
        for i in range(self.layercount):
            layer = NNConv(
                in_feats=in_feats, 
                out_feats=self.hidden_feats[i], 
                edge_func = nn.Linear(utils.bond_n_feats, in_feats*self.hidden_feats[i]), # maps edge feature to vector of shape in_feats*out_feats at each convlayer 
                aggregator_type = "sum", # NOT SURE IF THIS IS THE RIGHT ONE YET, TODO 
                bias=self.biases[i]
                )
            layer = layer.to(device)
            inner_layers.append(layer)
            
            #inner_layers.append(self.activations[i]) 

            in_feats = hidden_feats[i]
        
        self.convlayers = nn.ModuleList(inner_layers).to(device) 
            
        # set variable for pooling function 
        self.pooling_weight_and_sum_func = WeightAndSum(in_feats).to(device)


    def pooling_func(self, g, node_feats): 
        # defining pooling function to turn node features into graph features 
        # in this case, weighted sum and then max, concatenated together 
        g_sum = self.pooling_weight_and_sum_func(g, node_feats) 
        with g.local_scope(): 
            g.ndata['features'] = node_feats 
            g_max = dgl.max_nodes(g, 'features') 
        graph_feats = torch.cat([g_sum, g_max], dim=1) 
        return graph_feats 

    def forward(self, g, node_feats, edge_feats, timestep, conditioning_feats=None): # use self.conditioning_feats here yes 
        if (not self.allow_zero_in_degree): 
            g = dgl.add_self_loop(g) # quick fix for corner case 
            # add additional edges 
            add = [[0,0,0,0,0] for _ in range(len(g.nodes()))] 
            edge_feats = torch.cat((edge_feats, torch.Tensor(add).to(self.device))) 

        node_feats = node_feats.to(self.device)
        # run through convolution layers 
        for i in range(len(self.convlayers)):
            #print(self.device) 
            layer = self.convlayers[i] 
            node_feats = layer(g, node_feats, edge_feats)

            node_feats = self.activations[i](node_feats)
            node_feats = node_feats.to(self.device) 
            #print("layer done") 
            #print(g) 
            #print(self.device) 
        
        # get graph features (weighted sum and max) from node features 
        graph_feats = self.pooling_func(g, node_feats).to(self.device)

        graph_feats = graph_feats + self.timestep_to_conditioning_feats(timestep, self.conditioning_feats_count) 
        if (conditioning_feats != None): graph_feats = graph_feats + conditioning_feats 

        return nn.Tanh()(nn.ReLU()(graph_feats)) # to squish between 0 and 1 

    def timestep_to_conditioning_feats(self, t, channels): # credits: EMA diffusion model in pytorch 
        # returns a vector of length self.conditioning_feats_count as an embedding for timestep 
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    # in this specific example, graph embedding of GCN will be correct dimension for conditioning_feats 

