import os
os.environ['DGLBACKEND'] = 'pytorch'

from sys import stdout 

import torch 
import torch.nn as nn 
import torch.functional as F 

from rdkit import Chem 
from rdkit.Chem import AllChem 

import matplotlib.pyplot as plt 

import dgl 
import networkx as nx 

from dgl.data.utils import save_graphs, load_graphs 

import copy 

import utils 
import params 
from GCN import GCN, GCN_edge_conditional
#from utils import FTreeNode, FTree, FTreeToGraph 

import numpy as np 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 


# representations used in this 


class MassSpec(): # represents the mass spec, which is an environment variable, as a fragment tree. This representation is trained by trying to 

    embedding_size = 256 # 128 * 2 

    # TODO - make sure this works yes 
    num_buckets = 450  
    bucket_size = 5 # i'm assuming ppm? but not sure 
    num_peaks_considered = 30 
    
    # set the loss functions 
    mseloss = nn.MSELoss() 
    def rmseloss(pred, labels): 
        return torch.sqrt(MassSpec.mseloss(pred, labels)) 
    
    def __init__(self, training, ftree:utils.FTree, path, amplitude_path, gcn_learning_rate=params.learning_rate, predictor_learning_rate=params.learning_rate, with_pooling_func = False, device=device): # self loop is root 

        self.device = device 

        self.training = training 
        self.path = path 
        self.amplitude_path = amplitude_path 

        self.gcn_learning_rate = gcn_learning_rate 
        self.predictor_learning_rate = predictor_learning_rate 
        
        #GCN 
        # set the parameters of the model 
        self.layersizes = [64, 128] # all hidden layers (or just one) and output; output layer goes through pooling function so 
        self.afuncs = [nn.ReLU() for _ in range(len(self.layersizes)-1)] 
        self.afuncs.append(nn.ReLU()) # last activation function 
        
        
        if (not training): 
            # not training 
            # load model 
            self.model = GCN(utils.FTreeNode.n_feats, 1, self.layersizes, self.afuncs, with_pooling_func=with_pooling_func, device=self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            
            self.ftree = ftree 
            self.atom_counts = ftree.nodes[ftree.root].counts #utils.formula_to_atom_counts(ftree.nodes[ftree.root] 

            self.run(ftree) 
        
        else: 
            # training - predicting amplitude of each peak using m/z buckets; 0-1 because mass spec is normalized 

            # make 0-1 amplitude predictor, using sigmoid function to squeeze between 0 and 1 

            # changed to try except to try to save 


            amplitude_predictor_dropout = 0.0
            amplitude_predictor_hidden_feats_list = [128, 256, MassSpec.num_buckets] # because predicting for all positions in mass spec 
            amplitude_predictor_in_feats = self.layersizes[-1]*2 # since it has both weighted sum and max after pooling function 
            
            amplitude_predictor_inner_layers = [] 
            prev_hidden_feats = amplitude_predictor_hidden_feats_list[0] 
            for curr_hidden_feats in amplitude_predictor_hidden_feats_list[1:]: 
                amplitude_predictor_inner_layers += [nn.ReLU(), nn.Linear(prev_hidden_feats, curr_hidden_feats), nn.Sigmoid()]
                prev_hidden_feats = curr_hidden_feats 
                                
            self.amplitude_predictor = nn.Sequential(
                nn.Dropout(amplitude_predictor_dropout), 
                nn.Linear(amplitude_predictor_in_feats, amplitude_predictor_hidden_feats_list[0]), 
                nn.Sigmoid(), 
                *amplitude_predictor_inner_layers
            ).to(self.device)

            try: 
                self.amplitude_predictor.load_state_dict(torch.load(amplitude_path, map_location=self.device)) 
                self.amplitude_predictor.eval() 
            except Exception as e:
                print("CREATING NEW MASSSPEC AMPLITUDE PREDICTOR MODEL") 
                print(e)  

                if (amplitude_path == None or len(amplitude_path.strip()) == 0):  self.amplitude_path = "./RL_attempt/models/FTreeAmplitudePredictor_1.pt" 
                else: self.amplitude_path = amplitude_path 

            self.amplitude_optimizer = torch.optim.Adam(self.amplitude_predictor.parameters(), lr=self.predictor_learning_rate)
            
        


            # main model 

            try: 
                # load model 
                self.model = GCN(utils.FTreeNode.n_feats, 1, self.layersizes, self.afuncs, with_pooling_func=with_pooling_func, device=self.device)
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
            except Exception as e: # create model 
                print("CREATING NEW MASSSPEC GCN MODEL") 
                print(e) 
                self.model = GCN(utils.FTreeNode.n_feats, 1, self.layersizes, self.afuncs, with_pooling_func=with_pooling_func, device=self.device)

                if (self.path == None or len(self.path.strip()) == 0): self.path = "./RL_attempt/models/FTreeGCN_1.pt" 
                else: self.path = path 

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.gcn_learning_rate)
        
        
        # change save paths, just in case 
        #self.path = self.path[:-3] + "1" + self.path[-3:]
        #self.amplitude_path = self.amplitude_path[:-3] + "1" + self.amplitude_path[-3:]
        

    def run(self, ftree): 
        # convert graph to dgl graph 
        graph = utils.FTreeToGraph(ftree).to(self.device) 

        #print("BONDTYPES:", graph.edges())  

        # run model (only needs to run once if not training) 
        self.pred = self.model(graph, graph.ndata["features"].float().to(self.device)) 
        if (not self.training): 
            self.pred = self.pred.detach() # since should not learn 

        #print("SELF.PRED:",self.pred) 

        #print("BONDTYPES:", graph.edges())  
    
    def toTensor(self, device=None): 
        # run GNN to get graph-level embedding 
        if device==None: device=self.device
        return self.pred.to(device)
    

    # to try to train it to preserve features, we should run this through some dense layers, also inputting mass spec in some way, 
    # and asking it to guess the similarity between the mass spec used in ftree, and the mass spec inputted, 
    # using the information of the ftree it stored, and the mass spec 

    # actually, that's not very feasible either, instead just predict number of peaks and sum of the m/z ratios of all peaks 

    def index_select_loss(pred, label, top_k): 
        return MassSpec.rmseloss(pred[0, top_k], label[top_k])

    def train(self, ftrees, labels): # trains for one epoch 
        #print("FTREES:",ftrees)

        total_loss = 0 
        num_cases = 0 

        # for each mass spec, labels is list of peaks for each mass spec. Peaks are identified by being big amplitude and make up 80% of whole mass spec yes. Refer to utils for more details 
        for i in range(len(ftrees)): 

            self.run(ftrees[i]) 
            embedding = self.pred 

            # run the predictor 
            amplitude_res = self.amplitude_predictor(embedding).to(self.device)  
            
            # get top_k 
            top_k = torch.argsort(amplitude_res)[0,-MassSpec.num_peaks_considered:].to(self.device)  
            # convert to bool tensor 
            bool_top_k = torch.zeros(MassSpec.num_buckets, dtype=torch.bool).to(self.device) 
            bool_top_k[top_k] = True 

            target = torch.where(bool_top_k, torch.Tensor(labels[i]).to(self.device), torch.zeros(len(labels[i])).to(self.device)).to(self.device) 

            '''
            print("STUFFFFF") 
            print(amplitude_res) 
            print(top_k) 
            print("DEBUG RES:::") 
            print(amplitude_res[0, top_k]) 
            print(labels[i][top_k]) 
            print(nn.MSELoss()(amplitude_res[0, top_k], labels[i][top_k])) 
            print(torch.sqrt(nn.MSELoss()(amplitude_res[0, top_k], labels[i][top_k]))) 
            '''

            
            #amplitude_loss = MassSpec.index_select_loss(amplitude_res, labels[i].to(self.device), top_k).to(self.device) 
            
            amplitude_loss = MassSpec.mseloss(amplitude_res[0], target) 

            total_loss += amplitude_loss.item() 
            #print("Amplitude Loss:", amplitude_loss)

            self.optimizer.zero_grad()
            self.amplitude_optimizer.zero_grad() 
            amplitude_loss.backward()
            #print(amplitude_loss)
            self.optimizer.step()
            self.amplitude_optimizer.step() 

            num_cases += 1 
        
        return total_loss / num_cases 

    
    def save(self, path=None):
        if path==None: path = self.path  
        torch.save(self.model.state_dict(), path) 

    def save_amplitude_predictor(self, path=None): 
        if path==None: path = self.amplitude_path 
        torch.save(self.amplitude_predictor.state_dict(), path) 



class Action(): # represents an action taken 
    # REMOVE SOME H FROM FIRST AND SOME H FROM SECOND ATOM, THEN ADD A BOND OF SOME TYPE BETWEEN THEM 
    # NOTE THAT WE HAVENT SETTLED AROMATIC YET 
    # TODO: AROMATIC ONES 

    actionTypes = [1, 2, 3, 1.5] # 1.5 is todo aromatic, but aoweusdjfkned 

    def __init__(self, first, second, type): 
        self.first = first 
        self.second = second 
        self.type = type 
        
    def __str__(self): 
        return '('+str(self.first)+','+str(self.second)+'): '+str(self.type)+' ' 
    
    def print(self): 
        stdout.write(str(self)+'\n') 
    
    def __eq__(self, other): 
        if (not (type(other) == Action)): 
            return NotImplemented 
        same_idxs = (self.first == other.first and self.second == other.second) or (self.first == other.second and self.second == other.first) 
        return (same_idxs and self.type == other.type) 
    
    def __gt__(self, other): 
        if (not (type(other) == Action)): 
            return NotImplemented 
        return 0 

    def __lt__(self, other): 
        if (not (type(other) == Action)): 
            return NotImplemented 
        return 0 


class NoValidActionException(Exception): 
    "Raised when there are no valid actions to take on a state" 
    pass 


class MolStateAI(): 

    embedding_size = 256 # 128 * 2 

    # set the loss functions 
    mseloss = nn.MSELoss() 
    #def rmseloss(pred, labels): 
    #    return torch.sqrt(MassSpec.mseloss(pred, labels)) 

    def __init__(self, training:bool, path, center_path, radius_path, center=None, radius=None,  gcn_lr=params.learning_rate, weight_decay=5e-4, nu=0.2, focal_gamma = 0.0, device=device): 
        self.device = device 

        self.nu = nu 
        self.weight_decay = weight_decay 
        self.gcn_lr = gcn_lr 

        self.training = training 
        self.path = path 

        self.focal_gamma = focal_gamma 

        #GCN 
        # set the parameters of the model 
        self.layersizes = [64, 128] # all hidden layers and output layer; output layer goes through pooling function so afuncs yes 
        self.afuncs = [nn.ReLU() for _ in range(len(self.layersizes)-1)] 
        self.afuncs.append(nn.ReLU()) 

        #self.conditioning_feats = self.layersizes[-1]*2 # since it has both weighted sum and max after pooling function 
        # not needed 

        if (not training): 
            # not training 
            # load model 
            self.model = GCN_edge_conditional(utils.atom_n_feats, 1, self.layersizes, self.afuncs, device=self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        
        else: 

            # main model 
        
            try: 
                # load model 
                self.model = GCN_edge_conditional(utils.atom_n_feats, 1, self.layersizes, self.afuncs, device=self.device)
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
            except Exception as e: # create model 
                print ("CREATING NEW MOLSTATE GCN EDGE CONDITIONAL MODEL")
                print(e) 
                self.model = GCN_edge_conditional(utils.atom_n_feats, 1, self.layersizes, self.afuncs, device=self.device) 
                if (self.path == None or len(self.path.strip()) == 0): self.path = "./RL_attempt/models/MolStateGCN_1.pt"

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.gcn_lr, weight_decay=self.weight_decay)

        
        self.center_path = center_path # NOTE: may be None 
        self.radius_path = radius_path 

        if not training: 
            if center == None: 
                assert center_path != None, "ERROR: MUST DEFINE CENTER PATH (OR CENTER)" 
                center = torch.load(center_path) 
            
            if radius == None: 
                assert radius_path != None, "ERROR: MUST DEFINE RADIUS PATH (OR RADIUS)" 
                with open(radius_path, 'r') as rfile: 
                    radius = float(rfile.readline()) 
                
        
        self.center = center 
        self.radius = radius 
    
    def save(self, path=None): 
        if path==None: path = self.path 
        torch.save(self.model.state_dict(), path) 
    
    def save_hypersphere_params(self, center_path=None, radius_path=None): 
        if center_path==None: 
            assert self.center_path != None, "ERROR: CENTER PATH NOT DEFINED" 
            center_path = self.center_path 
        
        if radius_path==None: 
            assert self.radius_path != None, "ERROR: CENTER PATH NOT DEFINED" 
            radius_path = self.radius_path 
        
        torch.save(self.center, center_path) 

        rfile = open(radius_path, 'w') 
        rfile.write(str(self.radius)) 
        rfile.write("\n") 
        rfile.close() 


    def resave(self, path=None, center_path=None, radius_path=None): 
        self.save(path) 
        self.save_hypersphere_params(center_path, radius_path)
        self.saved = True 
    
    '''
    def get_embedding(self, state): 
        return self.model(state.graph, state.graph.ndata["features"].float().to(self.device), state.graph.edata["bondtypes"].to(self.device))
    '''

    def get_embedding(self, ms:MassSpec, state, depth): # note that depth is an IntTensor with 1 element 
        #print(state.graph.edata["bondTypes"]) 
        return self.model(state.graph.to(self.device), state.graph.ndata["features"].float().to(self.device), state.graph.edata["bondTypes"].to(self.device), depth.to(self.device), ms.toTensor().to(self.device)) 
    
    def get_score(self, ms:MassSpec, state, depth): 
        #print("GETTING EMBEDDING")
        embedding = self.get_embedding(ms, state, depth) 
        #print("EMBEDDING GOTTEN, DOING PREDICTION")
        _, scores = self.anomaly_score(embedding) 
        return scores 
    
    def learn_loss_update_radius(self, loss, dist): 
        # allow multiple backward passes 
        self.optimizer.zero_grad() 
        
        loss.backward(retain_graph=True) 

        self.optimizer.step() 

        self.radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu) # this line is taken from OCGNN repository 

    def learn_loss(self, loss): 
        # allow multiple backward passes 
        self.optimizer.zero_grad() 
        
        loss.backward(retain_graph=True) 

        self.optimizer.step() 
    
    # the following 3 functions are adapted from OCGNN github repository; arXiv:2002.09594 

    def init_center(self, gs, nfeats, efeats, timesteps, conditioning_feats, eps=0.001):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_hidden = self.layersizes[-1]*2 
        n_samples = 0
        c = torch.zeros(n_hidden, device=self.device)

        self.model.eval()
        with torch.no_grad():

            # for each 
            all_outputs = [] 
            for gidx in range(len(gs)): 
                outputs = self.model(gs[gidx], nfeats[gidx], efeats[gidx], timesteps[gidx], conditioning_feats[gidx])
                all_outputs.append(outputs) 
            
            all_outputs = torch.stack(all_outputs).to(device)  

            # get the inputs of the batch

            n_samples = all_outputs.shape[0] 
            c =torch.sum(all_outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def loss_function(self, outputs, target=None, mask=None): # target is a tensor with one element 
        if target==None: target=torch.Tensor([0]).to(self.device) 
        dist,scores = self.anomaly_score(outputs, mask)
        base_score_loss = torch.mean(torch.max(target.repeat(scores.shape), scores)) #MolStateAI.mseloss(target.repeat(scores.shape), scores) #(torch.mean(torch.abs(target.repeat(scores.shape) - scores))) 
        loss = self.radius ** 2 + ( (base_score_loss)**self.focal_gamma ) * (1 / self.nu) * base_score_loss # base_score_loss**self.focal_gamma, so that low loss gives low value, high loss gives higher value, making it focal. 
        return loss, dist, scores

    def anomaly_score(self, outputs, mask=None):
        if mask == None:
            dist = torch.sum((outputs - self.center) ** 2, dim=1)
        else:
            dist = torch.sum((outputs[mask] - self.center) ** 2, dim=1) 

        scores = nn.Sigmoid()(dist - self.radius ** 2) # sigmoid to squish score between 0 and 1 
        return dist,scores # higher score means more anomalous 


class MolState(): # represents a state of the molecule 

    def __init__(self, smiles, atom_counts=None, graph=None, init_type=0, device=device): # TODO: DO MORE WITH init_type 
        # NOTE: init_type tells you what rings you start with 
        self.smiles = smiles # can be None if atom_counts and graph are specified 
        
        # NOTE: self.atom_counts DOES NOT INCLUDE H 
        if (atom_counts==None): self.atom_counts = utils.smiles_to_atom_counts(smiles) 
        else: self.atom_counts = atom_counts 

        if (graph==None): self.graph = utils.SMILEStoGraphType(smiles, init_type) 
        else: self.graph = graph 

        self.atom_counts[1] = self.get_H_count() 

        self.device = device 
    
    def save(self, idx, save_path): 
        graph_labels = {"atom_counts": torch.IntTensor([self.atom_counts]), "idx": torch.IntTensor([idx])} 
        save_graphs(save_path, [self.graph], graph_labels) 
    
    def save_states(states, idxs, save_path): # .bin file 
        atom_countss = [] 
        graphs = [] 
        for s in states: 
            atom_countss.append(s.atom_counts) 
            graphs.append(s.graph.cpu())
        
        graph_labels = {"atom_counts": torch.IntTensor(atom_countss), "idx":torch.IntTensor(idxs)} 
        save_graphs(save_path, graphs, graph_labels)
    
    def load_states(path, smiles=None, smiless=None, device=device): # options to either have list of smiless or single smiles for all of them, or none, which is None for SMILES 
        graphs, labels = load_graphs(path) 
        atom_countss = labels['atom_counts'] 

        states = [] 
        for gidx in range(len(graphs)): 
            if (smiless != None): smiles=smiless[gidx] 
            states.append(MolState(smiles, atom_countss[gidx], graphs[gidx], device=device))
        
        return labels['idx'].tolist(), states 


    def take_action(self, action:Action): # function to get new state if an action would be taken from a state; NOTE THAT THIS DOES NOT CHANGE THE STATE ITSELF 

        #print("TAKE ACTION GRAPH:", self.graph) 

        # make copy of state to use 
        state = MolState(self.smiles, self.atom_counts, copy.deepcopy(self.graph), self.device) 
        
        required = action.type+1 # amount of Hs to be removed 
        # remove those Hs 
        state.graph.ndata['features'][action.first][1] -= required 
        state.graph.ndata['features'][action.first][0] += 1 

        state.graph.ndata['features'][action.second][1] -= required 
        state.graph.ndata['features'][action.second][0] += 1 

        # add edge 
        atomnum1 = utils.get_atomic_num_from_atom_feats(self.graph.ndata['features'][action.first])
        atomnum2 = utils.get_atomic_num_from_atom_feats(self.graph.ndata['features'][action.second])
        bond_energy = utils.getBEfromNum(atomnum1, atomnum2, action.type) 
        bondFeats = [bond_energy] 
        for bidx in range(len(utils.bondTypes)): 
            if bidx==action.type: bondFeats.append(1) 
            else: bondFeats.append(0) 
        #print("BONDFEATS:", bondFeats)
        state.graph.add_edges(torch.IntTensor([action.first, action.second]).to(self.device), torch.IntTensor([action.second, action.first]).to(self.device), data={"bondTypes": torch.tensor([bondFeats, bondFeats], device=self.device)}) 

        return state 
    
    
    def undo_action(self, action:Action): # function to get previous state if an action was undone from a state; NOTE THAT THIS DOES NOT CHANGE THE STATE ITSELF 

        #print("TAKE ACTION GRAPH:", self.graph) 

        # make copy of state to use 
        state = MolState(self.smiles, copy.deepcopy(self.atom_counts), copy.deepcopy(self.graph), self.device) 
        
        required = action.type+1 # amount of Hs to be added back 
        # add back those Hs 
        state.graph.ndata['features'][action.first][1] += required 
        state.graph.ndata['features'][action.first][0] -= 1 

        state.graph.ndata['features'][action.second][1] += required 
        state.graph.ndata['features'][action.second][0] -= 1 

        # find edge 
        edges_to_remove = [] 
        edge_list = list(state.graph.edges()) 
        for e in range(len(edge_list[0])): 
            if (edge_list[0][e] == action.first and edge_list[1][e] == action.second) or (edge_list[0][e] == action.second and edge_list[1][e] == action.first): 
                edges_to_remove.append(e) 
        
        # remove edge 
        state.graph.remove_edges(edges_to_remove) 

        return state 
    
    
    def get_valid_unactions(self): # just enumerating all bonds mm 
        edge_list = list(self.graph.edges()) 
        actions_to_take = set() # set of tuples of action: (first, second, btype) 
        for eidx in range(len(edge_list[0])): 
            # search for 1 in edge feature to get bond type 
            idx = 1 
            while idx < len(self.graph.edata['bondTypes'][eidx]): 
                if self.graph.edata['bondTypes'][eidx, idx].item() == 1: 
                    break 
                idx += 1 

            etype = idx - 1 
            if etype != 3: # make sure not aromatic bond 
                actions_to_take.add((edge_list[0][eidx], edge_list[1][eidx], etype)) 
        
        actions = [] 
        for action_tuple in list(actions_to_take): 
            if action_tuple[0] > action_tuple[1]: 
                #actions.append(Action(action_tuple[1], action_tuple[0], action_tuple[2])) 
                pass # prevent duplicates, since both are added. 
            else: 
                actions.append(Action(action_tuple[0], action_tuple[1], action_tuple[2])) 
            
        return actions 
    

    # for debugging 
    def is_valid_action(self, ufds, i, j, bond_order, extra_Hs, consider_connectivity=True): 
        # NOTE: bond_order = 1+btype 
        if (consider_connectivity): 
            ufds = utils.get_ufds(self.graph) # Union-Find Disjoint Set data structure 
            num_comps_start = ufds.get_num_disjoint_sets() 
            
            # get num Hs in each comp 
            comps = [] # root of each component 
            comp_num_Hs = [] # number of Hs in them 
            ii = 0 
            while (len(comps) < num_comps_start): 
                t = ufds.find_set(ii) 
                if (t not in comps): 
                    comps.append(t) 
                    comp_num_Hs.append(ufds.get_set_Hs(t)) 
                ii += 1 

        rem_Hs = self.graph.ndata['features'][:,1] 


        # calculate connectivity limitation - connectivity is only possible if num_components_after_action-1 <= remaining bonds = initial left - bond order 
        # --> bond order <= initial left - num_components_after_action + 1 
        # (initial left = round(extra_Hs)//2 ) 
        connectivity_limitation = round(extra_Hs)//2 
        print("CL:",connectivity_limitation)
        
        bond_order_limitation = 3 
        
        if (consider_connectivity): 
            # consider able to connect or not, ignoring num Hs in node feats 
            x = ufds.find_set(i) 
            y = ufds.find_set(j) 
            if x==y: num_comps_after = num_comps_start 
            else: num_comps_after = num_comps_start-1 # if from diff sets, after will -1 comps 
            connectivity_limitation -= num_comps_after 
            connectivity_limitation += 1 
            print("INITIAL CONNECTIVITY LIMITATION:",connectivity_limitation)
            print("i j:",i,j)
            print("x y:",x,y)
            
            # consider that num Hs in all components have to be >=1 and total must be >= 2N-2 after action 
            # >= 1 is considered in loop 
            # total - 2*bond_order >= 2*num_comps_after - 2 
            # rearrange: 2*bond_order <= total - 2*num_comps_after + 2 
            # bond_order <= total/2 - num_comps_after + 1 
            passable = True 
            sel_comps = [] # num of Hs in selected components 
            total = 0 
            for cidx in range(len(comps)): 
                t = comp_num_Hs[cidx] 
                if comps[cidx]==x or comps[cidx]==y: 
                    sel_comps.append(t) # settle >= 1 later 
                else: 
                    if t < 1: 
                        passable = False 
                        break 
                total += t 
            
            print(passable, total)
            
            if passable: 
                # check >= 1 
                if x!=y: 
                    # x != y, going to join two different components to become the sae 
                    if ( (sel_comps[0]-bond_order + sel_comps[1]-bond_order) < 1 ): 
                        print(505) 
                        # this bond order cannot 
                        return False 
                else: 
                    # x == y, sel_comps has only one element 
                    if (sel_comps[0] - 2*bond_order) < 1: 
                        print(511) 
                        # this bond order cannot 
                        return False 
                
                # check total 
                if (bond_order <= total/2 - num_comps_after + 1): 
                    passable = True 
                
                if passable and bond_order > 0: 
                    bond_order_limitation = bond_order 
                else: 
                    print(522) 
                    return False 
            else: 
                print(527)
                return False 

            print("BOND ORDER LIMITATION:", bond_order_limitation) 
            
            
        return ( bond_order-1 < min(int(rem_Hs[i].item()), int(rem_Hs[j].item()), connectivity_limitation, bond_order_limitation) ) 


    def get_valid_actions(self, extra_Hs, actions_tried_alr:list, consider_connectivity=True): # NOTE: DOES NOT CHECK DUPLICATES OR MULTIPLE BONDS BETWEEN SAME NODES 
        actions = [] 
        
        if (consider_connectivity): 
            ufds = utils.get_ufds(self.graph) # Union-Find Disjoint Set data structure 
            num_comps_start = ufds.get_num_disjoint_sets() 
            
            # get num Hs in each comp 
            comps = [] # root of each component 
            comp_num_Hs = [] # number of Hs in them 
            i = 0 
            while (len(comps) < num_comps_start): 
                t = ufds.find_set(i) 
                if (t not in comps): 
                    comps.append(t) 
                    comp_num_Hs.append(ufds.get_set_Hs(t)) 
                i += 1 

        rem_Hs = self.graph.ndata['features'][:,1] 
        for i in range(len(rem_Hs)-1): 
            if rem_Hs[i] == 0: continue 
            for j in range(i+1, len(rem_Hs)): 
                
                if i==j: continue 
                if rem_Hs[j] == 0: continue 
                # calculate connectivity limitation - connectivity is only possible if num_components_after_action-1 <= remaining bonds = initial left - bond order 
                # --> bond order <= initial left - num_components_after_action + 1 
                # (initial left = round(extra_Hs)//2 ) 
                connectivity_limitation = round(extra_Hs)//2 
                
                bond_order_limitation = 3 
                
                if (consider_connectivity): 
                    # consider able to connect or not, ignoring num Hs in node feats 
                    x = ufds.find_set(i) 
                    y = ufds.find_set(j) 
                    if x==y: num_comps_after = num_comps_start 
                    else: num_comps_after = num_comps_start-1 # if from diff sets, after will -1 comps 
                    connectivity_limitation -= num_comps_after 
                    connectivity_limitation += 1 
                    
                    # consider that num Hs in all components have to be >=1 and total must be >= 2N-2 after action 
                    # >= 1 is considered in loop 
                    # total - 2*bond_order >= 2*num_comps_after - 2 
                    # rearrange: 2*bond_order <= total - 2*num_comps_after + 2 
                    # bond_order <= total/2 - num_comps_after + 1 
                    passable = True 
                    sel_comps = [] # num of Hs in selected components 
                    total = 0 
                    for cidx in range(len(comps)): 
                        t = comp_num_Hs[cidx] 
                        if comps[cidx]==x or comps[cidx]==y: 
                            sel_comps.append(t) # settle >= 1 later 
                        else: 
                            if t < 1: 
                                passable = False 
                                break 
                        total += t 
                    
                    if passable: 
                        for bond_order in range(3,-1,-1): 
                            # check >= 1 
                            if x!=y: 
                                # x != y, going to join two different components to become the sae 
                                if ( (sel_comps[0]-bond_order + sel_comps[1]-bond_order) < 1 ): 
                                    # this bond order cannot 
                                    continue 
                            else: 
                                # x == y, sel_comps has only one element 
                                if (sel_comps[0] - 2*bond_order) < 1: 
                                    # this bond order cannot 
                                    continue 
                            
                            # check total 
                            if (bond_order <= total/2 - num_comps_after + 1): 
                                passable = True 
                                break 
                        
                        if passable and bond_order > 0: 
                            bond_order_limitation = bond_order 
                        else: 
                            #bond_order_limitation = -1 
                            #raise NoValidActionException 
                            continue 
                    else: 
                        #bond_order_limitation = -1 
                        # cannot take any actions 
                        #raise NoValidActionException 
                        continue 
                    
                    
                #print("ACTION:",i,j) 
                #print("CONNECTIVITY LIMITATION:", connectivity_limitation) 
                
                # NOTE: duplicates are not removed here 
                #print(rem_Hs[i].item(), rem_Hs[j].item()) 
                for btype in range(min(int(rem_Hs[i].item()), int(rem_Hs[j].item()), connectivity_limitation, bond_order_limitation)): 
                    #print("BONDTYPE:", btype) 
                    
                    # make sure action is not already taken 
                    already_taken = False 
                    for action in actions_tried_alr: 
                        if (action.first==i and action.second==j) or (action.first==j and action.second==i): 
                            if (action.type == btype): 
                                already_taken = True 
                                break 
                    
                    if (not already_taken): actions.append(Action(i, j, btype))
                    
                    
        
        if len(actions)==0: raise NoValidActionException 
        
        return actions 


    def get_next_states(self, extra_Hs, actions_tried_alr:list, return_actions=False): 

        '''
        print()
        print()
        print() 
        print("STATE NEXT STATES") 
        print() 
        print() 
        '''

        raw_actions = self.get_valid_actions(extra_Hs, actions_tried_alr) 

        #print("STATES GOT NEXT ACTIONS ALREADY")
        # prevent multiple bonds between same atoms 

        actions = [] 
        edge_list = list(self.graph.edges()) 
        for action in raw_actions: 
            valid = True 
            for eidx in range(len(edge_list[0])): 
                if ((action.first==edge_list[0][eidx].item()) and (action.second == edge_list[1][eidx].item())) or ((action.first==edge_list[0][eidx].item()) and (action.second == edge_list[1][eidx].item())): 
                    # action cannot, since already has bonds 
                    valid = False 
                    break 
            if valid: actions.append(action) 
        
        #print("PREVENT MULTIPLE BONDS ALREADY") 
        # check for different actions leading to same resulting state 

        new_states = [] 
        nx_graphs = [] 
        if return_actions: returned_actions = [] 

        for a in actions: 
            new_state = self.take_action(a) 

            #print("ACTION:") 
            #print(a.first, a.second, a.type) 
            #new_state.show_visualization() 

            # check for duplicates 
            repeat = False 
            g = utils.dgl_to_networkx_isomorphism(new_state.graph.cpu()) 
            for i in range(len(new_states)): 
                if utils.may_be_isomorphic(new_states[i].graph.cpu(), new_state.graph.cpu()): # if may be same graph 
                    if utils.is_isomorphic(new_states[i].graph.cpu(), new_state.graph.cpu(), nx_graphs[i], g): # check fully 
                        # if indeed same graph 
                        repeat = True 
                        #print("REPEATED, REMOVED") 
                        break 
                    # else, just go on 

            if repeat: continue 

            new_states.append(new_state) # if not duplicate 
            nx_graphs.append(g) 
            if (return_actions): returned_actions.append(a) 

        #print() 
        #print() 
        #print() 
        #print("FINISHED REMOVING DUPLICATES ALREADY")

        if return_actions: return (new_states, returned_actions) 
        return new_states 

    def get_H_count(self): 
        '''h_count = 0 
        for nodefeat in self.graph.ndata['features']: 
            h_count += nodefeat[1].item() 
        return h_count '''
        return self.graph.ndata['features'][:,1].sum().item()

    # for debugging 
    def check_degrees(self): 
        # check atom degrees + valence, to the features 
        issue = False 

        for i in range(self.graph.num_nodes): 
            electrons = Chem.GetPeriodicTable().GetNOuterElecs(self.graph.ndata['features'][i][0]) 
            electrons += self.graph.ndata['features'][i][2] 

            edge_list = list(self.graph.edges())
            for e_idx in range(len(edge_list)): 
                if edge_list[0][e_idx].item() != i and edge_list[1][e_idx].item() != i: continue # does not involve this node 
                n_electrons = self.graph.edata['bondTypes'][e_idx] 
                if n_electrons==4: n_electrons=1.5 # if aromatic, it's 1.5 
                electrons += n_electrons 
            electrons = round(electrons) 
            print(electrons, end=" ") 
            if electrons != 8: 
                issue = True 
        print() 
        return issue 
    
    # visualization properties
    #vis_xdist = 0.05 
    #vis_ydist = 0.05 
    vis_k = 0.3 

    def show_visualization(self, title=None, pos=None, draw_edge_labels=False, block=True): 
        plt.figure() 

        graph = self.graph.cpu() 

        # prepare 
        g = dgl.to_networkx(graph) 

        #stdout.write("Generating graph... \n")

        if pos==None: pos = nx.spring_layout(g, k=MolState.vis_k, iterations=20) 

        #stdout.write("Drawing graph... \n")

        # draw each kind of node 
        node_options = {"node_size": 400, "node_shape": 'o'} 

        labels = {} 

        for nodeType in range(len(utils.FTreeNode.atomTypes)): 
            nt = nodeType 
            if nt==1: continue # Hydrogen 
            if nt>1: nt -= 1 

            nodes = [] 
            for nidx in range(len(graph.nodes())): 
                if graph.ndata['features'][nidx][nt+2].item() == 1: 
                    nodes.append(nidx) 
                    labels[nidx] = utils.FTreeNode.atomTypes[nodeType] 

            #print(nodeType, nt)
            #print("ATOM TYPE:", utils.FTreeNode.atomTypes[nodeType]) 
            #print("COLOUR:", utils.FTreeNode.atomColours[nodeType]) 
            nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=utils.FTreeNode.atomColours[nodeType], **node_options) 

        # draw each kind of edge 

        edge_options = {"alpha": 0.7} 
        graph_edge_list = list(graph.edges()) 
        #print(graph_edge_list) 
        for edgeType in range(len(utils.bondTypes)): 
            edges = [] 
            for eidx in range(len(graph_edge_list[0])): 
                #print(graph.edata['bondTypes'][eidx][edgeType+1].item(), end=' ')
                if graph.edata['bondTypes'][eidx][edgeType+1].item() == 1: 
                    edges.append((graph_edge_list[0][eidx].item(), graph_edge_list[1][eidx].item())) 
            #print() 
            #print(edgeType, ":", edges)
            nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=utils.edgeColors[edgeType], **edge_options)

        nx.draw_networkx_labels(g, pos)

        if draw_edge_labels: # TODO: THIS IS VERY WRONG 
            edge_labels = {} 
            i = 0 
            for e in g.edges(): 
                edge_labels[e] = str(i//2) # THIS IS WRONG, E.G. BENZENE RING 
                i += 1 
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, alpha=0.5)

        if title != None: 
            plt.title(title) 

        plt.show(block=block) 
    
    def __eq__(self, other): 
        if (not (type(other) == MolState)): 
            return NotImplemented 
        return 0 
    
    def __gt__(self, other): 
        if (not (type(other) == MolState)): 
            return NotImplemented 
        return 0 

    def __lt__(self, other): 
        if (not (type(other) == MolState)): 
            return NotImplemented 
        return 0 


            




