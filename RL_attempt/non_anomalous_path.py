# concept: take inspiration from anomaly detection, since it generally has only non-anomalous training data. We do this to save time, since the state space in our thing is very very large. 
# goal: train the GCN to make an n-dimensional representation of the graph, maximizing the proportion of the non-anomalous data with embeddings contained within it, while minimizing the radius. 
# finally: GCN considers 

import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl 
import torch 
import torch.nn as nn 
from rdkit import Chem 
from rdkit.Chem.rdMolDescriptors import CalcMolFormula 

from agent import * 
from environment import * 
from representations import * 

import dataloader 
import utils 
import params 

import copy 
import random 
import queue 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
print(device)

mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_loss_top_30/search_0.0001_0.0001/models/mass_spec_training/FTreeGCN_training_epoch_50.pt'
path_prefix = './RL_attempt/non_anomalous_path_training'

gcn_lr = 0.00004 
weight_decay = 5e-4 
nu = 0.2 
discount_factor = 0.9 

filter_away_not_0_1 = True 
try_memo = True 
max_num_heavy_atoms = 12 


# testing 
test_smiless, test_ftrees, test_peakslist = dataloader.get_data('test') 

# preprocess some data 
test_filtered_smiless = [] 
test_filtered_ftrees = [] 
test_graphs = [] 

for idx in range(len(test_smiless)): 
    # get graph of target molecule to make MolState 
    target_graph = utils.SMILEStoGraph(test_smiless[idx]).to(device) 

    # filter away molecules that are too large 
    if sum(utils.smiles_to_atom_counts(test_smiless[idx], include_H=False)) > max_num_heavy_atoms: 
        continue 

    if (filter_away_not_0_1): 
        # make sure that we are only learning those with benzene or no benzene but not other aromatic structures 
        aromatic_bonds = [] 
        for e in range(len(target_graph.edata['bondTypes'])//2): 
            if target_graph.edata['bondTypes'][2*e,4] == 1: 
                aromatic_bonds.append(e) 
        
        if len(aromatic_bonds) != 0 and len(aromatic_bonds) != 6: continue # don't try this 
        if len(aromatic_bonds) == 6: 
            idxs = set() 
            edge_list = list(target_graph.edges()) 
            for e in aromatic_bonds: 
                idxs.add(edge_list[0][e].item()) 
                idxs.add(edge_list[1][e].item()) 
            
            idxs = list(idxs) 
            if len(idxs) != 6: continue # means not benzene ring 

            # check that all have degree 2 
            degs = [0 for _ in range(6)] 
            for e in aromatic_bonds: 
                degs[idxs.index(edge_list[0][e].item())] += 1 
                degs[idxs.index(edge_list[1][e].item())] += 1 
            
            skip = False 
            for d in degs: 
                if d != 2: 
                    skip = True 
                    break 
            
            if skip: continue 
    
    mass_spec = MassSpec(False, test_ftrees[idx], mass_spec_gcn_path, None, device=device) 
    init_formula = CalcMolFormula(Chem.MolFromSmiles(test_smiless[idx]))  
    target_state = MolState(test_smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 
    
    rem_Hs, fatal_errors = EnvMolecule.compare_state_to_mass_spec(target_state, mass_spec, True) 
    if len(fatal_errors) > 0: 
        continue 

    # now, verified to be okay :) 

    #num_bonds = len(list(target_graph.edges()))//2 
    num_bonds = len(target_state.get_valid_unactions()) 

    test_filtered_smiless.append(test_smiless[idx]) 
    test_filtered_ftrees.append(test_ftrees[idx]) 
    #print(target_graph.device, mass_spec.pred.device)
    test_graphs.append(target_graph) 



# preprocess the states 

test_memo_statess = [None for _ in range(len(test_filtered_smiless))] # memory of the states - can be reinitialized only here since it's useless to redo it every new parameter search 
bond_actionss = [None for _ in range(len(test_filtered_smiless))] 
test_num_bondss = [] 

for idx in range(len(test_filtered_smiless)): 
    # get graph of target molecule to make MolState 
    target_graph = test_graphs[idx] #utils.SMILEStoGraph(filtered_smiless[idx]) 


    init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[idx]))  
    target_state = MolState(test_filtered_smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 

    total_num_Hs = utils.formula_to_atom_counts(init_formula, True)[1] 

    #num_bonds = len(list(target_graph.edges())[0])//2 
    bond_actions = target_state.get_valid_unactions() # gets all bonds, or rather, Actions corresponding to them 
    num_bonds = len(bond_actions) 

    def state_generator(): 
        while True: 
            for bitmask in range(1, 1<<num_bonds): 
                state = copy.deepcopy(target_state) 

                b = bitmask 
                i = 0 
                depth = num_bonds 
                while b > 0: 
                    if b%2 == 0: 
                        state = state.undo_action(bond_actions[i]) 
                        depth -= 1 
                    i += 1 
                    b /= 2 
                
                yield state, depth 
    
    test_memo_statess[idx] = state_generator() 
    bond_actionss[idx] = bond_actions 
    test_num_bondss.append(num_bonds) 

''' NOTE: WILL SAMPLE ANOMALOUS STATES IN TEST FUNCTION ITSELF 

        # try adding extra 
        temp = state.get_next_states(total_num_Hs - state.get_H_count(), bond_actions) 
        if len(temp)>0: 
            test_anomalous_states.append(temp[random.randrange(len(temp))]) 
    
    test_memo_statess[idx] = test_memo_states 
'''

# save the sampled anomalous states, for reference, probably. 



def test(epoch, save_path:str): # save states with labels of what they are in a .bin file - not saving anymore, to save memory 
    with torch.no_grad(): # to save memory 
        # test function 
        state_ai = MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 

        # test scores 
        normal_scores = [] 
        anomalous_scores = [] 

        
        for idx in range(len(test_filtered_smiless)): 
            # get graph of target molecule to make MolState 
            target_graph = test_graphs[idx] #utils.SMILEStoGraph(filtered_smiless[idx]) 


            mass_spec = MassSpec(False, test_filtered_ftrees[idx], mass_spec_gcn_path, None, device=device) 
            init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[idx]))  
            target_state = MolState(test_filtered_smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 
            am = AgentMolecule(mass_spec, target_state, state_ai) 


            #print("EPOCH",epoch,"ITER",idx) 
            #print("SMILES:",filtered_smiless[idx]) 
            #print("FORMULA:", init_formula) 
            #target_state.show_visualization(block=False) 
            #print("TARGET STATE VALID UNACTIONS:", len(target_state.get_valid_unactions())) 
            #for ua in target_state.get_valid_unactions(): 
            #    print(ua) 

            #anomalous_states_chosen = [] 
            #anomalous_idxs = [] 

            #anom_idx = 0 
            for _ in range(1, 1<<test_num_bondss[idx]): 
                state, depth = next(test_memo_statess[idx]) 
                loss, dist, score = am.state_ai.loss_function(am.state_ai.get_embedding(am.mass_spec, state, torch.tensor([depth], device=device))) 
                normal_scores.append(score.item()) 

                # try testing anomalous 
                try: 
                    #print(mass_spec.atom_counts[1], state.get_H_count(), bond_actionss[idx])
                    temp = state.get_next_states(mass_spec.atom_counts[1] - state.get_H_count(), bond_actionss[idx]) 
                    #print(len(temp)) 
                    if (len(temp) > 0): 
                        i = random.randrange(len(temp))
                        _, _, score = am.state_ai.loss_function(am.state_ai.get_embedding(mass_spec, temp[i], torch.tensor([depth], device=device)))
                        anomalous_scores.append(score.item())
                        #anomalous_states_chosen.append(temp[i]) 
                        #anomalous_idxs.append(anom_idx) # save the index of the state that has more actions that can be taken 
                        #print(len(anomalous_states_chosen)) 

                except NoValidActionException: 
                    pass 
                    
                #anom_idx += 1 

        #print(anomalous_states_chosen) 

        #MolState.save_states(anomalous_states_chosen, anomalous_idxs, save_path) 
        
        print("EPOCH", epoch, "TEST AVG NORMAL SCORE:", sum(normal_scores)/len(normal_scores), "STDDEV:", np.std(normal_scores)) 
        print("EPOCH", epoch, "TEST AVG ANOMALOUS SCORE:", sum(anomalous_scores)/len(anomalous_scores), "STDDEV:", np.std(anomalous_scores)) 
        return normal_scores, anomalous_scores 







num_epochs = 100 
test_epoch_interval = 5 



# training 
smiless, ftrees, peakslist = dataloader.get_data('train') 

# preprocess some data 
filtered_smiless = [] 
filtered_ftrees = [] 
graphs = [] 
node_feats = [] 
edge_feats = [] 
timesteps = [] 
conditioning_feats = [] 

for idx in range(len(smiless)): 
    # get graph of target molecule to make MolState 
    target_graph = utils.SMILEStoGraph(smiless[idx]).to(device) 

    # filter away molecules that are too large 
    if sum(utils.smiles_to_atom_counts(smiless[idx], include_H=False)) > max_num_heavy_atoms: 
        continue 

    if (filter_away_not_0_1): 
        # make sure that we are only learning those with benzene or no benzene but not other aromatic structures 
        aromatic_bonds = [] 
        for e in range(len(target_graph.edata['bondTypes'])//2): 
            if target_graph.edata['bondTypes'][2*e,4] == 1: 
                aromatic_bonds.append(e) 
        
        if len(aromatic_bonds) != 0 and len(aromatic_bonds) != 6: continue # don't try this 
        if len(aromatic_bonds) == 6: 
            idxs = set() 
            edge_list = list(target_graph.edges()) 
            for e in aromatic_bonds: 
                idxs.add(edge_list[0][e].item()) 
                idxs.add(edge_list[1][e].item()) 
            
            idxs = list(idxs) 
            if len(idxs) != 6: continue # means not benzene ring 

            # check that all have degree 2 
            degs = [0 for _ in range(6)] 
            for e in aromatic_bonds: 
                degs[idxs.index(edge_list[0][e].item())] += 1 
                degs[idxs.index(edge_list[1][e].item())] += 1 
            
            skip = False 
            for d in degs: 
                if d != 2: 
                    skip = True 
                    break 
            
            if skip: continue 
    
    mass_spec = MassSpec(False, ftrees[idx], mass_spec_gcn_path, None, device=device) 
    init_formula = CalcMolFormula(Chem.MolFromSmiles(smiless[idx]))  
    target_state = MolState(smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 
    
    rem_Hs, fatal_errors = EnvMolecule.compare_state_to_mass_spec(target_state, mass_spec, True) 
    if len(fatal_errors) > 0: 
        continue 

    # now, verified to be okay :) 

    num_bonds = len(target_state.get_valid_unactions()) 
    #print(num_bonds) 

    filtered_smiless.append(smiless[idx]) 
    filtered_ftrees.append(ftrees[idx]) 
    #print(target_graph.device, mass_spec.pred.device)
    graphs.append(target_graph) 
    node_feats.append(target_graph.ndata['features']) 
    edge_feats.append(target_graph.edata['bondTypes']) 
    timesteps.append(torch.Tensor([num_bonds]).to(device)) 
    conditioning_feats.append(mass_spec.pred) 


# preprocess the states 

memo_statess = [None for _ in range(len(filtered_smiless))] # memory of the states - can be reinitialized only here since it's useless to redo it every new parameter search 
num_bondss = [] 

for idx in range(len(filtered_smiless)): 
    # get graph of target molecule to make MolState 
    target_graph = graphs[idx] #utils.SMILEStoGraph(filtered_smiless[idx]) 


    init_formula = CalcMolFormula(Chem.MolFromSmiles(filtered_smiless[idx]))  
    target_state = MolState(filtered_smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 

    #num_bonds = len(list(target_graph.edges())[0])//2 
    bond_actions = target_state.get_valid_unactions() 
    num_bonds = len(bond_actions) 

    def state_generator(): 
        while True: # because we will be doing multiple epochs 
            for bitmask in range(1, 1<<num_bonds): 
                
                state = copy.deepcopy(target_state) 

                b = bitmask 
                i = 0 
                depth = num_bonds 
                while b > 0: 
                    if b%2 == 0: 
                        state = state.undo_action(bond_actions[i]) 
                        depth -= 1 
                    i += 1 
                    b /= 2 
                
                yield state, depth # yielded depth is integer 
    
    memo_statess[idx] = state_generator() 
    
    num_bondss.append(num_bonds)


print("NUMBER OF TRAINING ITERS:",len(filtered_smiless)) 
#print(num_bondss) 


def train(start_epoch=0): 

    center_path = path_prefix+"/center_epoch_"+str(start_epoch)+".pt"
    radius_path = path_prefix+"/radius_epoch_"+str(start_epoch)+".txt" 

    if start_epoch != 0: 
        center = torch.load(center_path) 
        with open(radius_path, 'r') as rfile: 
            radius = float(rfile.readline()) 

        state_ai = MolStateAI(True, path_prefix+'/MolStateGCN_epoch_'+str(start_epoch)+'.pt', center_path , radius_path, center, radius, gcn_lr=gcn_lr, weight_decay=weight_decay, nu=nu, device=device) 

    else: 
        center = torch.tensor([], device=device) 
        radius = 0 

        state_ai = MolStateAI(True, path_prefix+'/MolStateGCN_epoch_'+str(start_epoch)+'.pt', center_path, radius_path, center, radius, gcn_lr=gcn_lr, weight_decay=weight_decay, nu=nu, device=device) 
            
        state_ai.center = state_ai.init_center(graphs, node_feats, edge_feats, timesteps, conditioning_feats) 

    #issue_count = 0 

    ress = [] 
    
    train_losses = [] 

    for epoch in range(start_epoch+1, 1+num_epochs): # epoch number is model at start of epoch 

        #print('epoch',epoch,'start')

        # learn loss 
        total_loss = 0 
        num_losses = 0 
        
        for idx in range(len(filtered_smiless)): 
            # get graph of target molecule to make MolState 
            target_graph = graphs[idx] #utils.SMILEStoGraph(filtered_smiless[idx]) 


            mass_spec = MassSpec(False, filtered_ftrees[idx], mass_spec_gcn_path, None, device=device) 
            init_formula = CalcMolFormula(Chem.MolFromSmiles(filtered_smiless[idx]))  
            target_state = MolState(filtered_smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 
            am = AgentMolecule(mass_spec, target_state, state_ai) 


            #print("EPOCH",epoch,"ITER",idx) 
            #print("SMILES:",filtered_smiless[idx]) 
            #print("FORMULA:", init_formula) 
            #target_state.show_visualization(block=False) 
            #print("TARGET STATE VALID UNACTIONS:", len(target_state.get_valid_unactions())) 
            #for ua in target_state.get_valid_unactions(): 
            #    print(ua) 


            for _ in range(1, 1 << num_bondss[idx]): 
                state, depth = next(memo_statess[idx]) # here, no unaction is taken, so it's just depth 
                target = 1 - (discount_factor ** depth) 
                loss, dist, _ = am.state_ai.loss_function(am.state_ai.get_embedding(am.mass_spec, state, torch.tensor([depth], device=device)), torch.tensor([target], device=device)) 
                total_loss += loss.item() 
                am.state_ai.learn_loss_update_radius(loss, dist) 
                num_losses += 1 
            
            #input() 
                
            # extra print message to be more certain that this is working 
            #print("ITER",idx,"DONE") 


        '''
        for i in range(len(losses_to_learn)): 
            total_loss += losses_to_learn[i].item() 
            am.state_ai.learn_loss_update_radius(losses_to_learn[i], dists[i]) 
            num_losses += 1 
        ''' 
        
        train_losses.append(total_loss / num_losses)
            
        print("EPOCH", epoch, "TRAIN LOSS:", train_losses[-1])

        
        
        if epoch%test_epoch_interval == 0: 
            am.state_ai.save(path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt') 
            am.state_ai.save_hypersphere_params(path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt") 
            ress.append(test(epoch, path_prefix+"/test_states_epoch_"+str(epoch)+".bin"))

    fout = open(path_prefix+"/test_scores.txt", 'a+') 
    for res in ress: 
        fout.write(str(res[0])) # normal 
        fout.write("\n") 
        fout.write(str(res[1])) # anomalous 
        fout.write("\n\n")
    fout.close()  

    
    fout = open(path_prefix+"/train_losses.txt", 'a+') 
    for res in train_losses: 
        fout.write(str(res)) 
        fout.write("\n") 
    fout.close()


