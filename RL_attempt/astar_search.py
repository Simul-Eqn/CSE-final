# ASTAR search algorithm combined with beam search 

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

mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_without_pooling/search_3e-07_1e-06/models/mass_spec_training/FTreeGCN_training_epoch_20.pt'  # an arbritrary path, as this wil only be used to prevent errors 
path_prefix = './RL_attempt/astar_search/' 

#gcn_lr = 0.00004 
#predictor_lr = 0.00004
#weight_decay = 5e-4 
#nu = 0.2 


num_top_guesses = 10 
num_guesses_per_state = 2 # -1 means all 
beam_width = 100 
types_considered = [0,1] # no benzene, 1 benzene - TODO: make the aromatic bonds thing check based on this too perhaps??? 


# placeholders to be set in init 
filter_away_not_0_1 = True 
test_filtered_smiless = [] 
test_filtered_ftrees = [] 
test_start_types = [] 
valid_count = 0 


def init(valid_fraction:float, 
         max_num_heavy_atoms:int, 
         filter_away_not_0_1_local:bool): 
    
    global test_filtered_smiless, test_filtered_ftrees, test_start_types, valid_count, filter_away_not_0_1 

    filter_away_not_0_1 = filter_away_not_0_1_local 

    # testing 
    test_smiless, test_ftrees, test_peakslist = dataloader.get_data('test') # test 

    # add validation cases 
    valid_smiless, valid_ftrees, valid_peakslist = dataloader.get_data('dev') 
    
    test_smiless += valid_smiless 
    test_ftrees += valid_ftrees 
    test_peakslist += valid_peakslist 



    # preprocess some data 
    test_filtered_smiless = [] 
    test_filtered_ftrees = [] 
    test_graphs = [] 
    if filter_away_not_0_1: 
        test_start_types = [] 

    for idx in range(len(test_smiless)): 
        # get graph of target molecule to make MolState 
        target_graph = utils.SMILEStoGraph(test_smiless[idx]).to(device) 

        # filter away molecules that are too large 
        if sum(utils.smiles_to_atom_counts(test_smiless[idx], include_H=False)) > max_num_heavy_atoms: 
            continue 

         # make sure that we are only learning those with benzene or no benzene but not other aromatic structures 
        aromatic_bonds = [] 
        for e in range(len(target_graph.edata['bondTypes'])//2): 
            if target_graph.edata['bondTypes'][2*e,4] == 1: 
                aromatic_bonds.append(e) 
        
        #start_type = 0 
        if (filter_away_not_0_1): 
            
            if len(aromatic_bonds) != 0 and len(aromatic_bonds) != 6: continue # don't try this 
            start_type = 0 # -------------------------------------------------------------------------------------------------
            if len(aromatic_bonds) == 6: 
                start_type = 1 # ---------------------------------------------------------------------------------------------
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
        else: 
            if len(aromatic_bonds) == len(target_graph.edata['bondTypes'])//2: 
                continue # skip this test case because there's no action to take 
        
        mass_spec = MassSpec(False, test_ftrees[idx], mass_spec_gcn_path, None, device=device) # path doesn't have to be correct, just needed for syntax issues 
        init_formula = CalcMolFormula(Chem.MolFromSmiles(test_smiless[idx]))  
        target_state = MolState(test_smiless[idx], utils.formula_to_atom_counts(init_formula), target_graph, 0, device) 
        
        rem_Hs, fatal_errors = EnvMolecule.compare_state_to_mass_spec(target_state, mass_spec, True) 
        if len(fatal_errors) > 0: 
            continue 

        # now, verified to be okay :) 

        num_bonds = len(list(target_graph.edges()))//2 

        test_filtered_smiless.append(test_smiless[idx]) 
        test_filtered_ftrees.append(test_ftrees[idx]) 
        #print(target_graph.device, mass_spec.pred.device)
        test_graphs.append(target_graph) 

        if filter_away_not_0_1: 
            test_start_types.append(start_type) 
        else: 
            # set it to -1 as it should not initialize like that 
            test_start_types.append(-1) 



    valid_count = int(valid_fraction * len(test_filtered_smiless)) 



# test first k function 
def test_first_k(epoch, num_leaves, save_prefix:str="", save_states=True): # NOTE: specifying save_prefix can be useful even if save_states=False, because it also saves error states there. 
    num_corrects = [0 for _ in range(num_leaves)] 
    num_tests = 0 
    test_state_ai = MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 

    action_accuracies = [] 
    correct_action_counts = [] 
    single_bond_counts = [] 
    total_action_counts = [] 

    issue_count = 0 

    for test_idx in range(valid_count, len(test_filtered_smiless)): 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]).to(device) 

        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        correct_actions = test_target_state.get_valid_unactions() 
        del test_target_state 

        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.Queue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            #if leaves.full(): 
            #    highest = max(leaves.queue) 
            #    if elem[0] > highest[0]: return 
            #    leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        if filter_away_not_0_1: 
            test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_types[test_idx]) 
        else: 
            test_start_graph = test_target_graph.clone() 
        
        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph.to(device), 0, device) 

        if not filter_away_not_0_1: 
            unactions = test_start_state.get_valid_unactions() 
            for ua in unactions: 
                test_start_state = test_start_state.undo_action(ua) 

        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_top_k_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)): 
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in correct_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_top_k_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                        
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( am.Q(am.paths_tree[new_idx], torch.tensor([am.paths_depths[new_idx]]).to(device)).item() , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("UGHH, SMTG IS WRONG WITH ADDING TO LEAF??? BLEURGH SMTG IS WRONG. ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, save_prefix+"test_issue_"+str(issue_count)+"_epoch_"+str(epoch)+"_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        #print("NUM STATES VISITED:",len(am.paths_tree)) 
        print(len(am.paths_tree), end=" ") # hehe 
        
        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ]
        
        # check if any are correct 
        i = 0 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph.cpu()): 
                num_corrects[i] += 1 
            i += 1 
        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        single_bond_counts.append(num_single_bond_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    res = np.array(num_corrects)/num_tests 
    res = res.tolist() 

    print() 
    print("EPOCH",epoch,"FIRST",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("AVG ACTION ACCURACY:",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts 




# valid first k function 
def valid_first_k(epoch, num_leaves, save_prefix:str="", save_states=True): # NOTE: specifying save_prefix can be useful even if save_states=False, because it also saves error states there. 
    num_corrects = [0 for _ in range(num_leaves)] 
    num_tests = 0 
    test_state_ai = MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 

    action_accuracies = [] 
    correct_action_counts = [] 
    total_action_counts = [] 
    single_bond_counts = [] 

    issue_count = 0 

    for test_idx in range(valid_count): 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]).to(device) 

        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        correct_actions = test_target_state.get_valid_unactions() 
        del test_target_state 

        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.Queue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            #if leaves.full(): 
            #    highest = max(leaves.queue) 
            #    if elem[0] > highest[0]: return 
            #    leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        if filter_away_not_0_1: 
            test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_types[test_idx]) 
        else: 
            test_start_graph = test_target_graph.clone() 
        
        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph.to(device), 0, device) 

        if not filter_away_not_0_1: 
            unactions = test_start_state.get_valid_unactions() 
            for ua in unactions: 
                test_start_state = test_start_state.undo_action(ua) 

        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_top_k_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)): 
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in correct_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_top_k_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                            
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( am.Q(am.paths_tree[new_idx], torch.tensor([am.paths_depths[new_idx]]).to(device)).item() , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("UGHH, SMTG IS WRONG WITH ADDING TO LEAF??? BLEURGH SMTG IS WRONG. ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, save_prefix+"test_issue_"+str(issue_count)+"_epoch_"+str(epoch)+"_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        #print("NUM STATES VISITED:",len(am.paths_tree)) 
        print(len(am.paths_tree), end=" ") # hehe 
        
        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ]
        
        # check if any are correct 
        i = 0 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph.cpu()): 
                num_corrects[i] += 1 
            i += 1 

        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 

        single_bond_counts.append(num_single_bond_actions) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    res = np.array(num_corrects)/num_tests 
    res = res.tolist() 

    print() 
    print("VALID: EPOCH",epoch,"FIRST",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("AVG ACTION ACCURACY:",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts 






# test top k function 
def test_top_k(epoch, num_leaves, save_prefix:str="", save_states=True): 
    num_corrects = [0 for _ in range(num_leaves)] 
    num_tests = 0 
    test_state_ai = MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 

    action_accuracies = [] 
    correct_action_counts = [] 
    total_action_counts = [] 

    single_bond_counts = [] 

    issue_count = 0 

    for test_idx in range(valid_count, len(test_filtered_smiless)): 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]).to(device) 

        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        correct_actions = test_target_state.get_valid_unactions() 
        del test_target_state 

        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.PriorityQueue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            if leaves.full(): 
                highest = max(leaves.queue) 
                if elem[0] > highest[0]: 
                    # delete state from memory 
                    am.paths_tree[elem[1][0]] = None 
                    return 
                # delete state from memory 
                am.paths_tree[highest[1][0]] = None 
                leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        if filter_away_not_0_1: 
            test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_types[test_idx]) 
        else: 
            test_start_graph = test_target_graph.clone() 

        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph.to(device), 0, device) 

        if not filter_away_not_0_1: 
            unactions = test_start_state.get_valid_unactions() 
            for ua in unactions: 
                test_start_state = test_start_state.undo_action(ua) 

        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_top_k_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)): 
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in correct_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_top_k_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                            
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( am.Q(am.paths_tree[new_idx], torch.tensor([am.paths_depths[new_idx]]).to(device)).item() , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("UGHH, SMTG IS WRONG WITH ADDING TO LEAF??? BLEURGH SMTG IS WRONG. ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, save_prefix+"test_issue_"+str(issue_count)+"_epoch_"+str(epoch)+"_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        #print("NUM STATES VISITED:",len(am.paths_tree)) 
        print(len(am.paths_tree), end=" ") # hehe 
        
        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ]
        
        # check if any are correct 
        i = 0 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph.cpu()): 
                num_corrects[i] += 1 
            i += 1 

        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 

        single_bond_counts.append(num_single_bond_actions) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    res = np.array(num_corrects)/num_tests 

    res = res.tolist() 

    print() 
    print("EPOCH",epoch,"TOP",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("AVG ACTION ACCURACY:",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts 


# valid top k function 
def valid_top_k(epoch, num_leaves, save_prefix:str="", save_states=True): 
    num_corrects = [0 for _ in range(num_leaves)] 
    num_tests = 0 
    test_state_ai = MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 

    issue_count = 0 

    action_accuracies = [] 
    correct_action_counts = [] 
    total_action_counts = [] 

    single_bond_counts = [] 

    for test_idx in range(valid_count): 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]).to(device) 

        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        correct_actions = test_target_state.get_valid_unactions() 
        del test_target_state 

        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.PriorityQueue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            if leaves.full(): 
                highest = max(leaves.queue) 
                if elem[0] > highest[0]: 
                    # delete state from memory 
                    am.paths_tree[elem[1][0]] = None 
                    return 
                # delete state from memory 
                am.paths_tree[highest[1][0]] = None 
                leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        if filter_away_not_0_1: 
            test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_types[test_idx]) 
        else: 
            test_start_graph = test_target_graph.clone() 

        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph.to(device), 0, device) 

        if not filter_away_not_0_1: 
            unactions = test_start_state.get_valid_unactions() 
            for ua in unactions: 
                test_start_state = test_start_state.undo_action(ua) 

        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_top_k_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)): 
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in correct_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_top_k_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                            
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( am.Q(am.paths_tree[new_idx], torch.tensor([am.paths_depths[new_idx]]).to(device)).item() , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("UGHH, SMTG IS WRONG WITH ADDING TO LEAF??? BLEURGH SMTG IS WRONG. ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, save_prefix+"test_issue_"+str(issue_count)+"_epoch_"+str(epoch)+"_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        #print("NUM STATES VISITED:",len(am.paths_tree)) 
        print(len(am.paths_tree), end=" ") # hehe 
        
        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ]
        
        # check if any are correct 
        i = 0 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph.cpu()): 
                num_corrects[i] += 1 
            i += 1 

        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 

        single_bond_counts.append(num_single_bond_actions) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    res = np.array(num_corrects)/num_tests 

    res = res.tolist() 

    print() 
    print("VALID: EPOCH",epoch,"TOP",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("AVG ACTION ACCURACY:",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts 






# random top k function 
def random_top_k(valid, num_leaves, save_prefix:str="", save_states=True): # valid True means validation, else test 
    num_corrects = [0 for _ in range(num_leaves)] 
    num_tests = 0 
    
    #epoch = 30 # arbritrary value to prevent errors 
    test_state_ai = None #MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 
    
    if valid: 
        r = range(valid_count) 
    else: 
        r = range(valid_count, len(test_filtered_smiless)) 

    issue_count = 0 

    action_accuracies = [] 
    correct_action_counts = [] 
    total_action_counts = [] 

    single_bond_counts = [] 

    for test_idx in r: 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]).to(device) 

        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        correct_actions = test_target_state.get_valid_unactions() 
        del test_target_state 


        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.PriorityQueue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            if leaves.full(): 
                highest = max(leaves.queue) 
                if elem[0] > highest[0]: 
                    # delete state from memory 
                    am.paths_tree[elem[1][0]] = None 
                    return 
                # delete state from memory 
                am.paths_tree[highest[1][0]] = None 
                leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        if filter_away_not_0_1: 
            test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_types[test_idx]) 
        else: 
            test_start_graph = test_target_graph.clone() 

        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph.to(device), 0, device) 

        if not filter_away_not_0_1: 
            unactions = test_start_state.get_valid_unactions() 
            for ua in unactions: 
                test_start_state = test_start_state.undo_action(ua) 

        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_k_random_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)): 
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in correct_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_k_random_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                            
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( random.uniform(0,1) , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("SMTG IS WRONG WITH ADDING TO LEAF ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, save_prefix+"test_issue_"+str(issue_count)+"_random_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        #print("NUM STATES VISITED:",len(am.paths_tree)) 
        print(len(am.paths_tree), end=" ") # hehe 
        
        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ]
        
        # check if any are correct 
        i = 0 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph.cpu()): 
                num_corrects[i] += 1 
            i += 1 
        
        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 

        single_bond_counts.append(num_single_bond_actions) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    res = np.array(num_corrects)/num_tests 
    res = res.tolist() 

    print() 
    if valid: 
        print("RANDOM: VALID: TOP",num_leaves,"GUESS SUCCESS RATES:", res ) 
    else: 
        print("RANDOM: TEST: TOP",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("AVG ACTION ACCURACY",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts 







# test from some depth 
def try_top_k_depth(epoch, num_leaves, test_depth, save_prefix:str, save_states:bool, valid:bool=False): 

    num_corrects = [0 for _ in range(num_leaves)] 
    action_accuracies = [] 
    correct_action_counts = [] 
    total_action_counts = [] 

    single_bond_counts = [] 

    num_tests = 0 
    test_state_ai = test_state_ai = MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/center_epoch_"+str(epoch)+".pt", path_prefix+"/radius_epoch_"+str(epoch)+".txt", device=device) 

    issue_count = 0 

    if valid: 
        r = range(valid_count) 
    else: 
        r = range(valid_count, len(test_filtered_smiless)) 

    for test_idx in r: 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]) 


        # remove bonds from graph to get a start graph - use convenient undo action function 
        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        removable_bonds = test_target_state.get_valid_unactions() 

        if len(removable_bonds) < test_depth: 
            # not enough bonds that can be removed 
            continue 

        samples = random.sample(list(range(len(removable_bonds))), test_depth) 

        removed_actions = [] 
       
        for sample in samples: 
            test_target_state = test_target_state.undo_action(removable_bonds[sample]) 
            removed_actions.append(removable_bonds[sample]) 
        
        test_start_graph = copy.deepcopy(test_target_state.graph).to(device) 
        del test_target_state 



        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.PriorityQueue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            if leaves.full(): 
                highest = max(leaves.queue) 
                if elem[0] > highest[0]: 
                    # delete state from memory 
                    am.paths_tree[elem[1][0]] = None 
                    return 
                # delete state from memory 
                am.paths_tree[highest[1][0]] = None 
                leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        #test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_type) 
        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph, test_start_types[test_idx], device) 
        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_top_k_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)):         
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        
        # get percentage of correct actions 
        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in removed_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_top_k_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                            
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( am.Q(am.paths_tree[new_idx], torch.tensor([am.paths_depths[new_idx]]).to(device)).item() , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("UGHH, SMTG IS WRONG WITH ADDING TO LEAF??? BLEURGH SMTG IS WRONG. ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, path_prefix+"/states/hindsight_experience_replay/test_issue_"+str(issue_count)+"_epoch_"+str(epoch)+"_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ] 
        
        # check if any are correct 
        i = 0 
        test_target_graph = test_target_graph.cpu() # just in case 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph): 
                num_corrects[i] += 1 
            i += 1 
        

        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 

        single_bond_counts.append(num_single_bond_actions) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    if len(correct_action_counts) == 0: 
        print("DEPTH",test_depth,"HAD NO POSSIBLE CASES!") 
        return [], [], [] 

    res = np.array(num_corrects)/num_tests 
    res = res.tolist() 

    print("EPOCH",epoch,"DEPTH",test_depth,"TOP",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("EPOCH",epoch,"DEPTH",test_depth,"AVG ACTION ACCURACY",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts 



# random from some depth 
def random_top_k_depth(num_leaves, test_depth, save_prefix:str, save_states:bool, valid:bool=False): # also get target results here. 

    num_corrects = [0 for _ in range(num_leaves)] 
    action_accuracies = [] 
    correct_action_counts = [] 
    total_action_counts = [] 

    single_bond_counts = [] 
    target_single_bond_percentages = [] 

    num_tests = 0 
    test_state_ai = None #MolStateAI(False, path_prefix+'/MolStateGCN_epoch_'+str(epoch)+'.pt', path_prefix+"/HypersphereParams_epoch_"+str(epoch)+".txt", None, None, device=device) 

    issue_count = 0 

    if valid: 
        r = range(valid_count) 
    else: 
        r = range(valid_count, len(test_filtered_smiless)) 

    for test_idx in r: 
        # target 
        test_target_graph = utils.SMILEStoGraph(test_filtered_smiless[test_idx]) 



        # remove bonds from graph to get a start graph - use convenient undo action function 
        test_target_state = MolState(test_filtered_smiless[test_idx], graph=test_target_graph) 
        removable_bonds = test_target_state.get_valid_unactions() 

        if len(removable_bonds) < test_depth: 
            # not enough bonds that can be removed 
            continue 

        samples = random.sample(list(range(len(removable_bonds))), test_depth) 

        removed_actions = [] 
       
        for sample in samples: 
            test_target_state = test_target_state.undo_action(removable_bonds[sample]) 
            removed_actions.append(removable_bonds[sample]) 
        
        test_start_graph = copy.deepcopy(test_target_state.graph).to(device) 
        del test_target_state 



        test_mass_spec = MassSpec(False, test_filtered_ftrees[test_idx], mass_spec_gcn_path, None, device=device) 
        test_init_formula = CalcMolFormula(Chem.MolFromSmiles(test_filtered_smiless[test_idx]))   

        pq = queue.PriorityQueue(beam_width) # element format: (heuristic, (paths_tree_idx, CurrentState), pair<Action, MolState>) 
        leaves = queue.PriorityQueue(num_leaves) # (1-score, (paths_tree_idx, CurrentState)) 
        # the reason for the random comparison functions in Action and MolState are so as to not give an error in this 

        # NOTE: heuristic used in ASTAR search here is (1-score)*(total_num_Hs//2) - num_Hs_left, so less anomalous and closer to goal (less Hs left) will be better for heuristic 
        # though, since priority queue takes lowest, we actually use num_Hs_left - (1-score)*(total_num_Hs//2) 
        # total_num_Hs//2 is also called score_multiplier 
        
        def add_to_pq(elem): 
            if pq.full(): 
                highest = max(pq.queue) 
                if elem[0] > highest[0]: return 
                pq.queue.remove(highest) 
            pq.put_nowait(elem) 
        
        def add_to_leaves(elem): 
            if leaves.full(): 
                highest = max(leaves.queue) 
                if elem[0] > highest[0]: 
                    # delete state from memory 
                    am.paths_tree[elem[1][0]] = None 
                    return 
                # delete state from memory 
                am.paths_tree[highest[1][0]] = None 
                leaves.queue.remove(highest) 
            leaves.put_nowait(elem) 
        

        #test_start_graph = utils.SMILEStoGraphType(test_filtered_smiless[test_idx], test_start_type) 
        test_start_state = MolState(test_filtered_smiless[test_idx], utils.formula_to_atom_counts(test_init_formula), test_start_graph, test_start_types[test_idx], device) 
        am = AgentMolecule(test_mass_spec, test_start_state, test_state_ai, start_epsilon=0, max_steps=100) 
        
        # used in computing heuristic 
        score_multiplier = test_start_state.get_H_count()//2 


        try: 
            results, pairs = am.get_k_random_pairs(0, num_guesses_per_state, True) 

            for i in range(len(results)):         
                add_to_pq(( am.paths_tree[0].get_H_count() - (1-results[i])*score_multiplier , (0 , copy.deepcopy(am.paths_tree[0])), pairs[i])) 

        except NoValidActionException: 
            # this is normal, because there's bound to be no solution using benzene if the answer has no benzene. 
            pass 

        
        # get target percentage of single bond actions 
        num_target_single_bonds = 0 
        for target_a in removed_actions: 
            if target_a.type == 0: 
                num_target_single_bonds += 1 

        target_single_bond_percentages.append(num_target_single_bonds/len(removed_actions)) 

        
        # get percentage of correct actions 
        num_correct_actions = 0 
        total_num_actions = 0 

        num_single_bond_actions = 0 

        while (not pq.empty()) and (not leaves.full()):
            score, state_tuple, pair = pq.get_nowait()
            idx, state = state_tuple 

            if pair[0] in removed_actions: 
                num_correct_actions += 1 
            total_num_actions += 1 

            if pair[0].type == 0: 
                num_single_bond_actions += 1 

            am.paths_tree[idx] = state 

            #record = am.take_top_k_actions(idx, 2, True) 
            new_idx = am._take_action(pair[1], pair[0], idx) 

            if (am.can_take_action(new_idx)):
                try: 
                    # add all actions from new_idx 
                    results, pairs = am.get_k_random_pairs(new_idx, num_guesses_per_state, True) 
                    for i in range(len(results)): 
                        
                        # make sure it's not already in 
                        already = False 
                        for leaf in leaves.queue: 
                            if utils.is_isomorphic(leaf[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        for pqitem in pq.queue: 
                            if utils.is_isomorphic(pqitem[1][1].graph.cpu(), pairs[i][1].graph.cpu()): 
                                already = True 
                                break 
                        
                        if already: 
                            continue 
                            
                        add_to_pq(( am.paths_tree[new_idx].get_H_count() - (1-results[i])*score_multiplier , (new_idx, copy.deepcopy(am.paths_tree[new_idx])), pairs[i])) 
                    
                    
                    # save memory, clear non-numeric values in current entry 
                    am.paths_tree[new_idx] = None 
                    am.paths_actions_tried[new_idx] = [] 
                    am.paths_prev_actions[new_idx] = None 
                    
                except NoValidActionException: # this is normal haha 
                    pass 
            else: 
                # insert as a potential solution 
                try: 
                    add_to_leaves( ( random.uniform(0,1) , (new_idx, copy.deepcopy(am.paths_tree[new_idx])) ) ) # because higher score means less anomalous, so, good. 
                except Exception as e: 
                    print("UGHH, SMTG IS WRONG WITH ADDING TO LEAF??? BLEURGH SMTG IS WRONG. ISSUE NUMBER",issue_count) 
                    print(e) 
                    am.paths_tree[new_idx].save(new_idx, path_prefix+"/states/hindsight_experience_replay/test_issue_"+str(issue_count)+"_random_index_"+str(new_idx)+".bin") 
                    issue_count += 1 

            am.paths_tree[new_idx] = None 

        
        if leaves.empty(): 
            print("ERROR: NO LEAVES SAVED IN TESTING",test_filtered_smiless[test_idx]) 
        else: 
            #print("DEBUG: TRAIN LEAVES:",leaves.queue)
            pass 

        leaf_states = [ l[1][1] for l in leaves.queue ] 
        
        # check if any are correct 
        i = 0 
        test_target_graph = test_target_graph.cpu() # just in case 
        while (not (leaves.empty())): 
            _, state_tuple = leaves.get_nowait() 
            leaf, state = state_tuple 
            if utils.is_isomorphic(state.graph.cpu(), test_target_graph): 
                num_corrects[i] += 1 
            i += 1 
        

        correct_action_counts.append(num_correct_actions) 
        total_action_counts.append(total_num_actions) 

        action_accuracy = num_correct_actions / total_num_actions 
        action_accuracies.append(action_accuracy) 

        single_bond_counts.append(num_single_bond_actions) 
        
        num_tests += 1 

        # save states 
        if save_states: MolState.save_states(leaf_states, list(range(len(leaf_states))), save_prefix+test_filtered_smiless[test_idx]+".bin") 

    if len(correct_action_counts) == 0: 
        print("DEPTH",test_depth,"HAD NO POSSIBLE CASES!") 
        return [], [], [], [] 

    res = np.array(num_corrects)/num_tests 
    res = res.tolist() 

    print("RANDOM DEPTH",test_depth,"TOP",num_leaves,"GUESS SUCCESS RATES:", res ) 
    print("RANDOM DEPTH",test_depth,"AVG ACTION ACCURACY",sum(action_accuracies)/num_tests) 
    print("FRACTION OF SINGLE BONDS:", sum((np.array(single_bond_counts) / np.array(total_action_counts))/num_tests))
    print() 

    return res, correct_action_counts, total_action_counts, single_bond_counts, sum(target_single_bond_percentages)/len(target_single_bond_percentages) 



