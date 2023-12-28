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



save_states = False 
k = 100 
num_guesses_per_state = 2 
test_type = "max_12" # rmbr to change astar_search.py max_num_heavy_atoms based on test_type ---------------- THIS IS VERY IMPT WEIORFHWOEFNLSKNF:FHUEBF:SJKFN:SEIHF:SIENF:SNFEFLSBFSKLIHEB 

filter_away_not_0_1 = False 

epoch_nums = [30, 65] 
gcn_lrs = [5e-04] 
nus = [0.05] 
cannots = [] 


valid_smiless = ['CCC(C(=O)N)N1CCCC1=O', 'C(C(C(=O)NCC(=O)O)N)S', 'C(CC(=O)N)C(C(=O)O)N', 'C(CCN)CC(C(=O)O)N', 'CC(C(=O)NC(C)C(=O)O)N', 'C(CCN)CC(=O)O', 'C(CC(C(=O)O)N)C(CN)O', 'C(C1C(C(C(C(O1)O)O)O)O)O', 'C(CO)NCCO', 'C(CC(C(=O)O)N)CNC(=O)N', 'CC(C(=O)O)N', 'CC(=O)NC(CO)C(=O)O', 'C(CCN)CC(C(=O)O)N', 'C(CC(C(=O)O)N)C(CN)O', 'C(CN)C(=O)O', 'CC(=O)NCCCC(=O)O', 'C(=O)(N)NC(=O)N', 'CC(=O)NC(CCSC)C(=O)O', 'C1CCNC(C1)C(=O)O', 'CC(C(CC(=O)O)N)O', 'CC(C)CC(C(=O)O)N', 'CC(=O)NCCCC(=O)O', 'CSCCC(C(=O)O)N', 'CC(=O)NCCCCN', 'CN(C)CC(=O)O', 'CC(=CC(=O)NCC(=O)O)C', 'CC(=O)NCCCS(=O)(=O)O', 'C(CCC(=O)O)CCO', 'C(C1C(C(C(C(O1)O)O)O)O)O', 'CC(C)(C(=O)O)N', 'CNCC(=O)O', 'C(CO)N(CCO)CCO', 'CSCCC(C(=O)O)N'] 
test_smiless = ['CC(=O)NC(CO)C(=O)O', 'CC(C(=O)O)N', 'C(CC(C(=O)O)N)CNC(=O)N', 'C(C(C(=O)O)N)C(=O)N', 'CCOP(=O)(OCC)OCC', 'C(CC(=O)N)C(C(=O)O)N', 'C(CC(C(=O)O)N)CN=C(N)N', 'C(CC(C(=O)O)N)CN', 'C(C(C(=O)O)N=C(N)N)C(=O)O', 'CC=C(C)C(=O)NCC(=O)O', 'CCCCOCCOCCO', 'C(CN)C(=O)O', 'C(COP(=O)(O)O)N', 'C(CCN)CN', 'C(CCN=C(N)N)CN', 'CC(C(C(=O)O)N)O', 'C(C(C(=O)O)N)O', 'CCC(C(=O)O)N', 'C(CC(=O)O)CN=C(N)N', 'C(CC(=O)O)C(C(=O)O)N', 'C1CC(NC1)C(=O)O', 'CC(C)C(C(=O)O)NC(=O)CN', 'C(CP(=O)(O)O)C(=O)O', 'C(CC(=O)O)C(C(=O)O)N', 'C(C1C(C(C(C(O1)O)N)O)O)O', 'CC(C)C(C(=O)O)N', 'C1C(CNC1C(=O)O)O', 'C(C1C(C(C(C(O1)O)O)O)O)O', 'CC(=O)NC(CCSC)C(=O)O', 'CC1=NCCC(N1)C(=O)O', 'C1CNCCC1C(=O)N', 'C1CCN(CC1)N=O', 'CC(=O)N1CCCC1CC(=O)O', 'C1CNC(C1O)C(=O)O'] 


# search possible gcn_lr and nu 
for gcn_lr in gcn_lrs: 
    for nu in nus: 
        if (gcn_lr, nu) in cannots: continue # skit because mm . 
        #if gcn_lr == 5e-06 and nu == 0.1: continue # to skip this case as it has alerady been done 
        path_prefix = './RL_attempt/non_anomalous_grid_search_'+str(test_type)+'/search_'+str(gcn_lr)+"_"+str(nu) 
        valid_ress = [] 
        ress = [] 
        print() 
        print() 
        print("TESTING: GCN_LR:",gcn_lr,"nu:", nu)

        avg_single_bond_percents = {} 

        for epoch in epoch_nums: 
            path = path_prefix+"/epoch_"+str(epoch)+"_top_"+str(k)+"_astar_tests_try"+str(num_guesses_per_state)+"/" 

            valid_path = path_prefix+"/epoch_"+str(epoch)+"_top_"+str(k)+"_astar_valids_try"+str(num_guesses_per_state)+"/" 

            single_bond_percents = [] 

            for smiles in valid_smiless: 
                try: 
                    idxs, states = MolState.load_states(valid_path+smiles+".bin") 
                    num_single_bonds = 0 
                    total_num_bonds = 0 
                    for state in states: 
                        for eidx in range(len(state.graph.edata['bondTypes'])): 
                            # turn one-hot encoding to edge type 
                            idx = 1 
                            while idx < len(state.graph.edata['bondTypes'][eidx]): 
                                if state.graph.edata['bondTypes'][eidx, idx].item() == 1: 
                                    break 
                                idx += 1 

                            etype = idx - 1 
                            if etype == 0: 
                                num_single_bonds += 1 
                            
                            total_num_bonds += 1 
                    
                    single_bond_percents.append(num_single_bonds/total_num_bonds) 
                except Exception as e: 
                    print("EXCEPTION:", e) 
                    print("AT SMILES:",smiles) 

            for smiles in test_smiless: 
                try: 
                    idxs, states = MolState.load_states(path+smiles+".bin") 
                    num_single_bonds = 0 
                    total_num_bonds = 0 
                    for state in states: 
                        for eidx in range(len(state.graph.edata['bondTypes'])): 
                            # turn one-hot encoding to edge type 
                            idx = 1 
                            while idx < len(state.graph.edata['bondTypes'][eidx]): 
                                if state.graph.edata['bondTypes'][eidx, idx].item() == 1: 
                                    break 
                                idx += 1 

                            etype = idx - 1 
                            if etype == 0: 
                                num_single_bonds += 1 
                            
                            total_num_bonds += 1 
                    
                    single_bond_percents.append(num_single_bonds/total_num_bonds) 
                except Exception as e: 
                    print("EXCEPTION:", e) 
                    print("AT SMILES:",smiles) 
            
            avg_single_bond_percents["EPOCH "+str(epoch)] = sum(single_bond_percents)/len(single_bond_percents) 
        
        
        
        # do test on random 
        path = path_prefix+"/random_top_"+str(k)+"_astar_tests_try"+str(num_guesses_per_state)+"/" 
        valid_path = path_prefix+"/random_top_"+str(k)+"_astar_valids_try"+str(num_guesses_per_state)+"/" 
        
        single_bond_percents = [] 

        for smiles in valid_smiless: 
            try: 
                idxs, states = MolState.load_states(valid_path+smiles+".bin") 
                num_single_bonds = 0 
                total_num_bonds = 0 
                for state in states: 
                    for eidx in range(len(state.graph.edata['bondTypes'])): 
                        # turn one-hot encoding to edge type 
                        idx = 1 
                        while idx < len(state.graph.edata['bondTypes'][eidx]): 
                            if state.graph.edata['bondTypes'][eidx, idx].item() == 1: 
                                break 
                            idx += 1 

                        etype = idx - 1 
                        if etype == 0: 
                            num_single_bonds += 1 
                        
                        total_num_bonds += 1 
                
                single_bond_percents.append(num_single_bonds/total_num_bonds) 
            except Exception as e: 
                print("EXCEPTION:", e) 
                print("AT SMILES:",smiles) 
        
        for smiles in test_smiless: 
            try: 
                idxs, states = MolState.load_states(path+smiles+".bin") 
                num_single_bonds = 0 
                total_num_bonds = 0 
                for state in states: 
                    for eidx in range(len(state.graph.edata['bondTypes'])): 
                        # turn one-hot encoding to edge type 
                        idx = 1 
                        while idx < len(state.graph.edata['bondTypes'][eidx]): 
                            if state.graph.edata['bondTypes'][eidx, idx].item() == 1: 
                                break 
                            idx += 1 

                        etype = idx - 1 
                        if etype == 0: 
                            num_single_bonds += 1 
                        
                        total_num_bonds += 1 
                
                single_bond_percents.append(num_single_bonds/total_num_bonds) 
            except Exception as e: 
                print("EXCEPTION:", e) 
                print("AT SMILES:",smiles) 
        
        avg_single_bond_percents["RANDOM"] = sum(single_bond_percents)/len(single_bond_percents) 


        # get for correct answer 
        single_bond_percents = [] 
        for smiles in valid_smiless: 
            num_single_bonds = 0 
            total_num_bonds = 0 
            target_graph = utils.SMILEStoGraph(smiles) 
            for eidx in range(len(target_graph.edata['bondTypes'])): 
                
                idx = 1 
                while idx < len(target_graph.edata['bondTypes'][eidx]): 
                    if target_graph.edata['bondTypes'][eidx, idx].item() == 1: 
                        break 
                    idx += 1 

                etype = idx - 1 
                if etype==0: 
                    num_single_bonds += 1 
                
                total_num_bonds += 1 
            
            single_bond_percents.append(num_single_bonds/total_num_bonds) 
        
        print(len(single_bond_percents))
        
        avg_single_bond_percents["TARGET"] = sum(single_bond_percents)/len(single_bond_percents) 



        

        for key, val in avg_single_bond_percents.items(): 
            print(key) 
            print(val) 
            print() 

"""
OUTPUT: 

TESTING: GCN_LR: 0.0005 nu: 0.05
33
EPOCH 30
0.943482507351867

EPOCH 65
0.9323865559382299

RANDOM
0.86076641812574

TARGET
0.8450894223621496


"""


