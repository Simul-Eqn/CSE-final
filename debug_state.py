import sys
sys.path.insert(0, "./RL_attempt/") 

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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

random.seed(10) 

replay_size = params.replay_buffer_size 


idxs, states = MolState.load_states(".\\RL_attempt\\non_anomalous_grid_search_max_15\\search_0.0005_0.1\\epoch_75_depth_4_top_100_astar_valids_try2\\C(=O)(N)NC(=O)N.bin")



# testing loading and save
'''

# test 1: single graph 
agentmol.paths_tree[62].save(62, "./RL_attempt/states/test.bin")
idxs, states = MolState.load_states("./RL_attempt/states/test.bin")

# show graphs to compare
print("\nCompare graphs after saving and reloading:") 
pos = nx.spring_layout(dgl.to_networkx(agentmol.paths_tree[62].graph), k=0.3, iterations=20)
agentmol.paths_tree[62].show_visualization("original graph", pos)
states[0].show_visualization("saved and reloaded", pos)

# see if valid actions is still correct
print("\nVerify valid actions:") 
a = states[0].get_valid_actions(10, [])
for aa in a: print(aa) 

# try taking actions
print("\nTake an action to make a triple bond to form a loop:") 
ns = states[0].take_action(Action(1,3,2))
ns.show_visualization("hehehe")



# test 2: many graphs
MolState.save_states(agentmol.paths_tree[20:30], [i for i in range(20,30)], "./RL_attempt/states/test2.bin") 
idxs, states = MolState.load_states("./RL_attempt/states/test2.bin")

# show graphs to compare
print("\nCompare graphs before and after loading:") 
for i in range(len(idxs)):
    pos = nx.spring_layout(dgl.to_networkx(agentmol.paths_tree[idxs[i]].graph), k=0.3, iterations=20)
    agentmol.paths_tree[idxs[i]].show_visualization("original "+str(idxs[i]), pos)
    states[i].show_visualization("reloaded "+str(idxs[i]), pos)

# compare all new states from actions for one of them 
p = agentmol.get_all_potential_pairs(29)
p2 = states[-1].get_next_states(EnvMolecule.compare_state_to_mass_spec(states[-1], agentmol.mass_spec, False), agentmol.paths_actions_tried[29],  return_actions=True) # note: this is not same format as p
s2, a2 = p2

print("Compare possible next states after taking actions of graphs before and after loading:") 
for i in range(39):
    pos = nx.spring_layout(dgl.to_networkx(p[i][1].graph), k=0.3, iterations=20)
    p[i][1].show_visualization(str(i)+" original", pos)
    s2[i].show_visualization(str(i)+" reloaded", pos)

'''




