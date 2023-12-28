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



def load_data(source="test"):
    return dataloader.get_data(source) 

def make_mass_spec(): 
    ms = MassSpec(True, None, './RL_attempt/models/FTreeGCN_testing_1.pt', "./RL_attempt/models/FTreeAmplitudePredictor_testing_1.pt", device) 
    ms.save() 
    ms.save_amplitude_predictor() 
    del ms 

def get_agent_molecule_idx(idx, smiless, ftrees, peakslist): 
    #smiless, ftrees, peakslist = dataloader.get_data(source) 
    global mass_spec 
    global init_formula 
    global init_graph 
    global init_state 
    global state_ai 

    mass_spec = MassSpec(False, ftrees[idx], './RL_attempt/models/FTreeGCN_testing_1.pt', None, device=device) 
    
    # init_state 
    init_formula = CalcMolFormula(Chem.MolFromSmiles(smiless[idx])) 
    init_graph = utils.get_init_graph(init_formula) 
    init_state = MolState(smiless[idx], utils.formula_to_atom_counts(init_formula), init_graph, 0, device) 
    state_ai = MolStateAI(True, "./RL_attempt/models/MolStateGCN_testing_1.pt", "./RL_attempt/models/MolStatePredictor_testing_1.pt", device) 

    return AgentMolecule(mass_spec, init_state, state_ai) 

def save_all(a:AgentMolecule): 
    a.state_ai.save() 
    try: 
        a.state_ai.save_predictor() 
    except: 
        pass 

    a.mass_spec.save() 
    try: 
        a.mass_spec.save_amplitude_predictor() 
    except: 
        pass 

def show_graphs(itr):
    for i in itr:
        agentmol.paths_tree[i].show_visualization() 


make_mass_spec() 

dataloader.warn = False
print("Loading data...") 
#testdata = load_data('test')
testdata = dataloader.get_idxth_entry(0, 'test') 

print("SMILES:", testdata[0][0]) 

#print("DATA idx 0: ") 
#print(testdata[0][0]) 
#print(testdata[1][0]) 
#print(testdata[2][0]) 

print("Creating molecule...") 
agentmol = get_agent_molecule_idx(0, *testdata)

#print(agentmol.paths_tree[0].graph) 

save_all(agentmol) 


#print("MOLECULE DATA: ") 
#print(agentmol.paths_tree[0].graph.ndata['features']) 
#print(agentmol.paths_tree[0].graph.edata['bondTypes']) 
#print()

#print(agentmol.paths_tree[0].graph) 

#print("STATE DICTS: ")
#print(mass_spec.model.state_dict()) 
#print(mass_spec.amplitude_predictor.state_dict())
#print(state_ai.model.state_dict()) 
#print(state_ai.final_predictor.state_dict()) 

pairs = agentmol._get_pairs_above_threshold(0, -1) 
#print("PAIRS", pairs) 
selected = pairs[0] 
#print("SELECTED:", selected)


# test some taking actions

agentmol._take_action(selected[1], selected[0], 0, True) 

agentmol.take_actions_above_threshold(1, -0.001)

agentmol.take_actions_above_threshold(10, 0)

#agentmol.reward_trajectory(25) 

print('\n\n\n TESTING YESSSS')

agentmol.max_steps = 20 
agentmol.epsilon = 0.5 

# test using queue to take actions from each state 
import queue 
sq = queue.SimpleQueue() 
sq.put_nowait(len(agentmol.paths_tree)-1)
sq.put_nowait(len(agentmol.paths_tree)-2) 
while (not sq.empty()):
    idx = sq.get_nowait()
    if (agentmol.can_take_action(idx)):
        try:
            record = agentmol.take_top_k_actions(idx, 2) 
            #record = [agentmol.epsilon_greedy_take_best_action(idx)] 
            for _, new_idx in record:
                sq.put_nowait(new_idx)
        except NoValidActionException: 
            print("NO VALID ACTIONS AT INDEX",idx,"REVERTING TO",agentmol.paths_parents[idx]) 
            sq.put_nowait(agentmol.paths_parents[idx]) 
    else:
        print(idx)




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




