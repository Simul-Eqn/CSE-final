import sys
sys.path.insert(0, "./RL_attempt/") 

import os
os.environ['DGLBACKEND'] = 'pytorch'

import networkx as nx 

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


smiles, ftree, _ = dataloader.get_idxth_entry(0, 'test')
smiles = smiles[0]
ftree = ftree[0] 

init_state = MolState(smiles, None, utils.get_init_graph(utils.smiles_to_formula(smiles)))
ms = MassSpec(False, ftree, './RL_attempt/mass_spec_lr_search_with_pooling/search_3e-07_3e-07/models/mass_spec_training/FTreeGCN_training_epoch_35.pt', None)
state_ai = MolStateAI(True, './MolStateGCN_test.pt', None, None, torch.zeros((256)).to(device), torch.tensor([0], device=device), device=device)

pos = nx.spring_layout(dgl.to_networkx(init_state.graph))


new_states, actions = init_state.get_next_states(EnvMolecule.compare_state_to_mass_spec(init_state, ms, False), [], True)
ns = new_states[0] 
