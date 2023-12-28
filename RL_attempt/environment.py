# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['DGLBACKEND'] = 'pytorch'

import collections
import copy

import torch 

from rdkit import Chem
# may try to use for visualization 
#from rdkit.Chem import Draw

from representations import MolState, MolStateAI, MassSpec, Action 
import utils 
import params 

warn = False 

class EnvMolecule(object): 
    # MDP for generating molecule 

    def compare_state_to_mass_spec(state:MolState, mass_spec:MassSpec, check_non_H=True): # returns the amount of extra 
        h_idx = utils.FTreeNode.atomTypes.index("H") 
        if check_non_H: 
            res = 0 # amount of extra Hs 
            fatal_errors=[] # if non-H doesnt match 
            for idx in range(len(utils.FTreeNode.atomTypes)): 
                if idx==h_idx: 
                    res = EnvMolecule.compare_state_to_mass_spec(state, mass_spec, False) 
                else: 
                    s_count = state.atom_counts[idx] 
                    ms_count = mass_spec.atom_counts[idx] 
                    if (s_count != ms_count): 
                        fatal_errors.append( (idx, s_count-ms_count) ) # which atom fatal error, and how much extra in state 
            return res, fatal_errors 
        else: 
            #print("STATE H COUNT:", state.get_H_count()) 
            #print("MS H COUNT:", mass_spec.atom_counts[h_idx]) 
            return state.get_H_count() - mass_spec.atom_counts[h_idx]

    def __init__(self, 
                 mass_spec:MassSpec, 
                 init_state:MolState, 
                 state_ai:MolStateAI, 
                 allowed_atom_types:list=None, 
                 allow_removal:bool=True, 
                 allow_no_modification:bool=True, 
                 allow_bonds_between_rings:bool=True, 
                 allowed_ring_sizes:list=None, 
                 max_steps:int=10, 
                 target_fn=None, 
                 start_counter:int=0, 
                 discount_factor:float=0.9, 
                 start_epsilon:float=params.epsilon_start, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
                 ): 
        '''
        Initializes MDP paramters 
        init_state is a possible initial state to start the molecule object from 
        max_steps is maximum number of steps to run 
        target_fn is function that takes in a MolState and a MassSpec an returns boolean for satisfying criterion 
        other variables are self-explanatory 
        '''

        # initialize variables from parameters 

        self.start_epsilon = start_epsilon 

        self.init_state = init_state 

        self.state_ai = state_ai 
        
        assert mass_spec != None, "Please give a target mass spec."
        self.mass_spec = mass_spec 
        
        if (allowed_atom_types == None): 
            if warn: print("Check, in case, that atom types allowed in representation are entered into Molecule (environment.py) in initialization")
            self.allowed_atom_types = params.atom_types 
        else: 
            self.allowed_atom_types = allowed_atom_types 
        
        self.allow_removal = allow_removal 
        self.allow_no_modification = allow_no_modification 
        self.allow_bonds_between_rings = allow_bonds_between_rings 

        if (allowed_ring_sizes == None): 
            if warn: print("Check, in case, that allowed ring sizes are entered into Molecule (environment.py) in initialization")
            self.allowed_ring_sizes = params.allowed_ring_sizes 
        else: 
            self.allowed_ring_sizes = allowed_ring_sizes 
        
        self.max_steps = max_steps 

        if target_fn==None: 
            def match(state:MolState, mass_spec:MassSpec):  
                '''
                extra, errors = EnvMolecule.compare_state_to_mass_spec(state, mass_spec, True) 
                #print(mass_spec.ftree.nodes[0].formula, ':')
                #print(extra) 
                #print(errors) 
                #state.show_visualization() 

                if (len(errors)>0): 
                    print("ERROR IN ENVMOLECULE WHEN COMPARING STATE AND MASS SPEC; STATE EXTRAS: ")
                    for e in errors: 
                        print(e) 
                
                if extra<0: 
                    print("ERROR: WENT TOO FAR, TOO LESS Hs. TERMINATE! ")
                
                return extra <= 0 
                ''' 
                extra = EnvMolecule.compare_state_to_mass_spec(state, mass_spec, False) 
                
                if extra < 0: 
                    print("ERROR: WENT TOO FAR, TOO LESS Hs. TERMINATE! ") 
                
                return extra <= 0 
            
            
            self.target_fn = match 

        else: self.target_fn = target_fn 

        self.device = device 


        self.discount_factor = discount_factor 


        # initialize run 
        self.reset(start_counter) 
    

    @property 
    def num_steps_taken(self): 
        return self.counter 
    
    def get_path(self): 
        return self.paths_tree, self.paths_parents, self.paths_prev_actions 
    
    def reset(self, start_counter): 
        # tree of paths taken (represented with nodes and indices of parents, with previous actions to trace each path by states and actions) 
        self.paths_tree = [copy.deepcopy(self.init_state)] # type: MolState 
        self.paths_parents = [0] # type: int 
        self.paths_actions_tried = [[]] # type: list[Action] 
        self.paths_prev_actions = [None] # type: Action 
        #self.paths_prev_actions_results = [None] # type: Tensor, that will be put into a loss then backpropagated 
        self.paths_counters = [start_counter] # type: int 
        self.paths_depths = [0] 
        
        self.epsilon = self.start_epsilon 
    
    def set_epsilon(self, epsilon): 
        self.epsilon = epsilon 

    def termination_condition(self, state:MolState): # whether should stop taking actions, irrespective of whether max step count was reached 
        return self.target_fn(state, self.mass_spec) 
    
    def can_take_action(self, state_idx:int): 
        terminated = (self.paths_counters[state_idx] >= self.max_steps) 
        terminated = ( terminated or (self.termination_condition(self.paths_tree[state_idx])) ) 
        return (not terminated) 
    
    def _take_action(self, new_state:MolState=None, action:Action=None, curr_state_idx:int=None, state:MolState=None, save_to_paths_tree=True): 
        # performs action at state, assuming that action is a valid action 
        
        # must have state_idx and action for recording path 
        assert curr_state_idx!=None, "ERROR IN (environment.py): MUST SPECIFY STATE INDEX TO PUT ACTION UNDER IF RECORDING PATH!" 
        assert action!=None, "ERROR IN (environment.py): MUST SPECIFY ACTION TAKEN TO REACH new STATE" 
        if state==None: state = self.paths_tree[curr_state_idx] 
        
        if new_state == None: 
            # must have at least either new state or action taken 
            #assert action != None, "ERROR IN (environment.py): MUST SPECIFY NEW STATE OR ACTION TAKEN" 
            # define new state as after taking action 
            new_state = state.take_action(action) 


        # check that action can/should be performed 
        assert  self.can_take_action(curr_state_idx) , "This episode is already terminated, cannot continue taking actions." 
        # (assumes action is legal) 

        if (save_to_paths_tree): 
            # save to paths tree 
            self.paths_tree.append(copy.deepcopy(new_state)) 
            self.paths_parents.append(curr_state_idx) 
            self.paths_actions_tried.append([]) 
            self.paths_prev_actions.append(copy.deepcopy(action))
            #self.paths_prev_actions_results.append(uvfa_result) 
            self.paths_counters.append(self.paths_counters[curr_state_idx]+1) 
            self.paths_depths.append(self.paths_depths[curr_state_idx]+1) 
            
            self.paths_actions_tried[curr_state_idx].append(copy.deepcopy(action)) 

            return len(self.paths_tree)-1 # this returns the index of the new state 

        '''
        terminated = False 
        if (self.record_path): terminated = (self.paths_counters[state_idx] >= self.max_steps) 
        terminated = ( terminated or (self.termination_condition(new_state)) ) 
        return Result(state=new_state, reward=self.reward(state), terminated=terminated )
        '''

    def get_state_visualizatIon(self, state:MolState): 
        # TODO: SOMEHOW RETURN A VISUALIZATION OF THIS STATE 
        return None 
    
    '''
    def save_uvfa(self, path): 
        torch.save(self.uvfa.state_dict(), path) 
    
    def load_uvfa_from(self, path): 
        self.uvfa = UVFA() 
        self.uvfa.load_state_dict(torch.load(path)) 
        self.uvfa.eval() 
    '''
