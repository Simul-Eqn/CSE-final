import os
os.environ['DGLBACKEND'] = 'pytorch'

import copy 

import torch 
import torch.nn as nn 
from rdkit import Chem 

import numpy as np 

import params 
from representations import Action, MolState, MassSpec 
from environment import EnvMolecule 

import random 

random.seed(10) 

replay_size = params.replay_buffer_size 

class AgentMolecule(EnvMolecule): #

    # set the loss functions 
    mseloss = nn.MSELoss() 
    def rmseloss(pred, labels): 
        return torch.sqrt(MassSpec.mseloss(pred, labels)) 

    '''
    
    def Q(self, action:Action, state:MolState=None, state_idx:int=None, return_new_state=False): # assumes state and state_idx are validated already 
        # can choose to enter not NoneType as state to save processing time if needed 
        # in this, each action only has one unique new state, so return new state is possible. 
        
        # Q function 

        if (type(state)==None): state = self.paths_tree[state_idx] 
        
        new_state = state.take_action(action) 

        # assumes uvfa is already initialized in utils 
        q = ( self.get_reward(state) + self.discount_factor*self.uvfa(torch.cat( [self.mass_spec.toTensor(self.device), new_state.toTensor(self.device)] )) )

        if (not return_new_state): return q 
        else: return q , new_state 

    ''' 
    
    def Q(self, state:MolState, depth, res_device=torch.device("cpu")): 
        # assume validated already 
        #print("START OF Q:",state.graph) 
        #print('start of Q')
        return self.state_ai.get_score(self.mass_spec, state, torch.tensor([depth], device=self.device)).to(res_device) 
        #return self.uvfa(torch.cat( [self.mass_spec.toTensor(self.device), state.toTensor(self.device)] )) # since no more get_reward 
        # return ( self.get_reward(state) + self.discount_factor * self.uvfa(torch.cat( [self.mass_spec.toTensor(self.device), new_state.toTensor(self.device)] )) ) 
        # NOTE: the uvfa aims to approximate Q function for new_state i think? When backpropagating loss for UVFA, should be like that. 

    '''
    def _get_best_action(self, state:MolState=None, state_idx:int=None): # assumes state and state_idx are validated 
        # note that state can be None as long as state_idx is entered 
        highest_value = 0 
        best_action = None 
        best_state = None 
        if (type(state)==None): state = self.paths_tree[state_idx] 
        for action in state.get_valid_actions(): 
            new_state = state.take_action(action) 
            value = self.Q(state, new_state) 
            if value > highest_value: 
                highest_value = value 
                best_action = action 
                best_state = new_state 
        
        return best_action, best_state 

    def take_best_action(self, state:MolState=None, state_idx:int=None): # assumes they are validated 
        action, new_state = self._get_best_action(state, state_idx) 
        return action, self._take_action(new_state, action, state, state_idx) # return format: action, state_idx (index in paths tree) 
    ''' 

    def _get_best_pair(self, stateIdx, return_result=False): # best pair will have lowest score, as lowest score means most likely to be correct :) 
        # get action-new_state pairs above a certain threshold of scores 

        state = self.paths_tree[stateIdx] 

        new_states, actions = state.get_next_states(EnvMolecule.compare_state_to_mass_spec(state, self.mass_spec, False), self.paths_actions_tried[stateIdx], return_actions=True) 
        
        new_state = new_states[0] 
        result = self.Q(new_state, torch.tensor([self.paths_depths[stateIdx]], device=self.device)) 
        value = result.item() 
        lowest_pair = (actions[0], new_state) 
        lowest_value = value # lowest means least anomalous, so most likely to be correct :) 
        if return_result: best_result = value 

        for i in range(1, len(new_states)): 
            new_state = new_states[i] 
            result = self.Q(new_state, torch.tensor([self.paths_depths[stateIdx]], device=self.device)) 
            value = result.item() 
            if (value < lowest_value): 
                lowest_pair = (actions[i], new_state) 
                lowest_value = value 
                if return_result: best_result = value 
        
        if return_result: return lowest_pair, best_result 
        else: return lowest_pair 
    
    def take_best_action(self, state_idx:int, return_result=False): 
        if return_result: 
            pair, result = self._get_best_pair(state_idx, True) 
            action, new_state = pair 
            return result, action, self._take_action(new_state, action, state_idx, True) 
        
        # else 
        action, new_state = self._get_best_pair(state_idx, False) 
        return action, self._take_action(new_state, action, state_idx, True) 
    
    def epsilon_greedy_take_best_action(self, state_idx:int, epsilon=None, return_result=False): 
        if epsilon==None: epsilon=self.epsilon 
        if random.random() > epsilon: 
            return self.take_best_action(state_idx, return_result) 
        else: 
            pairs = self._get_pairs_below_threshold(state_idx, 0.0, False) 
            pair = pairs[random.randrange(0, len(pairs))] 
            if return_result: 
                return (self.Q(pair[1], torch.tensor([self.paths_depths[state_idx]+1], device=self.device)), pair[0], self._take_action(pair[1], pair[0], state_idx, True))  # _take_action 
            else: 
                return (pair[0], self._take_action(pair[1], pair[0], state_idx, True))  # _take_action 
    
    def get_all_pairs(self, stateIdx): 
        # get all action-new_state pairs allowed now 

        state = self.paths_tree[stateIdx] 

        new_states, actions = state.get_next_states(EnvMolecule.compare_state_to_mass_spec(state, self.mass_spec, False), self.paths_actions_tried[stateIdx],  return_actions=True) 
        
        pairs = [] 
        for i in range(len(new_states)): 
            pairs.append((actions[i], new_states[i])) 
        
        return pairs 
    
    def get_all_potential_pairs(self, stateIdx): 
        # get all action-new_state pairs, even if already taken 

        state = self.paths_tree[stateIdx] 

        new_states, actions = state.get_next_states(EnvMolecule.compare_state_to_mass_spec(state, self.mass_spec, False), [],  return_actions=True) 
        
        pairs = [] 
        for i in range(len(new_states)): 
            pairs.append((actions[i], new_states[i])) 
        
        return pairs 
    
    
    def _get_pairs_below_threshold(self, stateIdx, threshold=0.7, return_results=False): # + results 
        # get action-new_state pairs above a certain threshold of scores 

        state = self.paths_tree[stateIdx] 
        depth = self.paths_depths[stateIdx]+1 

        pairs = [] 
        if return_results: results = [] 
        new_states, actions = state.get_next_states(EnvMolecule.compare_state_to_mass_spec(state, self.mass_spec, False), self.paths_actions_tried[stateIdx],  return_actions=True) 
        #print("GOT NEXT STATES ALR")
        #print(new_states) 
        for i in range(len(new_states)): 
            #print("NEW POSSIBLE STATE:",new_states[i]) 
            new_state = new_states[i] 

            #new_state.show_visualization() 

            result = self.Q(new_state, torch.tensor([depth], device=self.device)) 
            value = result.item() 
            #print('after q')
            #print("SCORE:",value) # hehe 
            if (value < threshold): # <, because lower score means better 
                pairs.append((actions[i], new_state))
                if return_results: results.append(value) 
        #print("RETURNING")
        if return_results: return pairs, results 
        else: return pairs 


    def get_k_random_pairs(self, stateIdx, k, return_results=False): # + results 
        # get action-new_state pairs above a certain threshold of scores 

        state = self.paths_tree[stateIdx] 

        pairs = [] 
        if return_results: results = [] 
        new_states, actions = state.get_next_states(EnvMolecule.compare_state_to_mass_spec(state, self.mass_spec, False), self.paths_actions_tried[stateIdx],  return_actions=True) 
        #print("GOT NEXT STATES ALR")
        #print(new_states) 
        for i in range(len(new_states)): 
            new_state = new_states[i] 

            value = random.uniform(0,1) # give random value to each 
            pairs.append((actions[i], new_state))
            if return_results: results.append(value) 
        

        results = np.array(results) 
        order = np.argsort(results) 
        if len(order)>k and k>0: order = order[:k] # so that k=-1 will actually give all pairs 

        returned_pairs = [] 
        for idx in order: 
            returned_pairs.append(pairs[idx]) 

        if return_results: 
            return results[order], returned_pairs 
        else: 
            return returned_pairs 


    
    def take_actions_below_threshold(self, state_idx:int, threshold=0.7, return_results=False): # assumes state, state_idx are validated 
        if return_results: 
            pairs, results = self._get_pairs_below_threshold(state_idx, threshold, True) 
            record = [] 

            for i in range(len(pairs)): # pair in pairs: 
                record.append((results[i].item(), pairs[i][0], self._take_action(pairs[i][1], pairs[i][0], state_idx, True))) # _take_action 

            return record 
        else: 
            pairs = self._get_pairs_below_threshold(state_idx, threshold, False) 
            record = [] 

            for i in range(len(pairs)): # pair in pairs: 
                record.append((pairs[i][0], self._take_action(pairs[i][1], pairs[i][0], state_idx, True))) # _take_action 

            return record 
    
        '''
        for i in range(len(state.get_valid_actions())): 
            action = state.get_valid_actions()[i] 
            value, new_state = self.evaluate_action(action, state, state_idx) 
            if (value > threshold): 
                record.append((action, self._take_action(new_state, action, state, state_idx))) 
        
        return record 
        '''
    
    def get_top_k_pairs(self, state_idx:int, k:int=2, return_results=False): 
        pairs, results = self._get_pairs_below_threshold(state_idx, 2, True) # use this instead of get all pairs, to calculate results 
    
        results = np.array(results) 

        order = np.argsort(results) 
        if len(order)>k and k>0: order = order[:k] # so that k=-1 will actually give all pairs 

        returned_pairs = [] 
        for idx in order: 
            returned_pairs.append(pairs[idx]) 

        if return_results: 
            return results[order], returned_pairs 
        else: 
            return returned_pairs 

    def take_top_k_actions(self, state_idx:int, k:int=3, return_results=False): # return_results will have each output the Q value output too 
        pairs, results = self._get_pairs_below_threshold(state_idx, 2, True)
        
        #print("GOT PAIRS ALREADY") 

        order = np.argsort(results) 
        if len(order)>k and k>0: order = order[:k] 
        # if <k actions, take less than k actions. 
        
        #print("ARGSORTED LAREDADY")

        record = [] 
        for i in order: # pair in pairs: 
            if (return_results): 
                record.append((results[i], pairs[i][0], self._take_action(pairs[i][1], pairs[i][0], state_idx, True))) # _take_action 
            else: 
                record.append((pairs[i][0], self._take_action(pairs[i][1], pairs[i][0], state_idx, True))) # _take_action 

        return record 
    
    def reward_trajectory(self, state_idx:int, score=1): 
        if state_idx<=0: return # no checking reward for initial state 
        #print("REWARD TRAJECTORY:", state_idx, ':', score) 

        '''
        # rewards an entire trajectory, training the uvfa 
        output = self.Q(self.paths_tree[state_idx]) 
        loss = AgentMolecule.rmseloss(output, torch.Tensor([score]))
        self.uvfa_optimizer.zero_grad() 
        loss.backward() 

        # go to previous node in trajectory 
        if self.paths_parents[state_idx] != state_idx: 
            self.reward_trajectory(self.paths_parents[state_idx], score*params.discount_factor) 
        '''
        # rewards an entire trajectory, training MolStateAI per case 
        output = self.Q(self.paths_tree[state_idx], torch.tensor([self.paths_depths[state_idx]], device=self.device), res_device=self.device) 
        loss = AgentMolecule.rmseloss(output, torch.Tensor([[score]]).to(self.device)) 
        self.state_ai.learn_loss(loss) 
        
        # go to previous node in trajectory 
        if self.paths_parents[state_idx] != state_idx: 
            self.reward_trajectory(self.paths_parents[state_idx], score*params.discount_factor) 
    
    def learn_state_score(self, state_idx:int, score=1): 
        output = self.Q(self.paths_tree[state_idx], torch.tensor([self.paths_depths[state_idx]], device=self.device), res_device=self.device) 
        loss = AgentMolecule.rmseloss(output, torch.Tensor([[score]]).to(self.device)) 
        self.state_ai.learn_loss(loss) 
    
    # TODO: hindsight experience replay version of learning 
        



