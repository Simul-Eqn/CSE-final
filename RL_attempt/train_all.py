import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl 
import torch 
import random 

import numpy as np 

seed = 10 

dgl.seed(seed) 
torch.manual_seed(seed)
random.seed(seed) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
print("\nMASS SPEC TRAINING:")
import mass_spec_training 
mass_spec_training.device = device 
mass_spec_training.with_pooling_func = False 
# search possible learning rates 
for gcn_lr in [3e-07, 5e-07, 1e-06]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06]: 
        print() 
        print("SEARCHING: GCN_LR:",gcn_lr,"predictor_lr:",predictor_lr)
        mass_spec_training.path_prefix = './RL_attempt/mass_spec_lr_search_without_pooling/search_'+str(gcn_lr)+"_"+str(predictor_lr) 
        mass_spec_training.gcn_lr = gcn_lr 
        mass_spec_training.predictor_lr = predictor_lr 
        mass_spec_training.train() 



# find best mass spec gcn path 
min_loss = 1 
pos = (-1, -1, -1) # (gcn_lr, predictor_lr, epoch#)
for gcn_lr in [8e-05, 0.0001, 0.00015]: 
    for predictor_lr in [8e-05, 0.0001, 0.00015]: 
        path_prefix = './RL_attempt/mass_spec_lr_search/search_'+str(gcn_lr)+"_"+str(predictor_lr) 

        losses_file = open(path_prefix+"/models/mass_spec_training/losses.txt", 'r') 
        losses = losses_file.readlines() 
        losses_file.close() 

        losses = [float(l) for l in losses] 

        print(losses) # make sure no issues 

        midx = np.argmin(losses)
        if losses[midx] < min_loss: 
            pos = (gcn_lr, predictor_lr, (midx+1)*5) 
'''


# TO CHANGE THIS MM 
#best_mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search/search_0.0001_0.0001/models/mass_spec_training/FTreeGCN_training_epoch_45.pt' #'./RL_attempt/mass_spec_lr_search/search_'+str(pos[0])+'_'+str(pos[1])+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(pos[2])+'.pt' 

#print(best_mass_spec_gcn_path) 

"""
print("\nNON ANOMALOUS PATH TRAINING:")
import non_anomalous_path 
#non_anomalous_path.torch.manual_seed(seed) 
non_anomalous_path.device = device 
non_anomalous_path.mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_with_pooling/search_3e-07_3e-07/models/mass_spec_training/FTreeGCN_training_epoch_35.pt' #'./RL_attempt/mass_spec_lr_search/search_'+str(pos[0])+'_'+str(pos[1])+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(pos[2])+'.pt' 
non_anomalous_path.max_num_heavy_atoms = 12 
non_anomalous_path.discount_factor = 1 # don't bother discounting, as heuristic in astar search already considers remianing Hs 
non_anomalous_path.num_epochs = 100 

non_anomalous_path.filter_away_not_0_1 = False 

# search possible gcn_lr and nu 
for gcn_lr in [5e-04]: 
    for nu in [0.05]: 
        #if gcn_lr == 5e-06 and nu == 0.1: continue # to skip this case as it has alerady been done  
        print("SEARCHING: GCN_LR:",gcn_lr,"nu:", nu)
        non_anomalous_path.path_prefix = './RL_attempt/non_anomalous_grid_search_max_12/search_'+str(gcn_lr)+"_"+str(nu) 
        non_anomalous_path.gcn_lr = gcn_lr 
        non_anomalous_path.nu = nu 
        if nu == 0.1: 
            non_anomalous_path.train(60) 
        else: 
            non_anomalous_path.train(0) 

del non_anomalous_path 
"""


# train for up to 15 atoms 
print("\n\n\nUP TO 15 ATOMS TRAINING: ")
import non_anomalous_path 
#non_anomalous_path.torch.manual_seed(seed) 
non_anomalous_path.device = device 
non_anomalous_path_mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_with_pooling/search_3e-07_3e-07/models/mass_spec_training/FTreeGCN_training_epoch_35.pt' #'./RL_attempt/mass_spec_lr_search/search_'+str(pos[0])+'_'+str(pos[1])+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(pos[2])+'.pt' 
non_anomalous_path_max_num_heavy_atoms = 15 
non_anomalous_path_discount_factor = 1 # don't bother discounting, as heuristic in astar search already considers remianing Hs 
non_anomalous_path_num_epochs = 100 

non_anomalous_path_filter_away_not_0_1 = False 

# search possible gcn_lr and nu 
for gcn_lr in [5e-04]: 
    for nu in [0.1]: 
        #if gcn_lr == 5e-06 and nu == 0.1: continue # to skip this case as it has alerady been done  
        print("SEARCHING: GCN_LR:",gcn_lr,"nu:", nu)
        non_anomalous_path.path_prefix = './RL_attempt/non_anomalous_grid_search_max_15/search_'+str(gcn_lr)+"_"+str(nu) 
        non_anomalous_path.gcn_lr = gcn_lr 
        non_anomalous_path.nu = nu 
        non_anomalous_path.train(0) 



'''
print("\nHINDSIGHT EXPERIENCE REPLAY TRAINING:")
import hindsight_experience_replay 
hindsight_experience_replay.device = device 
hindsight_experience_replay.mass_spec_gcn_path = best_mass_spec_gcn_path 
hindsight_experience_replay.types_considered = [0,1] # no benzene, 1 benzene 
hindsight_experience_replay.num_top_guesses = 10 
hindsight_experience_replay.beam_width = 40 

# search possible learning rates 
for gcn_lr in [5e-06, 8e-06, 2e-05]: 
    for predictor_lr in [5e-06, 8e-06, 2e-05]: 
        print() 
        print("SEARCHING: GCN_LR:",gcn_lr,"predictor_lr:",predictor_lr)
        hindsight_experience_replay.path_prefix = './RL_attempt/hindsight_experience_replay_lr_search/search_'+str(gcn_lr)+"_"+str(predictor_lr) 
        hindsight_experience_replay.gcn_lr = gcn_lr 
        hindsight_experience_replay.predictor_lr = predictor_lr 
        hindsight_experience_replay.train() 

'''
