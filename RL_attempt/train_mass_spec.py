import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl 
import torch 
import random 

import numpy as np 

seed = 10 

dgl.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
random.seed(seed) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("\nMASS SPEC TRAINING:")
import mass_spec_training 
mass_spec_training.device = device 
mass_spec_training.init("", 0.0, 0.0, num_epochs=30, test_epoch_interval=5, with_pooling_func=False) 

# search possible learning rates 
for gcn_lr in [3e-07]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06, 5e-06, 1e-05]: 
        print() 
        print("SEARCHING: GCN_LR:",gcn_lr,"predictor_lr:",predictor_lr)
        mass_spec_training.path_prefix = './RL_attempt/mass_spec_lr_search_without_pooling/search_'+str(gcn_lr)+"_"+str(predictor_lr) 
        mass_spec_training.gcn_lr = gcn_lr 
        mass_spec_training.predictor_lr = predictor_lr 
        mass_spec_training.train() 



# find best mass spec gcn path 
min_loss = 1 
pos = (-1, -1, -1) # (gcn_lr, predictor_lr, epoch#) 
for gcn_lr in [3e-07]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06, 5e-06, 1e-05]: 
        path_prefix = './RL_attempt/mass_spec_lr_search_without_pooling/search_'+str(gcn_lr)+"_"+str(predictor_lr) 

        losses_file = open(path_prefix+"/models/mass_spec_training/losses.txt", 'r') 
        losses = losses_file.readlines() 
        losses_file.close() 

        losses = [float(l) for l in losses] 

        print(losses) # make sure no issues 

        midx = np.argmin(losses)
        if losses[midx] < min_loss: 
            pos = (gcn_lr, predictor_lr, (midx+1)*5) 


best_mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_without_pooling/search_'+str(pos[0])+'_'+str(pos[1])+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(pos[2])+'.pt' 
print("BEST MASS SPEC GCN PATH:", best_mass_spec_gcn_path)

