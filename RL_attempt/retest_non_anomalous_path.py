chosen_mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_without_pooling/search_3e-07_1e-06/models/mass_spec_training/FTreeGCN_training_epoch_20.pt' 
focal_gamma = 0.0 
# 0.0 means not focal loss 

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




# train non-anomalous path for max 12, filtered 
print("\nNON ANOMALOUS PATH RETESTING (max 12, filtered to only have non-aronmatic molecules or those with one benzene ring):")
import non_anomalous_path 
#non_anomalous_path.torch.manual_seed(seed) 
non_anomalous_path.device = device 

non_anomalous_path.init(chosen_mass_spec_gcn_path, "", 85, 5, -0.0, 0.0005, -0.0, 0.95, True, focal_gamma, 12) 

# search possible gcn_lr and nu 
for gcn_lr in [5e-04]: 
    for nu in [0.1]: 
        print("SEARCHING: GCN_LR:",gcn_lr,"nu:", nu)
        non_anomalous_path.path_prefix = './RL_attempt/non_anomalous_grid_search_max_12_filtered_0_1/search_'+str(gcn_lr)+"_"+str(nu) 
        non_anomalous_path.gcn_lr = gcn_lr 
        non_anomalous_path.nu = nu 
        for epoch in range(5, non_anomalous_path.num_epochs+1, non_anomalous_path.test_epoch_interval): 
            norm, anom = non_anomalous_path.test(epoch, "") 
            fout = open(non_anomalous_path.path_prefix+"/test_scores.txt", 'a+') 
            fout.write(str(norm)) # normal 
            fout.write("\n") 
            fout.write(str(anom)) # anomalous 
            fout.write("\n\n")
            fout.close()  

del non_anomalous_path 





# reset seed, resetting random number generator 
seed = 10 

dgl.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
random.seed(seed) 


# train non-anomalous path for max 12, unfiltered 
print("\n\n\nNON ANOMALOUS PATH RETESTING (max 12):")
import non_anomalous_path 
#non_anomalous_path.torch.manual_seed(seed) 
non_anomalous_path.device = device 

non_anomalous_path.init(chosen_mass_spec_gcn_path, "", 100, 5, -0.0, 0.0005, -0.0, 0.95, False, focal_gamma, 12)

# search possible gcn_lr and nu 
for gcn_lr in [5e-04]: 
    for nu in [0.1]: 
        print("SEARCHING: GCN_LR:",gcn_lr,"nu:", nu)
        non_anomalous_path.path_prefix = './RL_attempt/non_anomalous_grid_search_max_12/search_'+str(gcn_lr)+"_"+str(nu) 
        non_anomalous_path.gcn_lr = gcn_lr 
        non_anomalous_path.nu = nu 
        for epoch in range(5, non_anomalous_path.num_epochs+1, non_anomalous_path.test_epoch_interval): 
            norm, anom = non_anomalous_path.test(epoch, "") 
            fout = open(non_anomalous_path.path_prefix+"/test_scores.txt", 'a+') 
            fout.write(str(norm)) # normal 
            fout.write("\n") 
            fout.write(str(anom)) # anomalous 
            fout.write("\n\n")
            fout.close()  

del non_anomalous_path 



