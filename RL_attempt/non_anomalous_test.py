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


import non_anomalous_path 
print("TESTING WITH ASTAR SEARCH: ") 
non_anomalous_path.device = device 
non_anomalous_path.mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_with_pooling/search_3e-07_3e-07/models/mass_spec_training/FTreeGCN_training_epoch_35.pt' #'./RL_attempt/mass_spec_lr_search/search_'+str(pos[0])+'_'+str(pos[1])+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(pos[2])+'.pt' 
# search possible gcn_lr and nu 
for gcn_lr in [2e-05]: 
    for nu in [0.1, 0.2, 0.3]: 
        #if gcn_lr == 5e-06 and nu == 0.1: continue # to skip this case as it has alerady been done 
        non_anomalous_path.path_prefix = './RL_attempt/non_anomalous_grid_search/search_'+str(gcn_lr)+"_"+str(nu) 
        ress = [] 
        print() 
        print() 
        print("TESTING: GCN_LR:",gcn_lr,"nu:", nu)

        for epoch in range(5, 81, 5): 
            path = non_anomalous_path.path_prefix+"/epoch_"+str(epoch)+"_random_sample_tests/"
            os.mkdir(path) 
            ress.append(non_anomalous_path.test(epoch, path)) # TODO in there: sample states, save states 

        fout = open(non_anomalous_path.path_prefix+"/test_random_sample_results.txt", 'w') 
        for res in ress: 
            fout.write(str(res)) 
            fout.write("\n") 
        fout.close()