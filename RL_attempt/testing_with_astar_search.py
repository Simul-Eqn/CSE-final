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







save_states = False 
k = 100 
num_guesses_per_state = 3 
test_type = "max_12" # rmbr to change astar_search.py max_num_heavy_atoms based on test_type ---------------- THIS IS VERY IMPT WEIORFHWOEFNLSKNF:FHUEBF:SJKFN:SEIHF:SIENF:SNFEFLSBFSKLIHEB 

filter_away_not_0_1 = False 

epoch_nums = [30, 65] 
gcn_lrs = [5e-04] 
nus = [0.05] 
cannots = [] 


import astar_search 
print("TESTING TOP K WITH ASTAR SEARCH, k =", k, "guesses per state:", num_guesses_per_state) 
print(test_type) 
astar_search.device = device 
# NOTE THAT A MASS SPEC GCN PATH ACTUALLY ALSO HAS TO BE SPECIFIED IN ASTAR_SEARCH.PY ITSELF, BECAUSE OF PREPROCESSING... THOUGH IT PROBABLY DOESNT NEED TO BE THE CORRECT ONE :) 
astar_search.mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_with_pooling/search_3e-07_3e-07/models/mass_spec_training/FTreeGCN_training_epoch_35.pt' #'./RL_attempt/mass_spec_lr_search/search_'+str(pos[0])+'_'+str(pos[1])+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(pos[2])+'.pt' 

astar_search.filter_away_not_0_1 = filter_away_not_0_1 # if False, more test data is avaialble :) 

astar_search.num_guesses_per_state = num_guesses_per_state # comment if necessary --------------------------------------------------------------------------------------------------------------------------------------------- 



# search possible gcn_lr and nu 
for gcn_lr in gcn_lrs: 
    for nu in nus: 
        if (gcn_lr, nu) in cannots: continue # skit because mm . 
        #if gcn_lr == 5e-06 and nu == 0.1: continue # to skip this case as it has alerady been done 
        astar_search.path_prefix = './RL_attempt/non_anomalous_grid_search_'+str(test_type)+'/search_'+str(gcn_lr)+"_"+str(nu) 
        valid_ress = [] 
        ress = [] 
        print() 
        print() 
        print("TESTING: GCN_LR:",gcn_lr,"nu:", nu)

        for epoch in epoch_nums: 
            path = astar_search.path_prefix+"/epoch_"+str(epoch)+"_top_"+str(k)+"_astar_tests_try"+str(astar_search.num_guesses_per_state)+"/" 

            valid_path = astar_search.path_prefix+"/epoch_"+str(epoch)+"_top_"+str(k)+"_astar_valids_try"+str(astar_search.num_guesses_per_state)+"/" 
            try: 
                os.mkdir(path) 
            except: 
                pass # error is that the directory already exists; no need to worry... 

            try: 
                os.mkdir(valid_path) 
            except: 
                pass # error is that the directory already exists; no need to worry... 

            # valid first 
            valid_res = astar_search.valid_top_k(epoch, k, valid_path, save_states) 
            print("VALID PERCENTAGE:", sum(valid_res))

            fout = open(astar_search.path_prefix+"/valid_astar_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
            fout.write("EPOCH "+str(epoch)+": \n") 
            fout.write(str(valid_res)) 
            fout.write("\n") 
            fout.close()


            # then, test 
            res = astar_search.test_top_k(epoch, k, path, save_states) 
            print("PERCENTAGE:", sum(res)) 

            fout = open(astar_search.path_prefix+"/test_astar_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
            fout.write("EPOCH "+str(epoch)+": \n") 
            fout.write(str(res)) 
            fout.write("\n") 
            fout.close()
        
        
        
        # do test on random 
        path = astar_search.path_prefix+"/random_top_"+str(k)+"_astar_tests_try"+str(astar_search.num_guesses_per_state)+"/" 
        valid_path = astar_search.path_prefix+"/random_top_"+str(k)+"_astar_valids_try"+str(astar_search.num_guesses_per_state)+"/" 
        
        try: 
            os.mkdir(path) 
        except: 
            pass # error is that the directory already exists; no need to worry... 

        try: 
            os.mkdir(valid_path) 
        except: 
            pass # error is that the directory already exists; no need to worry... 
        
        
        random_valid = astar_search.random_top_k(True, k, valid_path, save_states) 
        fout = open(astar_search.path_prefix+"/valid_astar_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
        fout.write("RANDOM: \n") 
        fout.write(str(random_valid)) 
        fout.write("\n") 
        fout.close()

        random_test = astar_search.random_top_k(False, k, path, save_states) 
        fout = open(astar_search.path_prefix+"/test_astar_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
        fout.write("RANDOM: \n") 
        fout.write(str(random_test)) 
        fout.write("\n") 
        fout.close()
        


