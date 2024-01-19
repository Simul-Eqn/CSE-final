mass_spec_gcn_path = './RL_attempt/mass_spec_lr_search_without_pooling/search_3e-07_1e-06/models/mass_spec_training/FTreeGCN_training_epoch_20.pt' 
num_guesses_per_state = 3 



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







# test max 15 
print("\n\n\nNON ANOMALOUS PATH TESTING (max 15):")
save_states = False 
k = 100 

test_type = "max_15" 
filter_away_not_0_1 = False 
max_num_atoms = 15 
epoch_nums = [15] 
gcn_lrs = [5e-05] 
nus = [0.1] 
cannots = [] 
test_random = True 

depth_range = list(range(1,8)) # unfortunately, due to time constraints, this is necessary 


import astar_search 
print("TESTING TOP K WITH ASTAR SEARCH, k =", k, "guesses per state:", num_guesses_per_state) 
print(test_type) 
astar_search.device = device 

astar_search.init(0.5, max_num_atoms, filter_away_not_0_1) 
astar_search.mass_spec_gcn_path = mass_spec_gcn_path 
astar_search.num_guesses_per_state = num_guesses_per_state # comment if necessary --------------------------------------------------------------------------------------------------------------------------------------------- 


# search possible gcn_lr and nu 
for gcn_lr in gcn_lrs: 
    for nu in nus: 
        if (gcn_lr, nu) in cannots: continue # skit because mm . 
        #if gcn_lr == 5e-06 and nu == 0.1: continue # to skip this case as it has alerady been done 
        astar_search.path_prefix = './RL_attempt/non_anomalous_grid_search_'+str(test_type)+'/search_'+str(gcn_lr)+"_"+str(nu) 
        #valid_ress = [] 
        #ress = [] 
        print() 
        print() 
        print("TESTING: GCN_LR:",gcn_lr,"nu:", nu)

        for epoch in epoch_nums: 
            print("EPOCH:",epoch)

            for depth in depth_range: 
                print("DEPTH:", depth)

                path = astar_search.path_prefix+"/epoch_"+str(epoch)+"_depth_"+str(depth)+"_top_"+str(k)+"_astar_tests_try"+str(astar_search.num_guesses_per_state)+"/" 
                valid_path = astar_search.path_prefix+"/epoch_"+str(epoch)+"_depth_"+str(depth)+"_top_"+str(k)+"_astar_valids_try"+str(astar_search.num_guesses_per_state)+"/" 
                try: 
                    os.mkdir(path) 
                except: 
                    pass # error is that the directory already exists; no need to worry... 

                try: 
                    os.mkdir(valid_path) 
                except: 
                    pass # error is that the directory already exists; no need to worry... 

                # valid first 
                valid_res = astar_search.try_top_k_depth(epoch, k, depth, valid_path, save_states, valid=True) 
                print(valid_res[1]) 
                print(valid_res[2]) 
                #print("VALID PERCENTAGE:", sum(valid_ress[-1][0]))

                # save to file 
                fout = open(astar_search.path_prefix+"/valid_astar_depth_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
                fout.write(str(depth)+" EPOCH "+str(epoch)+": \n") 
                fout.write(str(valid_res)) 
                fout.write("\n") 
                fout.close()


                # then, test 
                res = astar_search.try_top_k_depth(epoch, k, depth, path, save_states, valid=False) 
                print(res[1]) 
                print(res[2]) 
                #print("PERCENTAGE:", sum(ress[-1][0])) 

                # save to file 
                fout = open(astar_search.path_prefix+"/test_astar_depth_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
                fout.write(str(depth)+" EPOCH "+str(epoch)+": \n") 
                fout.write(str(res)) 
                fout.write("\n") 
                fout.close()

                print() 
            
            print() 

        
        if test_random: 
            
            # do for random 
            for depth in depth_range: 
                # do test on random 
                path = astar_search.path_prefix+"/random_depth_"+str(depth)+"_top_"+str(k)+"_astar_tests_try"+str(astar_search.num_guesses_per_state)+"/" 
                valid_path = astar_search.path_prefix+"/random_depth_"+str(depth)+"_top_"+str(k)+"_astar_valids_try"+str(astar_search.num_guesses_per_state)+"/" 
                
                try: 
                    os.mkdir(path) 
                except: 
                    pass # error is that the directory already exists; no need to worry... 

                try: 
                    os.mkdir(valid_path) 
                except: 
                    pass # error is that the directory already exists; no need to worry... 
                
                
                random_valid = astar_search.random_top_k_depth(k, depth, valid_path, save_states, valid=True) # last one is target 
                print(random_valid[1]) 
                print(random_valid[2]) 

                # save to file 
                fout = open(astar_search.path_prefix+"/valid_astar_depth_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
                fout.write(str(depth)+" RANDOM: \n") 
                fout.write(str(random_valid[:-1])) 
                fout.write("\n") 
                fout.write(str(depth)+" TARGET: \n") 
                fout.write(str(random_valid[-1])) 
                fout.write("\n")
                fout.close()

                random_test = astar_search.random_top_k_depth(k, depth, path, save_states, valid=False) # last one is target 
                print(random_test[1]) 
                print(random_test[2]) 

                # save to file 
                fout = open(astar_search.path_prefix+"/test_astar_depth_top_"+str(k)+"_results_try"+str(astar_search.num_guesses_per_state)+".txt", 'a+') 
                fout.write(str(depth)+" RANDOM: \n") 
                fout.write(str(random_test[:-1])) 
                fout.write("\n") 
                fout.write(str(depth)+" TARGET: \n") 
                fout.write(str(random_test[-1])) 
                fout.write("\n")
                fout.close()

                print() 


