import os 

import matplotlib.pyplot as plt 
import numpy as np 



k = 100 
num_guesses_per_state = 3 
test_type = "max_12" # rmbr to change astar_search.py max_num_heavy_atoms based on test_type 

gcn_lrs = [5e-04] 
nus = [0.05] 
cannots = [] 


for gcn_lr in gcn_lrs: 
    for nu in nus: 
        if (gcn_lr, nu) in cannots: continue # skit because mm . 
        save_path = './RL_attempt/figures/astar_'+str(test_type)+'_results'
        try: 
            os.mkdir(save_path) 
        except: 
            pass # means directory already exists yay 


        path_prefix = './RL_attempt/non_anomalous_grid_search_'+str(test_type)+'/search_'+str(gcn_lr)+"_"+str(nu) 


        # plot for valids 

        fin = open(path_prefix+"/valid_astar_top_"+str(k)+"_results_try"+str(num_guesses_per_state)+".txt", 'r') 
        data = fin.readlines() 
        fin.close() 

        plt.figure() 
        plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", "+str(num_guesses_per_state)+" actions per state, validation)") 
        plt.xlabel("Number of guesses allowed") 
        plt.ylabel("Fraction of correct outputs") 

        for i in range(len(data)//2):  
            name = data[2*i].strip()[:-1] 
            label = name 

            res = eval(data[2*i + 1]) 
            cumulative = [] 
            prev = 0 
            for r in res: 
                prev += r 
                cumulative.append(prev) 
            
            
            plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label) 

        plt.legend() 
        plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(nu)+"_"+str(num_guesses_per_state)+"_moves_valid.svg") 
        plt.show() 



        # plot for tests 

        fin = open(path_prefix+"/test_astar_top_"+str(k)+"_results_try"+str(num_guesses_per_state)+".txt", 'r') 
        data = fin.readlines() 
        fin.close() 

        plt.figure() 
        plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", "+str(num_guesses_per_state)+" actions per state, test)") 
        plt.xlabel("Number of guesses allowed") 
        plt.ylabel("Fraction of correct outputs") 

        for i in range(len(data)//2): 
            name = data[2*i].strip()[:-1] 
            label = name 

            res = eval(data[2*i + 1]) 
            cumulative = [] 
            prev = 0 
            for r in res: 
                prev += r 
                cumulative.append(prev) 
            
            
            plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label) 

        plt.legend() 
        plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(nu)+"_"+str(num_guesses_per_state)+"_moves_test.svg") 
        plt.show() 


