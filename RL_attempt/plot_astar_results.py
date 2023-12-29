import os 

import matplotlib.pyplot as plt 
import numpy as np 


def plot_results(k = 100, 
                 num_guesses_per_state = 2, 
                 test_type = "max_12", 
                 gcn_lrs = [5e-04], 
                 nus = [0.05], 
                 cannots = [] ):

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

            actions_names = [] 
            actions_data = [] 

            for i in range(len(data)//2):  
                name = data[2*i].strip()[:-1] 
                label = name 
                actions_names.append(name) 

                raw_res = eval(data[2*i + 1]) 
                res = raw_res[0] 
                actions_data.append(raw_res) 
                cumulative = [] 
                prev = 0 
                for r in res: 
                    prev += r 
                    cumulative.append(prev) 
                
                
                plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label) 

            plt.legend() 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(nu)+"_"+str(num_guesses_per_state)+"_moves_valid.svg") 
            plt.show() 




            plt.figure() 
            plt.title("ASTAR SEARCH CORRECT ACTION RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", "+str(num_guesses_per_state)+" actions per state, validation)") 
            plt.xlabel("Number of guesses allowed") 
            plt.ylabel("Fraction of correct actions taken") 

            for i in range(len(actions_names)): 
                name = actions_names[i] 
                label = name 

                xs = [] 
                ys = [] 

                dvkey = name 
                dvval =  actions_data[i] 

                xs.append(dvkey) 
                ys.append(np.mean(np.array(dvval[1])/np.array(dvval[2]))) 
                    
                plt.plot(xs, ys, label=label) 

            plt.legend(loc="upper right") 
            plt.savefig(save_path+"/search_"+str(gcn_lr)+"_"+str(nu)+"_"+str(num_guesses_per_state)+"_actions_valid.svg")
            plt.show() 





            # plot for tests 

            fin = open(path_prefix+"/test_astar_top_"+str(k)+"_results_try"+str(num_guesses_per_state)+".txt", 'r') 
            data = fin.readlines() 
            fin.close() 

            plt.figure() 
            plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", "+str(num_guesses_per_state)+" actions per state, test)") 
            plt.xlabel("Number of guesses allowed") 
            plt.ylabel("Fraction of correct outputs") 

            actions_names = [] 
            actions_data = [] 

            for i in range(len(data)//2): 
                name = data[2*i].strip()[:-1] 
                label = name 
                actions_names.append(name) 

                raw_res = eval(data[2*i + 1]) 
                res = raw_res[0] 
                actions_data.append(raw_res) 
                cumulative = [] 
                prev = 0 
                for r in res: 
                    prev += r 
                    cumulative.append(prev) 
                
                
                plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label) 

            plt.legend() 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(nu)+"_"+str(num_guesses_per_state)+"_moves_test.svg") 
            plt.show() 
        





        plt.figure() 
        plt.title("ASTAR SEARCH CORRECT ACTION RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", "+str(num_guesses_per_state)+" actions per state, test)") 
        plt.xlabel("Number of guesses allowed") 
        plt.ylabel("Fraction of correct actions taken") 

        for i in range(len(actions_names)): 
            name = actions_names[i] 
            label = name 

            xs = [] 
            ys = [] 

            dvkey = name 
            dvval =  actions_data[i] 

            xs.append(dvkey) 
            ys.append(np.mean(np.array(dvval[1])/np.array(dvval[2]))) 
                
            plt.plot(xs, ys, label=label) 

        plt.legend(loc="upper right") 
        plt.savefig(save_path+"/search_"+str(gcn_lr)+"_"+str(nu)+"_"+str(num_guesses_per_state)+"_actions_test.svg")
        plt.show() 

