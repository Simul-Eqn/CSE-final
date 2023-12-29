import os 

import matplotlib.pyplot as plt 
import numpy as np 

colours = ['lime', 'mediumblue', 'orange', 'gold', 'silver'] 
random_colour = "red" 


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

            fin = open(path_prefix+"/valid_astar_depth_top_"+str(k)+"_results_try"+str(num_guesses_per_state)+".txt", 'r') 
            raw_data = fin.readlines() 
            fin.close() 

            data = {} 
            for i in range(len(raw_data)//2): 
                t = raw_data[i*2].strip()[:-1] 
                idx = t.index(' ') 
                d = int(t[:idx])
                name = t[idx+1:] 
                if name in data: 
                    data[name][d] = eval(raw_data[i*2+1]) 
                else: 
                    data[name] = {d: eval(raw_data[i*2+1])}  
            
            #print(data)


            plt.figure() 
            plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", validation)") 
            plt.xlabel("Number of guesses allowed") 
            plt.ylabel("Fraction of correct outputs") 
            
            cidx = 0 
            for dkey, dval in data.items():  
                name = dkey 
                label = name 

                for dvkey, dvval in dval.items(): 

                    res = dvval[0] 
                    cumulative = [] 
                    prev = 0 
                    for r in res: 
                        prev += r 
                        cumulative.append(prev) 
                    
                    if label == "RANDOM": 
                        plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label+" "+str(dvkey), color=random_colour) 
                    else: 
                        plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label+" "+str(dvkey), color=colours[cidx]) 
                cidx += 1 

            plt.legend(loc="upper right", ncol=2, fontsize='x-small') 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(num_guesses_per_state)+"_"+str(nu)+"_depth_success_valid.svg") 
            plt.show() 



            plt.figure() 
            plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", validation)") 
            plt.xlabel("Depth") 
            plt.ylabel("Total fraction of correct outputs") 

            for dkey, dval in data.items():  
                name = dkey 
                label = name 

                xs = [] 
                ys = [] 
                for dvkey, dvval in dval.items(): 
                    xs.append(dvkey) 
                    ys.append(sum(dvval[0])) 
                    
                    
                plt.plot(xs, ys, label=label) 

            plt.legend(loc="upper right") 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(num_guesses_per_state)+"_"+str(nu)+"_depth_compare_success_valid.svg") 
            plt.show() 

            

            plt.figure() 
            plt.title("ASTAR SEARCH CORRECT ACTION RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", validation)") 
            plt.xlabel("Depth") 
            plt.ylabel("Fraction of correct actions taken") 

            for dkey, dval in data.items():  
                name = dkey 
                label = name 

                xs = [] 
                ys = [] 

                for dvkey, dvval in dval.items(): 
                    xs.append(dvkey) 
                    ys.append(np.mean(np.array(dvval[1])/np.array(dvval[2]))) 
                    
                plt.plot(xs, ys, label=label) 

            plt.legend(loc="upper right") 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(num_guesses_per_state)+"_"+str(nu)+"_depth_action_valid.svg") 
            plt.show() 




            # plot for tests 

            fin = open(path_prefix+"/test_astar_depth_top_"+str(k)+"_results_try"+str(num_guesses_per_state)+".txt", 'r') 
            raw_data = fin.readlines() 
            fin.close() 

            data = {} 
            for i in range(len(raw_data)//2): 
                t = raw_data[i*2].strip()[:-1] 
                idx = t.index(' ') 
                d = int(t[:idx])
                name = t[idx+1:] 
                if name in data: 
                    data[name][d] = eval(raw_data[i*2+1]) 
                else: 
                    data[name] = {d: eval(raw_data[i*2+1])}  
            

            plt.figure() 
            plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", test)") 
            plt.xlabel("Number of guesses allowed") 
            plt.ylabel("Fraction of correct outputs") 

            cidx = 0 

            for dkey, dval in data.items():  
                name = dkey 
                label = name 
                
                for dvkey, dvval in dval.items(): 

                    res = dvval[0] 
                    cumulative = [] 
                    prev = 0 
                    for r in res: 
                        prev += r 
                        cumulative.append(prev) 
                    
                    if label == "RANDOM": 
                        plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label+" "+str(dvkey), color=random_colour) 
                    else: 
                        plt.plot(list(range(1, len(cumulative) + 1)), cumulative, label=label+" "+str(dvkey), color=colours[cidx]) 
                cidx += 1 

            plt.legend(loc="upper right", ncol=2, fontsize='x-small') 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(num_guesses_per_state)+"_"+str(nu)+"_depth_success_test.svg") 
            plt.show() 



            plt.figure() 
            plt.title("ASTAR SEARCH SUCCESS RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", test)") 
            plt.xlabel("Depth") 
            plt.ylabel("Total fraction of correct outputs") 

            for dkey, dval in data.items():  
                name = dkey 
                label = name 

                xs = [] 
                ys = [] 
                for dvkey, dvval in dval.items(): 
                    xs.append(dvkey) 
                    ys.append(sum(dvval[0])) 
                    
                    
                plt.plot(xs, ys, label=label) 

            plt.legend(loc="upper right") 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(num_guesses_per_state)+"_"+str(nu)+"_depth_compare_success_test.svg") 
            plt.show() 


            

            plt.figure() 
            plt.title("ASTAR SEARCH CORRECT ACTION RATES FOR GCN_LR = "+str(gcn_lr)+", NU = "+str(nu)+" \n("+test_type+", test)") 
            plt.xlabel("Depth") 
            plt.ylabel("Fraction of correct actions taken") 

            for dkey, dval in data.items():  
                name = dkey 
                label = name 

                xs = [] 
                ys = [] 

                for dvkey, dvval in dval.items(): 
                    xs.append(dvkey) 
                    ys.append(np.mean(np.array(dvval[1])/np.array(dvval[2]))) 
                    
                plt.plot(xs, ys, label=label) 

            plt.legend(loc="upper right") 
            plt.savefig(save_path+'/search_'+str(gcn_lr)+"_"+str(num_guesses_per_state)+"_"+str(nu)+"_depth_action_test.svg") 
            plt.show() 


