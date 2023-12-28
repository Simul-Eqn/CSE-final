import matplotlib.pyplot as plt 
import numpy as np 




plt.figure() 

bins = np.array(list(range(0, 103, 2)), dtype=np.float32)/100 

for gcn_lr in [5e-04]: 
    for nu in [0.1]: 
        with open('./RL_attempt/non_anomalous_grid_search_max_12_filtered_0_1/search_'+str(gcn_lr)+"_"+str(nu) +"/test_scores.txt", 'r') as scoresfile: # infile location 
            all_scores = scoresfile.readlines() 

            overall_normals = [] 
            overall_anomalies = [] 

            i = 0 
            for epoch in range(5, 101, 5): # loop all epochs 
                normal = eval(all_scores[i]) 
                anomalous = eval(all_scores[i+1]) 

                n = np.array(normal) 
                #n = n/(n.shape[0]) 
                a = np.array(anomalous) 
                #a = a/(a.shape[0]) 
                print(epoch, n.size, a.size, max(n), max(a))
                
                plt.hist(n, label='normal', alpha=0.5, color='green', bins=bins, weights=np.ones_like(n, dtype=np.float32)/(n.size)) 
                plt.hist(a, label='anomalous', alpha=0.5, color='red', bins=bins, weights=np.ones_like(a, dtype=np.float32)/(a.size)) 
                plt.title("GCN_LR "+str(gcn_lr)+" NU "+str(nu)+" EPOCH "+str(epoch)+" SCORES - MAX 12 \n(either non-aromatic or with one benzene ring only)") # plot title 
                plt.savefig("./RL_attempt/figures/non_anomalous_max_12_filtered_0_1_scores_visualization/search_"+str(gcn_lr)+"_"+str(nu)+"/epoch_"+str(epoch)+".svg") # save figure location 
                plt.show() 

                overall_normals += normal 
                overall_anomalies += anomalous 

                i += 3 

            """
            # commented because this seemed useless 
            n = np.array(overall_normals) 
            #n = n/(n.shape[0]) 
            a = np.array(overall_anomalies) 
            #a = a/(a.shape[0]) 
            plt.hist(n, label='normal', alpha=0.5, color='green', bins=bins, weights=np.ones_like(n, dtype=np.float32)/(n.size)) 
            plt.hist(a, label='anomalous', alpha=0.5, color='red', bins=bins, weights=np.ones_like(a, dtype=np.float32)/(a.size)) 
            plt.title("GCN_LR "+str(gcn_lr)+" NU "+str(nu)+" OVERALL SCORES ") 
            plt.show() 
            plt.savefig("./RL_attempt/figures/non_anomalous_scores_visualization/search_"+str(gcn_lr)+"_"+str(nu)+"/overall.svg")
            """

