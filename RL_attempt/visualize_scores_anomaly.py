import matplotlib.pyplot as plt 
import numpy as np 


cases = {
    'max_12_filtered_0_1': {'test_type': 'max_12_filtered_0_1', 'epoch_range': range(5, 101, 5), 'plot_title':'SCORES - MAX 12 \n(either non-aromatic or with one benzene ring only)'}, 
    'max_12': {'test_type': 'max_12', 'epoch_range': range(5, 101, 5), 'plot_title':'SCORES - MAX 12'}, 
    #'max_15': {'test_type': 'max_15', 'epoch_range': range(5, 101, 5), 'plot_title':'SCORES - MAX 15'} 
}

plt.figure() 

bins = np.array(list(range(0, 103, 2)), dtype=np.float32)/100 

for k, v in cases.items(): 
    for gcn_lr in [5e-04]: 
        for nu in [0.1]: 
            with open('./RL_attempt/non_anomalous_grid_search_'+v['test_type']+'/search_'+str(gcn_lr)+"_"+str(nu) +"/test_scores.txt", 'r') as scoresfile: # infile location 
                all_scores = scoresfile.readlines() 

                overall_normals = [] 
                overall_anomalies = [] 

                i = 0 
                for epoch in v['epoch_range']: # loop all epochs 
                    normal = eval(all_scores[i]) 
                    anomalous = eval(all_scores[i+1]) 

                    n = np.array(normal) 
                    #n = n/(n.shape[0]) 
                    a = np.array(anomalous) 
                    #a = a/(a.shape[0]) 
                    print(epoch, n.size, a.size, np.mean(n), np.std(n), np.mean(a), np.std(a))
                    
                    plt.hist(n, label='normal', alpha=0.5, color='green', bins=bins, weights=np.ones_like(n, dtype=np.float32)/(n.size)) 
                    plt.hist(a, label='anomalous', alpha=0.5, color='red', bins=bins, weights=np.ones_like(a, dtype=np.float32)/(a.size)) 
                    plt.legend() 
                    plt.title("GCN_LR "+str(gcn_lr)+" NU "+str(nu)+" EPOCH "+str(epoch)+" "+v['plot_title']) # plot title 
                    plt.savefig("./RL_attempt/figures/non_anomalous_"+v['test_type']+"_scores_visualization/search_"+str(gcn_lr)+"_"+str(nu)+"/epoch_"+str(epoch)+".svg") # save figure location 
                    plt.show() 

                    overall_normals += normal 
                    overall_anomalies += anomalous 

                    i += 3 


