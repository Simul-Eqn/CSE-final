


# search possible gcn_lr and nu 
for gcn_lr in [5e-06, 8e-06, 2e-05]: 
    for nu in [0.1, 0.2, 0.3]: 
        path_prefix = './RL_attempt/non_anomalous_grid_search/search_'+str(gcn_lr)+"_"+str(nu) 

        for epoch in range(5, 81, 5): 
            f = open(path_prefix+"/HypersphereParams_epoch_"+str(epoch)+".txt", 'r') 
            data = f.readlines() 
            f.close() 

            # process data 
            tensor = ''.join([i.strip() for i in data[:-1]]) 
            tensor = "torch.T" + tensor[1:] 
            #tensor = data[0].strip().replace("torch.Tensor", "torch.tensor") 

            radius = data[-1] 

            #print(tensor) 
            #print(radius) 

            f = open(path_prefix+"/HypersphereParams_epoch_"+str(epoch)+".txt", 'w') 
            f.write(tensor) 
            f.write("\n") 
            f.write(radius) 
            #f.write('\n') 
            f.close() 

        

