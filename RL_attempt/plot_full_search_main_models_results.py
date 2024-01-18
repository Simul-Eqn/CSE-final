import plot_astar_results 

cases = {#'max_12_filtered_0_1': [100, 2, "max_12_filtered_0_1", [5e-04], [0.1], []], 
         #'max_12': [100, 2, "max_12", [5e-04], [0.1], []], 
         'max_15': [100, 2, "max_15", [5e-05], [0.1], []], 
         #'max_12_filtered_0_1': [100, 3, "max_12_filtered_0_1", [5e-04], [0.1], []], 
         #'max_12': [100, 3, "max_12", [5e-04], [0.1], []], 
         'max_15': [100, 3, "max_15", [5e-05], [0.1], []], 
         }

for k, v in cases.items(): 
    plot_astar_results.plot_results(*v) 
