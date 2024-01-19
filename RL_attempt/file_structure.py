# run from IDLE, not VS Code, so os.getcwd() will get inside RL_attempt 

import os 

parent_dir = os.path.join(os.getcwd(), "RL_attempt") 


# make for mass spec lr search 
path1 = os.path.join(parent_dir, "mass_spec_lr_search_without_pooling") 
os.mkdir(path1)
for gcn_lr in [3e-07]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06, 5e-06, 1e-05]: 
            search_path = os.path.join(path1, 'search_'+str(gcn_lr)+'_'+str(predictor_lr))
            os.mkdir(search_path) 

            models_path = os.path.join(search_path, 'models') 
            os.mkdir(models_path) 

            ms_training_path = os.path.join(models_path, "mass_spec_training") 
            os.mkdir(ms_training_path)



for test_type in ['max_12', 'max_12_filtered_0_1', 'max_15']: 
    # make for non-anomalous grid search 
    path1 = os.path.join(parent_dir, "non_anomalous_grid_search_"+test_type) 
    os.mkdir(path1)
    if test_type == "max_15": 
        gcn_lrs = [5e-05] 
    else: 
         gcn_lrs = [5e-04] 
    for gcn_lr in gcn_lrs: 
        for nu in [0.1]: 
            search_path = os.path.join(path1, 'search_'+str(gcn_lr)+"_"+str(nu))
            os.mkdir(search_path) 


fig_path = os.path.join(parent_dir, "figures") 
os.mkdir(fig_path) 

ms_comp_path = os.path.join(fig_path, "mass_spec_comparison") 
os.mkdir(ms_comp_path) 
os.mkdir(os.path.join(ms_comp_path, "train")) 
os.mkdir(os.path.join(ms_comp_path, "test")) 


for test_type in ['max_12_filtered_0_1', 'max_12', 'max_15']: 
    os.mkdir(os.path.join(fig_path, "astar_"+test_type+"_results")) # 2 moves 

    scores_vis_path = os.path.join(fig_path, "non_anomalous_"+test_type+"_scores_visualization") 
    os.mkdir(scores_vis_path) 
    if test_type == "max_15": 
        gcn_lrs = [5e-05] 
    else: 
         gcn_lrs = [5e-04] 
    for gcn_lr in gcn_lrs: 
            for nu in [0.1]: 
                if test_type == "max_15": gcn_lr /= 10 
                os.mkdir(os.path.join(scores_vis_path, "search_"+str(gcn_lr)+"_"+str(nu))) 
