# run from IDLE, not VS Code, so os.getcwd() will get inside RL_attempt 

import os 

parent_dir = os.getcwd() 

'''
# make for mass spec lr search 
path1 = os.path.join(parent_dir, "mass_spec_lr_search") 
os.mkdir(path1)
for gcn_lr in [3e-07, 5e-07, 1e-06]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06]: 
            search_path = os.path.join(path1, 'search_'+str(gcn_lr)+'_'+str(predictor_lr))
            os.mkdir(search_path) 

            models_path = os.path.join(search_path, 'models') 
            os.mkdir(models_path) 

            ms_training_path = os.path.join(models_path, "mass_spec_training") 
            os.mkdir(ms_training_path)


            


# make for imitation learning lr search 
path1 = os.path.join(parent_dir, "imitation_learning_lr_search") 
os.mkdir(path1)
for gcn_lr in [5e-06, 8e-06, 2e-05]: 
    for predictor_lr in [5e-06, 8e-06, 2e-05]: 
            search_path = os.path.join(path1, 'search_'+str(gcn_lr)+'_'+str(predictor_lr))
            os.mkdir(search_path) 

            # models 
            models_path = os.path.join(search_path, 'models') 
            os.mkdir(models_path) 

            imitation_learning_path_1 = os.path.join(models_path, "imitation_learning") 
            os.mkdir(imitation_learning_path_1)

            # states 
            states_path = os.path.join(search_path, 'states') 
            os.mkdir(states_path) 

            imitation_learning_path_2 = os.path.join(states_path, "imitation_learning") 
            os.mkdir(imitation_learning_path_2)




# make for hindsight experience replay lr search 
path1 = os.path.join(parent_dir, "hindsight_experience_replay_lr_search") 
os.mkdir(path1)
for gcn_lr in [5e-06, 8e-06, 2e-05]: 
    for predictor_lr in [5e-06, 8e-06, 2e-05]: 
            search_path = os.path.join(path1, 'search_'+str(gcn_lr)+'_'+str(predictor_lr))
            os.mkdir(search_path) 

            # models 
            models_path = os.path.join(search_path, 'models') 
            os.mkdir(models_path) 

            hindsight_experience_replay_path_1 = os.path.join(models_path, "hindsight_experience_replay") 
            os.mkdir(hindsight_experience_replay_path_1)

            # states 
            states_path = os.path.join(search_path, 'states') 
            os.mkdir(states_path) 

            hindsight_experience_replay_path_2 = os.path.join(states_path, "hindsight_experience_replay") 
            os.mkdir(hindsight_experience_replay_path_2)
'''
'''
            # logs 
            logs_path = os.path.join(search_path, 'logs') 
            os.mkdir(logs_path) 

            hindsight_experience_replay_path_3 = os.path.join(logs_path, "hindsight_experience_replay") 
            os.mkdir(hindsight_experience_replay_path_3)

            # make log files 
            f = open(hindsight_experience_replay_path_3+'/train_log.txt', 'w') 
            f.close() 

            f = open(hindsight_experience_replay_path_3+'/test_log.txt', 'w') 
            f.close() 
''' 


# make for non-anomalous grid search 
path1 = os.path.join(parent_dir, "non_anomalous_grid_search") 
os.mkdir(path1)
for gcn_lr in [5e-04]: 
    for nu in [0.1, 0.15]: 
            search_path = os.path.join(path1, 'search_'+str(gcn_lr)+"_"+str(nu))
            os.mkdir(search_path) 


