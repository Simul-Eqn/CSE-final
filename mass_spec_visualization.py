import sys
sys.path.insert(0, "./RL_attempt/") 

import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl 
import torch 
import torch.nn as nn 
from rdkit import Chem 
from rdkit.Chem.rdMolDescriptors import CalcMolFormula 

from agent import * 
from environment import * 
from representations import * 

import dataloader 
import utils 
import params 

import copy
import random

        
import matplotlib.pyplot as plt


'''
# MAKE GRAPH - TEST LOSSES 
plt.rcParams['figure.figsize'] = [10,5]
plt.figure() 
epoch_nums = [i*5 for i in range(1, 11)] 

for gcn_lr in [3e-07]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06, 5e-06, 1e-05]: 
        path_prefix = './RL_attempt/mass_spec_lr_search/search_'+str(gcn_lr)+"_"+str(predictor_lr) 

        losses_file = open(path_prefix+"/models/mass_spec_training/losses.txt", 'r') 
        losses = losses_file.readlines() 
        losses_file.close() 

        losses = [float(l) for l in losses] 
        
        plt.plot(epoch_nums, losses, label=str(gcn_lr)+", "+str(predictor_lr))  

plt.legend(loc='upper center', bbox_to_anchor=(0.6, 1),ncol=3, fancybox=True, shadow=True, title="LEGEND: [GCN_learning_rate], [predictor_learning_rate]") 
plt.xlabel("Epoch number") 
plt.ylabel("Test loss (rmse)") 
plt.title("Training Fragment Tree GCN and predictor to predict original mass spec") 
plt.show() 
'''

# MAKE GRAPH - TRAIN LOSSES 
plt.rcParams['figure.figsize'] = [10,5]
plt.figure() 
epoch_nums = [i for i in range(1, 51)] 

for gcn_lr in [3e-07]: 
    for predictor_lr in [3e-07, 5e-07, 1e-06, 5e-06, 1e-05]: 
        path_prefix = './RL_attempt/mass_spec_lr_search/search_'+str(gcn_lr)+"_"+str(predictor_lr) 

        losses_file = open(path_prefix+"/models/mass_spec_training/train_losses.txt", 'r') 
        losses = losses_file.readlines() 
        losses_file.close() 

        losses = [float(l) for l in losses] 
        
        plt.plot(epoch_nums[20:], losses[20:], label=str(gcn_lr)+", "+str(predictor_lr))  

plt.legend(loc='upper center', bbox_to_anchor=(0.6, 0.6),ncol=3, fancybox=True, shadow=True, title="LEGEND: [GCN_learning_rate], [predictor_learning_rate]") 
plt.xlabel("Epoch number") 
plt.ylabel("Train loss (rmse)") 
plt.title("Training Fragment Tree GCN and predictor to predict original mass spec") 
plt.show() 



# MAKE FIGURES OF MASS SPEC VISUALLY ----------------------------------------- 
get = "train"


plt.rcParams['figure.figsize'] = [10,8]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# get data 
test_smiless, test_ftrees, test_peakslist = dataloader.get_data(get) 
test_labels = [] 
for test_peaks in test_peakslist: 
    test_labels.append(utils.peaks_to_ms_buckets(test_peaks, MassSpec.num_buckets, MassSpec.bucket_size)) 

# get result 
path = './RL_attempt/mass_spec_lr_search/search_3e-07_3e-07/models/mass_spec_training/FTreeGCN_training_epoch_35.pt'
predictor_path = './RL_attempt/mass_spec_lr_search/search_3e-07_3e-07/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_35.pt'
save_path_prefix = './RL_attempt/figures/mass_spec_comparison/'+get+'/'


ms = MassSpec(True, None, path, predictor_path, gcn_learning_rate=0, predictor_learning_rate = 0, device=device) 

for test_idx in range(len(test_smiless)) : 
    ms.run(test_ftrees[test_idx]) 
    embedding = ms.pred.to(device) 
    amplitude_res = ms.amplitude_predictor(embedding)[0].to(device) 
    top_k = torch.argsort(amplitude_res)[-MassSpec.num_peaks_considered:] 

    filtered_model_res = amplitude_res[top_k].cpu() 
    filtered_labels = test_labels[test_idx].to(device)[top_k].cpu()

    plt.figure() 

    plt.subplot(2,1,1) 
    plt.title("draw all") 
    for bidx in range(MassSpec.num_buckets):
        plt.bar(bidx*MassSpec.bucket_size, amplitude_res[bidx].item(), width=1, align='edge', color="#ff0000")
        plt.bar((bidx+1)*MassSpec.bucket_size, test_labels[test_idx][bidx].item(), width=-1, align='edge', color="#00ff00")

    plt.subplot(2,1,2) 
    plt.title("draw top 30") 
    for bidx in range(30):
        plt.bar((top_k[bidx].item())*MassSpec.bucket_size, filtered_model_res[bidx].item(), width=1, align='edge', color="#ff0000")
        plt.bar((top_k[bidx].item()+1)*MassSpec.bucket_size, filtered_labels[bidx].item(), width=-1, align='edge', color="#00ff00")

    #plt.show()
    plt.savefig(save_path_prefix+test_smiless[test_idx]+".svg")
    plt.close() 








    

