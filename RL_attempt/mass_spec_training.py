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

path_prefix = './RL_attempt'
gcn_lr = 0.00005
predictor_lr = 0.00005 
with_pooling_func = False 



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
print(device)

print("Loading data...")

test_smiless, test_ftrees, test_peakslist = dataloader.get_data('test') 
test_labels = [] 
for test_peaks in test_peakslist: 
    test_labels.append(utils.peaks_to_ms_buckets(test_peaks, MassSpec.num_buckets, MassSpec.bucket_size)) 

# test stuff 
def test(epoch): 
    test_mass_spec = MassSpec(True, test_ftrees[0], path_prefix+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(epoch)+'.pt', path_prefix+'/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_'+str(epoch)+'.pt', with_pooling_func=with_pooling_func, device=device)
    total_loss = 0 
    num_tests = 0 
    for i in range(len(test_ftrees)): 

        
        test_mass_spec.run(ftrees[i]) 
        embedding = test_mass_spec.pred.to(device) 

        # run the predictor 
        amplitude_res = test_mass_spec.amplitude_predictor(embedding).to(device) 
        # get top_k 
        top_k = torch.argsort(amplitude_res)[-MassSpec.num_peaks_considered:].to(device) 
        
        amplitude_loss = MassSpec.index_select_loss(amplitude_res, labels[i].to(device), top_k) 
        total_loss += amplitude_loss.cpu().item() 
        num_tests += 1 
    
    res = total_loss/num_tests 

    print("EPOCH",epoch,"LOSS:", res ) 
    print() 

    return res 




num_epochs = 50 
test_epoch_interval = 5 

# training 
smiless, ftrees, peakslist = dataloader.get_data('train') 

labels = [] 

for peaks in peakslist: 
    labels.append(utils.peaks_to_ms_buckets(peaks, MassSpec.num_buckets, MassSpec.bucket_size)) 


print("Starting")

def train(): 

    mass_spec = MassSpec(True, None, path_prefix+'/models/mass_spec_training/FTreeGCN_training_epoch_0.pt', path_prefix+'/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_0.pt', gcn_learning_rate=gcn_lr, predictor_learning_rate=predictor_lr, with_pooling_func=with_pooling_func, device=device) 

    train_ress = [] 
    ress = [] 

    for epoch in range(1, 1+num_epochs): 
        
        train_ress.append(mass_spec.train(ftrees, labels)) 
        print(epoch, end=' ') 

        if epoch%test_epoch_interval == 0: 
            mass_spec.save(path_prefix+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(epoch)+'.pt') 
            mass_spec.save_amplitude_predictor(path_prefix+'/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_'+str(epoch)+'.pt')
            ress.append(test(epoch)) 
            print("TRAIN:", train_ress[-1])
    
    fout = open(path_prefix+"/models/mass_spec_training/losses.txt", 'w') 
    for res in ress: 
        fout.write(str(res)) 
        fout.write("\n") 
    fout.close() 

    fout = open(path_prefix+"/models/mass_spec_training/train_losses.txt", 'w') 
    for train_res in train_ress: 
        fout.write(str(train_res)) 
        fout.write("\n") 
    fout.close() 


