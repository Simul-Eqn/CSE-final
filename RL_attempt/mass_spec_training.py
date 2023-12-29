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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
print(device) 


path_prefix = "" 

test_ftrees = [] 
test_labels = [] 
ftrees = [] 
labels = [] 

num_epochs = 0 
test_epoch_interval = 0 

gcn_lr = 0.0 
predictor_lr = 0.0 

with_pooling_func = False 


def init(path_prefix:str, gcn_lr:float,  predictor_lr:float, num_epochs:int=50, test_epoch_interval:int=5, with_pooling_func:bool=False): 

    # load data 


    smiless, ftrees, peakslist = dataloader.get_data('train') 
    labels = [] 
    for peaks in peakslist: 
        labels.append(utils.peaks_to_ms_buckets(peaks, MassSpec.num_buckets, MassSpec.bucket_size)) 


    test_smiless, test_ftrees, test_peakslist = dataloader.get_data('test') 
    test_labels = [] 
    for test_peaks in test_peakslist: 
        test_labels.append(utils.peaks_to_ms_buckets(test_peaks, MassSpec.num_buckets, MassSpec.bucket_size)) 
    
    # update global variables with these parameters 
    globals().update(locals()) 






# test stuff 
def test(epoch): 
    test_mass_spec = MassSpec(True, test_ftrees[0], path_prefix+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(epoch)+'.pt', path_prefix+'/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_'+str(epoch)+'.pt', with_pooling_func=with_pooling_func, device=device)
    total_loss = 0 
    num_tests = 0 
    for i in range(len(test_ftrees)): 

        
        test_mass_spec.run(test_ftrees[i]) 
        embedding = test_mass_spec.pred.to(device) 

        # run the predictor 
        amplitude_res = test_mass_spec.amplitude_predictor(embedding).to(device) 
        # get top_k 
        top_k = torch.argsort(amplitude_res)[-MassSpec.num_peaks_considered:].to(device) 
        
        amplitude_loss = MassSpec.index_select_loss(amplitude_res, test_labels[i].to(device), top_k) 
        total_loss += amplitude_loss.cpu().item() 
        num_tests += 1 
    
    res = total_loss/num_tests 

    print("EPOCH",epoch,"LOSS:", res ) 
    print() 

    return res 





def train(): 

    mass_spec = MassSpec(True, None, path_prefix+'/models/mass_spec_training/FTreeGCN_training_epoch_0.pt', path_prefix+'/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_0.pt', gcn_learning_rate=gcn_lr, predictor_learning_rate=predictor_lr, with_pooling_func=with_pooling_func, device=device) 

    for epoch in range(1, 1+num_epochs): 
        
        train_res = mass_spec.train(ftrees, labels) 
        print(epoch, end=' ') 


        fout = open(path_prefix+"/models/mass_spec_training/train_losses.txt", 'a+') 
        fout.write(str(train_res)) 
        fout.write("\n") 
        fout.close() 


        if epoch%test_epoch_interval == 0: 
            mass_spec.save(path_prefix+'/models/mass_spec_training/FTreeGCN_training_epoch_'+str(epoch)+'.pt') 
            mass_spec.save_amplitude_predictor(path_prefix+'/models/mass_spec_training/FTreeAmplitudePredictor_training_epoch_'+str(epoch)+'.pt')
            res = test(epoch) 
    
            fout = open(path_prefix+"/models/mass_spec_training/losses.txt", 'a+') 
            fout.write(str(res)) 
            fout.write("\n") 
            fout.close() 

    


