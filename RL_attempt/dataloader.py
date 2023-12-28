import os, fnmatch 
os.environ['DGLBACKEND'] = 'pytorch'
import json 
import pandas as pd 

from utils import FTreeNode, FTree
from sys import stdout 

warn = False 
log_in_file = False 
log_file_base_location = "./RL_attempt/logs/dataloader_log_.txt" 


blacklisted_smiles = ['C1=CC=C2C(=C1)C=NC3=CC=CC=C23'] 


def find_treeview_location(name):
    pattern = "*"+name+"*.json" 
    path = "./canopus/treeviews"
    
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name) 
    return None 


def get_data(load="test"): 
    if log_in_file: 
        stdout = open(log_file_base_location[:-4] + load + log_file_base_location[-4:], 'a') 
        import time 
        stdout.write("\n\nGET DATA AT TIME: " + time.asctime() + " \n\n") 

    
    data = pd.read_json("./canopus/"+load+".json") 
    #print(data)
    smiless = [] 
    ftrees = [] 
    peakslist = [] 

    for idx in range(len(data['id_'])):
        # verify that there are no charges 
        smiles = data['smiles'][idx] 
        if (smiles in blacklisted_smiles) or (smiles.find("+]") != -1) or (smiles.find("-]") != -1): 
            if warn:
                stdout.write("SKIPPING TEST "+str(idx)+": CONTAINS CHARGES")
            continue 
        
        # find tree 
        location = find_treeview_location(data["id_"][idx])
        if (location==None):
            if warn:
                stdout.write("SKIPPING TEST "+str(idx)+": CANNOT FIND LOCATION OF TREE FOR "+data['id_'][idx]+"\n")
            continue 
        with open( location , 'r') as json_file: 
            tree = json.load(json_file) 
            
        
        # start adding data 
        smiless.append(smiles) 
        
        # ensure test case contains only elements allowed 
        allowed = True 
        formula = data['formula'][idx] 
        elements = ''.join(f for f in formula if (not f.isdigit())) 
        elements = elements.upper() 
        for e in elements: 
            if e not in FTreeNode.atomTypes: 
                allowed = False 
                break 
        if (not allowed): 
            if warn:
                stdout.write("SKIPPING TEST "+str(idx)+": "+formula+" DUE TO CONTAINING ILLEGAL ELEMENTS\n") 
            smiless.pop() 
            continue 
        
        frags = tree['fragments']
        losses = tree['losses'] 

        # make fragment tree and append 
        nodes = [FTreeNode(frags[0]['molecularFormula'], 0, 0)] 
        for i in range(1, len(frags)): 
            nodes.append(FTreeNode(frags[i]['molecularFormula'], i, losses[i-1]['source'])) 

        ftrees.append(FTree(nodes, 0)) 

        # make peaks list 
        peakslist.append(data['spectrum'][idx]) 
    
    if log_in_file: 
        stdout.close() 
    
    return smiless, ftrees, peakslist 
    

def get_idxth_entry(idx=0, load='test'): 
    data = pd.read_json("./canopus/"+load+".json") 
    #print(data)
    smiless = [] 
    ftrees = [] 
    peakslist = [] 

    # verify that there are no charges 
    smiles = data['smiles'][idx] 
    while (smiles in blacklisted_smiles) or (smiles.find("+]") != -1) or (smiles.find("-]") != -1): 
        if warn:
            stdout.write("SKIPPING TEST "+str(idx)+": CONTAINS CHARGES")
        idx += 1 
        smiles = data['smiles'][idx] 

    smiless.append(data['smiles'][idx]) 
    location = find_treeview_location(data["id_"][idx])
    if (location==None):
        if warn:
            stdout.write("SKIPPING TEST "+str(idx)+": CANNOT FIND LOCATION OF TREE FOR "+data['id_'][idx]+"\n")
        del data 
        del smiless 
        del ftrees 
        del peakslist 
        return get_idxth_entry(idx+1, load)
    
    with open( location , 'r') as json_file: 
        tree = json.load(json_file) 
    
    print("OPENED FILE")
    
    # ensure test case contains only elements allowed 
    allowed = True 
    formula = data['formula'][idx] 
    elements = ''.join(f for f in formula if (not f.isdigit())) 
    elements = elements.upper() 
    for e in elements: 
        if e not in FTreeNode.atomTypes: 
            allowed = False 
            break 
    if (not allowed): 
        if warn:
            stdout.write("SKIPPING TEST "+str(idx)+": "+formula+" DUE TO CONTAINING ILLEGAL ELEMENTS\n") 
        del data 
        del smiless 
        del ftrees 
        del peakslist 
        return get_idxth_entry(idx+1, load)
    
    frags = tree['fragments']
    losses = tree['losses'] 
    
    print("MAKING FTREE")

    # make fragment tree and append 
    nodes = [FTreeNode(frags[0]['molecularFormula'], 0, 0)] 
    for i in range(1, len(frags)): 
        nodes.append(FTreeNode(frags[i]['molecularFormula'], i, losses[i-1]['source'])) 

    ftrees.append(FTree(nodes, 0)) 

    # make peaks list 
    peakslist.append(data['spectrum'][idx]) 
    
    print("DONE")
    
    return smiless, ftrees, peakslist 


#smiless, _, _ = get_data('train')
#print(smiless.index('CC1C(OC(=O)CN(C(=O)C=CC(=C)NC(=O)CNC(=O)C(C1=O)(C)O)C)CCCCCCCCCCCC(C)C')) 

