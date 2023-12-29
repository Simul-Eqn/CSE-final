import os
os.environ['DGLBACKEND'] = 'pytorch'

from rdkit import Chem 
from rdkit.Chem.rdMolDescriptors import CalcMolFormula 
import molmass 

import torch 
import dgl 
import networkx as nx 
import numpy as np 

import copy 

import params 

import matplotlib.pyplot as plt 


# FOR MOLECULE GNN -------------------------------------------------------------------------------------------------------------

# UTILITY FUNCTIONS THAT CAN BE EDITED 

def getAtomData(atom): 
    # old: 
    # atomic number, atomic mass, formal charge, number of hydrogens, is in aromatic, degree 
    #return [atom.GetAtomicNum(), atom.GetMass(), atom.GetFormalCharge(), atom.GetTotalNumHs(), int(atom.GetIsAromatic()), atom.GetDegree()]

    # now: 
    # atomic number, degree, number of Hs left 
    feats = [atom.GetDegree(), atom.GetTotalNumHs()] # not atom.GetImplicitValence(), because that is wrong. 
    for atype in params.atom_types: 
        feats.append(int(atype == atom.GetSymbol())) 
    
    if atom.GetSymbol() == "P": feats[1] += 2 # special case of phosphorus 
    return feats 

atom_n_feats = len(params.atom_types) + 2



# CONSTANTS 

# lookup table for bond types: single, double, triple, aromatic 
bondTypes = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
edgeColors = ['#11ee11', '#1111ee', '#ee0000', '#606060']

bond_n_feats = len(bondTypes)+1 # bondType index and bond energy 

def bondType_idx_to_encoding(bondtype_idx): 
    res = [] 
    for bt_idx in range(len(bondTypes)): 
        if bondtype_idx==bt_idx: res.append(1) 
        else: res.append(0) 

    return res 


# Format of BondDBFile: 
# a,b,btype,energy 
# a and b are atomic numbers of bonds 
# btype is bond type, as above, 0-indexed 
# energy is float in kJ/mol 

# THIS IS TODO, SINCE BOND ENERGY PREDICTION IS NOT FEASIBLE 
bondDBFile = open("./RL_attempt/data/bonds.csv", 'r') 
raw_bondDB = bondDBFile.readlines() 
bondDBFile.close() 

bondDB = {} 
for i in range(len(raw_bondDB)):
    (a, b, btype, energy) = raw_bondDB[i].split(",") 
    bondDB[(int(a), int(b), int(btype))] = float(energy) # since a, b, btype must be integers, dictionary lookup is fine 
    bondDB[(int(b), int(a), int(btype))] = float(energy) # insert the reverse too, for convenience 


def getBE(a, b, btype): 
    '''
    try: 
        return bondDB[(a.GetAtomicNum(), b.GetAtomicNum(), bondTypes.index(btype))] # this is an approximate. 
    except: 
        print("CANNOT GET BOND ENERGY OF "+str(btype)+" BOND BETWEEN "+str(a.GetAtomicNum())+" AND "+str(b.GetAtomicNum())) 
        print("INDICES OF ATOMS: ")
        print(a.GetIdx()) 
        print(b.GetIdx()) 
        return float(input("Manually input bond energy(kJ/mol): ")) 
    ''' 

    # for now, return 0 
    return 0.0 

def getBEfromNum(a, b, btype): 
    '''
    try: 
        return bondDB[(a, b, btype)] 
    except: 
        print("CANNOT GET BOND ENERGY OF "+str(btype)+" BOND BETWEEN "+str(a)+" AND "+str(b)) 
        return float(input("Manually input bond energy(kJ/mol): "))
    ''' 

    # for now, return 0 
    return 0.0 

def get_atomic_num_from_atom_feats(feats): # does not include H 
    i = 0 
    for f in feats[2:]: 
        if (f == 1): 
            return Chem.GetPeriodicTable().GetAtomicNumber(params.atom_types[i]) 
        i += 1 


def SMILEStoMol(smiles): 
    mol = Chem.rdmolfiles.MolFromSmiles(smiles) 
    return mol 

def MoltoGraph(mol): 
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    #if debug: Draw.ShowMol(mol) 

    bondFrom = [] 
    bondTo = [] 
    edgeFeats = [] 

    for b in bonds:
        # get bond data 
        begin = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        bondtype_idx = bondTypes.index(b.GetBondType())

        bond_energy = getBE(b.GetBeginAtom(), b.GetEndAtom(), b.GetBondType()) # get bond energy of bond 

        # add bond 
        bondFrom.append(begin) 
        bondTo.append(end) 
        edgeFeats.append( [bond_energy] + bondType_idx_to_encoding(bondtype_idx) ) 

        # add backwards bond since networkx is directed 
        bondFrom.append(end) 
        bondTo.append(begin) 
        edgeFeats.append([bond_energy] + bondType_idx_to_encoding(bondtype_idx) )

    nodeFeatures = torch.tensor( [getAtomData(atom) for atom in atoms] ) 
    edgeFeats = torch.tensor(edgeFeats) 


    graph = dgl.graph((torch.tensor(bondFrom), torch.tensor(bondTo)), num_nodes=len(atoms), idtype=torch.int32)
    graph.ndata['features'] = nodeFeatures 
    graph.edata['bondTypes'] = edgeFeats 
    #print('features', gnn.ndata['features'])
    #print('bond types', gnn.edata['bondtypes'])
    #gnn = nx.to_undirected(gnn) 

    return graph 

    

def SMILEStoGraph(smiles): 
    mol = Chem.rdmolfiles.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    #if debug: Draw.ShowMol(mol) 

    bondFrom = [] 
    bondTo = [] 
    edgeFeats = [] 

    for b in bonds:
        # get bond data 
        begin = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        bondtype_idx = bondTypes.index(b.GetBondType())

        bond_energy = getBE(b.GetBeginAtom(), b.GetEndAtom(), b.GetBondType()) # get bond energy of bond 

        # add bond 
        bondFrom.append(begin) 
        bondTo.append(end) 
        edgeFeats.append( [bond_energy] + bondType_idx_to_encoding(bondtype_idx) ) 

        # add backwards bond since networkx is directed 
        bondFrom.append(end) 
        bondTo.append(begin) 
        edgeFeats.append( [bond_energy] + bondType_idx_to_encoding(bondtype_idx) )

    nodeFeatures = torch.tensor([getAtomData(atom) for atom in atoms], dtype=torch.float) 
    edgeFeats = torch.tensor(edgeFeats) 


    graph = dgl.graph((torch.tensor(bondFrom), torch.tensor(bondTo)), num_nodes=len(atoms), idtype=torch.int32)
    graph.ndata['features'] = nodeFeatures 
    graph.edata['bondTypes'] = edgeFeats 
    #graph = nx.to_undirected(graph) 

    return graph 


def smiles_to_formula(smiles:str): 
    return CalcMolFormula(Chem.MolFromSmiles(smiles))  


def smiles_to_atom_counts(smiles:str, include_H=True): 
    if include_H: 
        atomtypes = FTreeNode.atomTypes 
    else: 
        atomtypes = params.atom_types 
    smiles = smiles.lower() 
    counts = [] 
    for atom in atomtypes: 
        counts.append(smiles.count(atom.lower())) 
    
    return counts 

def formula_to_atom_counts(formula:str, include_H=True):
    '''
    counts = [0 for _ in range(FTreeNode.n_feats)] 
    i = 0 
    while i < (len(formula)): 
        try: 
            atomType = FTreeNode.atomTypes.index(formula[i]) # TODO: WHAT IF SOME ELEMENTS AREN'T JUST ONE LETTER THOUGH 
        except: 
            # error as atom not in list was found 
            print("ERROR: ATOM "+formula[i]+" NOT FOUND, SKIPPING")
        
        count = 0 
        # read in all the digits behind to numbers 
        while (i+1 < len(formula) and formula[i+1].isdigit()): 
            count *= 10 
            count += int(formula[i+1]) 
            i += 1 
        if count==0: count=1 
        
        # save to counts list
        counts[atomType] = count 
        
        i += 1 
    
    #print(formula+": ")
    #print(counts) 
    # tested to be reliable already :) 
    '''

    if include_H: 
        atomtypes = FTreeNode.atomTypes 
    else: 
        atomtypes = params.atom_types 

    count_series = molmass.Formula(formula).composition().dataframe()['Count'] 
    # make sure no atom not in list was found 
    for atom in count_series.keys(): 
        if atom not in atomtypes: 
            if atom == 'H': continue 
            print("ERROR: ATOM "+str(atom)+" NOT IN ALLOWED LIST, SKIPPING ATOM") 
    
    counts = [] 
    for target in atomtypes: 
        try: 
            counts.append(count_series[target]) 
        except: # no such atom 
            counts.append(0) 

    return counts 

'''
# quick way to prevent duplicates in graph: using some invariants as fast ways to distinguish between them. The only different thing is order of nodes 
# e.g. vertex n-colourability 
def state_get_invariants(state): 
    # Tutte polynomial - using networkx 
    g = dgl.to_networkx(state.graph) 
    return nx.tutte_polynomial(g.to_undirected(as_view=True)) 
'''
# NOTE: instead of using graph invariants, now comparing node feature sequence and edge feature sequence, and if they match, will fully check isomorphism 


def may_be_isomorphic(g1, g2): 
    # note: both are dgl graphs 
    nf1 = g1.ndata['features'].tolist() 
    ef1 = g1.edata['bondTypes'].tolist() 

    nf2 = g2.ndata['features'].tolist() 
    ef2 = g2.edata['bondTypes'].tolist() 

    #print(nf1) 
    #print(nf2) 
    #print(ef1) 
    #print(ef2) 

    nf1.sort() 
    nf2.sort() 
    ef1.sort() 
    ef2.sort() 
    
    #print() 
    #print() 

    #print(nf1) 
    #print(nf2) 
    #print(ef1) 
    #print(ef2) 

    diff = False 
    for i in range(len(nf1)): 
        for j in range(len(nf1[i])): 
            if nf1[i][j] != nf2[i][j]: 
                diff = True 
                #print(i, j, nf1[i][j], nf2[i][j]) 
                break 
    #print(diff) 
    if (diff): return False 
    
    for i in range(len(ef1)): 
        for j in range(len(ef1[i])): 
            if ef1[i][j] != ef2[i][j]: 
                diff = True 
                #print(i, j, ef1[i][j], ef2[i][j]) 
                break 
    
    #print(diff) 

    return (not diff) 

def dgl_to_networkx_isomorphism(g1): 
    G1 = nx.DiGraph(dgl.to_networkx(g1)) 

    #print(G1) 
    #print(type(G1))

    g1_n_attrs = {} 
    for i in range(len(g1.nodes())): 
        g1_n_attrs[i] = {'idx': i} 

    g1_e_attrs = {} 

    for i in range(len(g1.edges()[0])):
        g1_e_attrs[(g1.edges()[0][i].item(), g1.edges()[1][i].item())] = {'idx': i} 
        i += 1 
    
    #print(g1_n_attrs) 
    #print(g1_e_attrs) 

    nx.set_node_attributes(G1, g1_n_attrs) 
    nx.set_edge_attributes(G1, g1_e_attrs)

    return G1 

def is_isomorphic(g1, g2, G1=None, G2=None): 
    # note: both are dgl graphs 
    def node_match(n1, n2): 
        #print("NODE MATCH", n1, n2, ':', g1.ndata['features'][n1['idx']] == g2.ndata['features'][n2['idx']] ) 
        #print() 
        return (g1.ndata['features'][n1['idx']] == g2.ndata['features'][n2['idx']]).all() 
    
    def edge_match(e1, e2): 
        #print("EDGE MATCH", e1, e2, ':', g1.edata['bondTypes'][e1['idx']] == g2.edata['bondTypes'][e2['idx']] ) 
        #print() 
        return (g1.edata['bondTypes'][e1['idx']] == g2.edata['bondTypes'][e2['idx']]).all()
    
    if G1==None: 
        G1 = dgl_to_networkx_isomorphism(g1) 
    if G2==None: 
        G2 = dgl_to_networkx_isomorphism(g2) 
    
    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match) 


def get_init_graph(target_formula): 
    counts = formula_to_atom_counts(target_formula, include_H=False) 
    #counts = FTree.get_atom_counts(target_formula)
    nodeFeatures = [] 
    sum = 0 
    for i in range(len(params.atom_types)): 
        #print("PARAMS ATOM TYPES:", params.atom_types)
        if counts[i] == 0: continue # does not exist 
        #if FTreeNode.atomTypes[i] == "H": continue # don't add Hs 

        # calculate node features 
        feats = [0, 8-Chem.GetPeriodicTable().GetNOuterElecs(params.atom_types[i])] + [int(atype==i) for atype in range(len(params.atom_types))] 
        if params.atom_types[i] == "P": 
            feats[1] += 2 # phosphorus has 2 extra bonds, usually 

        #print(params.atom_types[i]) 
        #print(feats)

        for j in range(counts[i]): 
            nodeFeatures.append(copy.deepcopy(feats))
        
        sum += counts[i] 
    
    graph = dgl.graph((torch.Tensor(),torch.Tensor()), num_nodes = sum, idtype=torch.int32) 
    graph.ndata['features'] = torch.Tensor(nodeFeatures) 
    #graph.edata['bondTypes'] = torch.Tensor(shape=(5,0)) 

    return graph 

def get_init_one_benzene_graph(target_formula): 
    # same as get_init_graph except there is now a benzene ring 
    # verify that formula is okay 
    counts = formula_to_atom_counts(target_formula) 
    #counts = FTree.get_atom_counts(target_formula) 
    assert counts[FTreeNode.atomTypes.index("C")] >= 6, "Not enough carbon atoms!" 
    
    graph = get_init_graph(target_formula) 

    c_idxs = [] 
    carbon_idx = 2+params.atom_types.index("C") 
    for i in range(len(graph.ndata['features'])): 
        if graph.ndata['features'][i][carbon_idx] == 1: 
            c_idxs.append(i) 
            if len(c_idxs) == 6: break 
    
    #assert len(c_idxs)==6, "Not enough carbon atoms!" 
    
    return add_benzene_to_graph(graph, c_idxs) 

def get_init_two_benzene_graph(target_formula): 
    # same as get_init_graph except there is now a benzene ring 
    # verify that formula is okay 
    counts = formula_to_atom_counts(target_formula) 
    #counts = FTree.get_atom_counts(target_formula) 
    assert counts[FTreeNode.atomTypes.index("C")] >= 12, "Not enough carbon atoms!" 
    
    graph = get_init_graph(target_formula) 

    # find positions of carbons to add 
    c_idxs_1 = [] 
    c_idxs_2 = [] 
    carbon_idx = 2+params.atom_types.index("C") 
    for i in range(len(graph.ndata['features'])): 
        if graph.ndata['features'][i][carbon_idx] == 1: 
            if len(c_idxs_1) < 6: 
                c_idxs_1.append(i) 
            elif len(c_idxs_2) < 6: 
                c_idxs_2.append(i) 
                if len(c_idxs_2) == 6: break 
    
    #assert len(c_idxs_2)==6, "Not enough carbon atoms!" 
    
    # add benzenes 
    graph = add_benzene_to_graph(graph, c_idxs_1) 
    graph = add_benzene_to_graph(graph, c_idxs_2) 
    return graph 

def get_init_naphthalene_graph(target_formula): 
    # same as get_init_graph except there is now a benzene ring 
    # verify that formula is okay 
    counts = formula_to_atom_counts(target_formula) 
    # counts = FTree.get_atom_counts(target_formula) 
    #print(counts)
    assert counts[FTreeNode.atomTypes.index("C")] >= 10, "Not enough carbon atoms!" 
    
    graph = get_init_graph(target_formula) 

    c_idxs = [] 
    carbon_idx = 2+params.atom_types.index("C") 
    for i in range(len(graph.ndata['features'])): 
        #print(i)
        if graph.ndata['features'][i][carbon_idx] == 1: 
            c_idxs.append(i) 
            if len(c_idxs) == 10: break 
    
    #assert len(c_idxs)==10, "Not enough carbon atoms!" 
    print(graph.ndata['features']) 
    print(c_idxs)
    
    return add_naphthalene_to_graph(graph, c_idxs) 

def add_benzene_to_graph(graph, froms): 
    tos = froms[1:] 
    tos.append(froms[0]) 
    edata = {"bondTypes": torch.Tensor([ (bondDB[(6, 6, 3)], 0, 0, 0, 1) for _ in range(6)])} 

    graph.add_edges(froms, tos, edata) 
    graph.add_edges(tos, froms, edata) # to make it bidirected 

    # set carbon atom remaining Hs 
    for idx in froms: 
        graph.ndata['features'][idx][1] -= 3 

    return graph 

def add_naphthalene_to_graph(graph, c_idxs): 
    #print(c_idxs) 
    froms = [] 
    tos = [] 

    def add_edge(a, b): 
        #print(a, b) 
        froms.append(a) 
        tos.append(b) 
    
    # add first benzene ring 
    for i in range(6): 
        j = i+1 
        j = j%6 
        add_edge(c_idxs[i], c_idxs[j]) 
        add_edge(c_idxs[j], c_idxs[i]) 
    
    # add second ring to it - essentially adding another 5-carbon ring, except one of the edges is changed (see if j==5)
    for i in range(5): 
        j = i+1 
        if j==5: j=-1 
        i += 5 
        j += 5 
        add_edge(c_idxs[i], c_idxs[j]) 
        add_edge(c_idxs[j], c_idxs[i]) 
    
    edata = {"bondTypes": torch.Tensor([ (bondDB[(6, 6, 3)], 0, 0, 0, 1) for _ in range(len(froms))])} 
    
    graph.add_edges(torch.IntTensor(froms), torch.IntTensor(tos), edata) # already bidirected in add_edge function so. 

    # set carbon atom remaining Hs 
    for i in range(len(c_idxs)): 
        if i == 4 or i == 5: 
            graph.ndata['features'][c_idxs[i]][1] = 0 
        else: graph.ndata['features'][c_idxs[i]][1] = 1 
    
    return graph 

def SMILEStoGraphType(smiles, type:int): 
    formula = smiles_to_formula(smiles) 
    if type == 0: return get_init_graph(formula) 
    if type==1: return get_init_one_benzene_graph(formula) 
    if type==2: return get_init_naphthalene_graph(formula) 
    if type==3: return get_init_two_benzene_graph(formula) 
    return -1 


# FOR FTREE GNN: note that this has directed edges ---------------------------------------------------------------------------------------------------------------

# node features: number of each atom in each node of fragment tree 
class FTreeNode(): 
    #atomTypes = ['C', 'H', 'N', 'O', 'P', 'S'] # this is a lookup table of all atom types, with H, in contrast to params 
    atomTypes = copy.deepcopy(params.atom_types) 
    atomTypes.insert(1, "H") 
    n_feats = len(atomTypes)+1 # + 1 because mass 

    # for visualization purposes 
    atomColours = ['#aaaaaa', '#eeeeee', '#33dd33', '#dd3333', '#dd9300', '#dddd00'] 

    def __init__(self, formula, idx, parentIdx): 
        # read in formula to amount of each atom 
        self.formula = formula 
        self.counts = formula_to_atom_counts(formula)     
        self.idx = idx 
        self.parentIdx = parentIdx 
        self.mass = molmass.Formula(formula).mass 

        self.feats = copy.deepcopy(self.counts) 
        self.feats.append(self.mass)

class FTree(): # container for fragment tree nodes 
    def __init__(self, nodes, root:int): # root is index of node 
        self.nodes = nodes 
        self.n_nodes = len(nodes) # n_nodes will tell you what index to add the next node 
        self.root = root 
    
    def add_node(self, node): 
        self.nodes.append(node) 
        self.n_nodes += 1 
    
    def make_and_add_node(self, formula, parentIdx): 
        self.nodes.append(FTreeNode(formula, self.n_nodes, parentIdx)) 
        self.n_nodes += 1 
    
    '''
    # util 
    def get_atom_counts(formula, atomTypes=FTreeNode.atomTypes): 
        counts = [0 for _ in range(len(atomTypes))] 
        i = 0 
        while i < (len(formula)): 
            atomType = atomTypes.index(formula[i]) 
            count = 0 
            # read in all the digits behind to numbers 
            while (i+1 < len(formula) and formula[i+1].isdigit()): 
                count *= 10 
                count += int(formula[i+1]) 
                i += 1 
            if count==0: count=1 
            
            # save to counts list
            counts[atomType] = count 
            
            i += 1

        return counts   
    '''
        
        
def FTreeToGraph(ftree:FTree):
    bondFrom = [] 
    bondTo = [] 

    nodes = ftree.nodes 

    for n in nodes:
        # check that is not root 
        if n.idx == n.parentIdx: continue 
        
        # add bond 
        bondFrom.append(n.idx) 
        bondTo.append(n.parentIdx) 
    
    nodeFeatures = torch.tensor([n.feats for n in nodes]) 

    graph = dgl.graph((torch.tensor(bondFrom), torch.tensor(bondTo)), num_nodes=len(nodes), idtype=torch.int32)
    graph.ndata['features'] = nodeFeatures 
    return graph 

'''
def peaks_to_label(peaks, predictor_peak_count): # this will be MassSpec.predictor_peak_count 
    # converts peaks from dataloader to labels for MassSpec training use 
    # peaks from dataloader is dictionary with key as m/z and value as percentage 
    amplitude_sorted = sorted(peaks.items(), key = lambda i: -i[1]) # sort based on decreasing peak amplitude 

    label = [] 
    for i in range(predictor_peak_count): 
        label.append((1, amplitude_sorted[i][0], amplitude_sorted[i][1])) 

    rem = predictor_peak_count - len(amplitude_sorted) 
    while (rem > 0): 
        label.append((0,0,0)) 

    return torch.Tensor(np.array(label)) 
''' 
def peaks_to_ms_buckets(peaks, n_buckets, bucket_size): 
    # peaks is list of pairs (m/z, amplitude) 
    # n_buckets will be something in representations 
    # bucket_size will be same units as amplitude in peaks 
    
    label = np.zeros((n_buckets)) 
    #print(label)
    
    for peak_key in peaks: 
        # NOTE: peak is a string 
        try: 
            label[int(float(peak_key)/bucket_size)] = peaks[peak_key]/100 
        except Exception as e: 
            #print(e) 
            #print(int(float(peak)/bucket_size))
            # probaby due to peak being outside of buckets 
            print("WARNING: PEAK AT", peak_key, "FOUND WHEN CONVERTING TO BUCKETS; DOES NOT FIT IN ANY BUCKET") 
            continue 
    
    return torch.Tensor(label) 



def agentmol_save_all(a): 
    # a must be type AgentMolecule 
    a.state_ai.save() 
    try: 
        a.state_ai.save_predictor() 
    except: 
        pass 

    a.mass_spec.save() 
    try: 
        a.mass_spec.save_amplitude_predictor() 
    except: 
        pass 





# getting action space 

class UFDS(): 
    def __init__(self, rem_Hs:list, N:int=-1): 
        if (N==-1): N=len(rem_Hs) 
        self.parents = [i for i in range(N)] 
        self.ranks = [0 for _ in range(N)] 
        self.set_sizes = [1 for _ in range(N)] 
        self.num_sets = N 
        self.set_rem_Hs = rem_Hs 
    
    def find_set(self, i:int): 
        if self.parents[i] != i: 
            self.parents[i] = self.find_set(self.parents[i]) # should not reach max recursion liimt, since small molecules 
        
        return self.parents[i] 
    
    def is_same_set(self, i:int, j:int): 
        return self.find_set(i) == self.find_set(j) 
    
    def get_num_disjoint_sets(self): 
        return self.num_sets 
    
    def get_set_size(self, i:int): 
        return self.set_sizes[self.find_set(i)] 
    
    def union_sets(self, i:int, j:int): 
        x = self.find_set(i) 
        y = self.find_set(j) 
        if (x==y): return 
        
        if (self.ranks[x] > self.ranks[y]): # make sure x is 'shorter' than y (not necessarily correct but, optimization) 
            t = x 
            x = y 
            y = t 
        
        self.parents[x] = y # add x to y 
        
        self.set_rem_Hs[y] += self.set_rem_Hs[x] 
        
        if (self.ranks[x] == self.ranks[y]): self.ranks[y] += 1 # may need to increase depth of y 
        
        self.set_sizes[y] += self.set_sizes[x] 
        
        self.num_sets -= 1 
    
    def get_set_Hs(self, i:int): 
        return self.set_rem_Hs[self.parents[i]] 
            

def get_ufds(graph): # Union-Find Disjoint Set data structure 
    ufds = UFDS(list(map(int, graph.ndata['features'] [:,1])), graph.num_nodes()) 
    
    edge_list = list(graph.edges()) 
    #print(edge_list)
    if (len(edge_list[0])==0): return ufds 
    
    for i in range(len(edge_list[0])): # no longer //2, so as not to assume edge like that. 
        #print("UNION",(edge_list[0][2*i].item(), edge_list[1][2*i].item())) 
        ufds.union_sets(edge_list[0][i].item(), edge_list[1][i].item()) 
    
    return ufds 
    

    
    

def show_agentmol_states(agentmol, itr, k=0.3, draw_edge_labels=False): # show agentmol states 
    pos = nx.spring_layout(dgl.to_networkx(agentmol.paths_tree[0].graph), k=k, iterations=20) 
    
    for i in itr: 
        agentmol.paths_tree[i].show_visualization(pos = pos, title = str(i), draw_edge_labels = draw_edge_labels) 

def show_state_graphs(states, itr, k=0.3, draw_edge_labels=False): 
    pos = nx.spring_layout(dgl.to_networkx(states[0].graph), k=k, iterations=20) 
    
    for i in itr: 
        states[i].show_visualization(pos = pos, title = str(i), draw_edge_labels = draw_edge_labels) 


def trace_reversed_path(agentmol, idx:int): 
    backwards = [idx] 
    while (idx != 0): 
        idx = agentmol.paths_parents[idx] 
        backwards.append(idx) 
    
    return backwards 

def trace_path(agentmol, idx:int): 
    backwards = [idx] 
    while (idx != 0): 
        idx = agentmol.paths_parents[idx] 
        backwards.append(idx) 
    
    path_length = len(backwards) 
    return path_length, [backwards.pop() for _ in range(path_length)] 

def debug_is_valid_action(agentmol, idx, action, EnvMolecule): 
    return agentmol.paths_tree[idx].is_valid_action(get_ufds(agentmol.paths_tree[idx].graph), action.first, action.second, action.type+1, EnvMolecule.compare_state_to_mass_spec(agentmol.paths_tree[idx], agentmol.mass_spec, False)) 
    
    
    
    