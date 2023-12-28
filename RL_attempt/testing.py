

import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import networkx as nx
import matplotlib.pyplot as plt 



# testing starting graph things 

import utils

#print(utils.FTreeNode.atomTypes)

'''
formula_1 = "C6H12O6"
g1 = utils.get_init_one_benzene_graph(formula_1)

G1 = dgl.to_networkx(g1)
plt.figure()
nx.draw(G1)
plt.show()
'''

formula_2 = "C11H22O"
g2 = utils.get_init_one_benzene_graph(formula_2) 

'''
G2 = dgl.to_networkx(g2)
plt.figure()
nx.draw(G2)
plt.show() 
'''

print(g2.ndata['features'])
print(g2.edata['bondTypes'])

# testing representations MolState show 
import representations 
ai = representations.MolStateAI(True, None, None) 
state = representations.MolState(ai, "", utils.formula_to_atom_counts(formula_2), g2) 
state.show_visualization() 

action = representations.Action(10, 11, 1) 
state2 = state.take_action(action) 
print(state2.graph.ndata['features']) 
print(state2.graph.edata['bondTypes'])
state2.show_visualization() 


