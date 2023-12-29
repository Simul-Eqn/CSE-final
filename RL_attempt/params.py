# inspired by MolDQN-pytorch 

start_molecule = None

# learning params 
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000

# not used, just assumed 
optimizer = "Adam"

# tused params 
atom_types = ["C", "N", "O", "P", "S"] # ignoring H in this, so this is atom types allowed inside the state 
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
allowed_ring_sizes = [3, 4, 5, 6]
max_steps_per_episode = 40 # except this 

# more learning params - not used 
replay_buffer_size = 1000000 
learning_rate = 1e-4
gamma = 0.95 
discount_factor = 0.9
