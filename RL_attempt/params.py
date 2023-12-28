# inspired by MolDQN-pytorch 

start_molecule = None

# learning params 
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000

# not used? just assumed 
optimizer = "Adam"

# these used 
atom_types = ["C", "N", "O", "P", "S"] # ignoring H in this, so this is atom types allowed inside the state 
# above one also used more 
max_steps_per_episode = 40 # except this 
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
allowed_ring_sizes = [3, 4, 5, 6]

# more learning params 
replay_buffer_size = 1000000 # not yet 
learning_rate = 1e-4
gamma = 0.95 # not yet 
discount_factor = 0.9
