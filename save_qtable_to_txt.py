import numpy as np 

# init q-table
Q_table = np.zeros([625, 4])	# new: [625,4] , old: 2**16

# load q-table from file
loaded_qtable = np.load("q_table.npy")
assert loaded_qtable.shape == Q_table.shape, "Q-table sizes do not agree"
Q_table = loaded_qtable

# save q-table to txt-file
np.savetxt("qtable.txt", Q_table, fmt="%4f", delimiter=", ", newline="\n")