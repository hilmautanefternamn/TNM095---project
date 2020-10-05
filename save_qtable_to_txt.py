import numpy as np 

# init q-table
Q_table = np.zeros([2**16, 4])	# [625,4]

# load q-table from file
loaded_qtable = np.load("filename.npy")
assert loaded_qtable.shape == Q_table.shape, "Q-table sizes do not agree"
Q_table = loaded_qtable

# save q-table to txt-file
np.savetxt("filename.txt", Q_table, fmt="%4f", delimiter=", ", newline="\n")