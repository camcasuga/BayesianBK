import numpy as np
import sys

n_samples = int(sys.argv[1])
#type_lhs = str(sys.argv[2])

# read all files in a folder
def read_file(which_param):
    folder = 'mv5/orthLHS/{}d/trains'.format(n_samples)
    file_name = folder + '/' + str(which_param) + '.txt'
    return np.loadtxt(file_name)

train = [read_file(i) for i in range(n_samples)]  
np.savetxt('mv5/orthLHS/{}d/train.dat'.format(n_samples), train, newline = '\n', fmt='%s')