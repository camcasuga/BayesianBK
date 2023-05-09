import numpy as np
import sys

n_samples = int(sys.argv[1])

# read all files in a folder
def read_file(which_param):
    folder = 'mv5/orthLHS/49d/trains'
    file_name = folder + '/' + str(which_param) + '.txt'
    return np.loadtxt(file_name)

train = [read_file(i) for i in range(n_samples)]  
np.savetxt('mv5/orthLHS/49d/train.dat', train, newline = '\n', fmt='%s')