import numpy as np
import sys

n_samples = int(sys.argv[1])

# read all files in a folder
def read_file(which_param):
    folder = 'mv5/plainLHS/49d'
    file_name = folder + '/' + which_param + '.dat'
    return np.loadtxt(file_name)

train = [read_file(i) for i in range(n_samples)]  
np.savetxt('mv5/plainLHS/49d/train.dat', train, newline = '\n', fmt='%s')