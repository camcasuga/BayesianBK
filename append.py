import numpy as np
import sys

model = str(sys.argv[1])
n_paramvectors = int(sys.argv[2])
type_lhs = int(sys.argv[3])

if type_lhs == 1:
	type_lhs_name = 'plainLHS'
if type_lhs == 2:
	type_lhs_name = 'orthLHS'

# read all files in a folder
def read_file(which_param):
    folder = 'postsamples/mv5/trains'.format(model,type_lhs_name, n_paramvectors)
    file_name = folder + '/' + str(which_param) + '.txt'
    return np.loadtxt(file_name)

train = [read_file(i) for i in range(n_paramvectors)]  
np.savetxt('postsamples/mv5/train.dat'.format(model, type_lhs_name, n_paramvectors), train, newline = '\n', fmt='%s')