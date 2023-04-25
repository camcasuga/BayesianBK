import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import qmc
from math import pow
import sys

folder = str(sys.argv[1]) # path to folder 
model = str(sys.argv[1])

def get_lhcsamples(nparams, nsamples, param_limits = None ,seed_ = None, strength_ = 1):
	sp_sampler = qmc.LatinHypercube(d = nparams, seed = seed_, strength = strength_)
	sp_sample = sp_sampler.random(n = nsamples)
	if param_limits is None: 
		return sp_sample
	sp_params = qmc.scale(sp_sample, param_limits[0], param_limits[1])
	return sp_params

n_paramvectors = sys.argv[3]

if model == 'mv':
	l_bounds = [0.001, 0.1, 1.0] # Qs0² , C², sigma0/2
	u_bounds = [2.0, 100.0, 40.0] # we are using MVe where the anomalous dimension gamma = 1
	n_params = 3
	mylimits = np.array([l_bounds, u_bounds])
	myparams = get_lhcsamples(n_params, n_paramvectors, param_limits = mylimits, seed_ = 10, strength_ = 1)
	np.savetxt(folder + '/{}model/{}d/theta.dat'.format(model, str(n_paramvectors)), myparams, newline = '\n')
	i = 0
	lines = []
	for qs02, c2 in myparams[:, 0:2]:
		#cmd = 'OMP_NUM_THREADS=2 ./rcbk/build/bin/rcbk -ic MV {} 1 0.01 1 -rc BALITSKY -alphas_scaling {} -maxy 12 -fast -output w3p20d/{}.dat'.format(str(qs02), str(c2), str(i))
		#os.system(cmd)
		print("sbatch -J bk submit.sh {0} {1} {2}".format(str(qs02),str(c2),str(i)))
		i += 1

if model == 'mve':
	l_bounds = [0.001, 0.5, 0.1, 1.0] # Qs0² , ec, C², sigma0/2
	u_bounds = [2.0, 100.0, 100,0, 40.0] # we are using MVe where the anomalous dimension gamma = 1
	n_params = 4
	mylimits = np.array([l_bounds, u_bounds])
	myparams = get_lhcsamples(n_params, n_paramvectors, param_limits = mylimits, seed_ = 10, strength_ = 1)
	np.savetxt(folder + '/{}model/{}d/theta.dat'.format(model, str(n_paramvectors)), myparams, newline = '\n')
	#print('design points saved in: ./' + model_dir + '/theta.dat')
	i = 0
	lines = []
	for qs02, ec, c2 in myparams[:, 0:3]:
		#cmd = 'OMP_NUM_THREADS=2 ./rcbk/build/bin/rcbk -ic MV {} 1 0.01 {} -rc BALITSKY -alphas_scaling {} -maxy 12 -fast -output w3p20d/{}.dat'.format(str(qs02), str(ec), str(c2), str(i))
		#os.system(cmd)
		line = "sbatch -J bk submit.sh {0} {1} {2} {3}".format(str(qs02),str(ec), str(c2),str(i))
		print(line)
		i += 1

np.savetxt('submit_bk_jobs_{}_{}_{}d.sh'.format(model, type_lhs_name, n_paramvectors), lines, newline = '\n', fmt='%s')