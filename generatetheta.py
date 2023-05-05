import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import qmc
from math import pow
import sys

#folder = str(sys.argv[1]) # path to folder 
model = str(sys.argv[1])
n_paramvectors = int(sys.argv[2])
type_lhs = int(sys.argv[3]) # 1 for plain LHS # 2 for orthogonal 


if type_lhs == 1:
	type_lhs_name = 'plainLHS'
if type_lhs == 2:
	type_lhs_name = 'orthLHS'


def get_lhcsamples(nparams, nsamples, param_limits = None ,seed_ = None, strength_ = type_lhs):
	sp_sampler = qmc.LatinHypercube(d = nparams, seed = seed_, strength = strength_)
	sp_sample = sp_sampler.random(n = nsamples)
	if param_limits is None: 
		return sp_sample
	sp_params = qmc.scale(sp_sample, param_limits[0], param_limits[1])
	return sp_params

model_dir = '{}/{}/{}d'.format(model, type_lhs_name, str(n_paramvectors))
isExist = os.path.exists(model_dir)
if not isExist: os.mkdir(model_dir)
isExist2 = os.path.exists(model_dir + '/bks')
if not isExist2: os.mkdir(model_dir + '/bks')

if model == 'mv':
	l_bounds = [.001, .1, 10.] # Qs02 , C2, sigma0/2
	u_bounds = [0.2, 40., 30.] # we are using MVe where the anomalous dimension gamma = 1
	n_params = 3
	mylimits = np.array([l_bounds, u_bounds])
	myparams = get_lhcsamples(n_params, 
			   n_paramvectors, 
			   param_limits = mylimits, 
			   seed_ = 10, 
			   strength_ = type_lhs)
	np.savetxt(model_dir + '/theta.dat', myparams, newline = '\n')
	print('design points saved in: ./' + model_dir + '/theta.dat')
	i = 0
	lines = ['#!/bin/bash', '\n']
	for qs02, c2 in myparams[:, 0:2]:
		#cmd = 'OMP_NUM_THREADS=2 ../rcbk/build/bin/rcbk -ic MV {} 1 0.01 1 -rc BALITSKY -alphas_scaling {} -maxy 12 -fast -output {}/bks/{}.dat'.format(str(qs02), str(c2), model_dir, str(i))
		#os.system(cmd)
		line = "sbatch -J bk submitmv.sh {0} {1} {2}".format(str(qs02), str(c2),str(i))
		lines.append(line)
		i += 1

if model == 'mve':
	l_bounds = [.001, .5, 0.1, 10.] # Qs02 , ec, C2, sigma0/2
	u_bounds = [0.2, 40., 40., 30.] # we are using MVe where the anomalous dimension gamma = 1
	n_params = 4
	mylimits = np.array([l_bounds, u_bounds])
	myparams = get_lhcsamples(n_params, 
			   n_paramvectors, 
			   param_limits = mylimits, 
			   seed_ = 10, strength_ = type_lhs)
	np.savetxt(model_dir + '/theta.dat', myparams, newline = '\n')
	print('design points saved in: ./' + model_dir + '/theta.dat')
	i = 0
	lines = ['#!/bin/bash', '\n']
	#myparams = np.vstack(np.loadtxt(model_dir + "/theta.dat", unpack = True)).T
	print(myparams)
	for qs02, ec, c2 in myparams[:, 0:3]:
		#cmd = 'OMP_NUM_THREADS=2 ../rcbk/build/bin/rcbk -ic MV {} 1 0.01 {} -rc BALITSKY -alphas_scaling {} -maxy 12 -fast -output {}/bks/{}.dat'.format(str(qs02), str(ec), str(c2), model_dir, str(i))
		#os.system(cmd)
		line = "sbatch -J bk submitmve.sh {0} {1} {2} {3}".format(str(qs02),str(ec), str(c2),str(i))
		lines.append(line)
		i += 1

if model == 'mv5':
	l_bounds = [.001, .5, .5, 0.1, 10.] # Qs02 , ec, C2, sigma0/2, gamma
	u_bounds = [1., 2.0, 40., 40., 30.] # we are using MVe where the anomalous dimension gamma = 1
	n_params = 5
	mylimits = np.array([l_bounds, u_bounds])
	myparams = get_lhcsamples(n_params, 
			   n_paramvectors, 
			   param_limits = mylimits, 
			   seed_ = 10, strength_ = type_lhs)
	np.savetxt(model_dir + '/theta.dat', myparams, newline = '\n')
	print('design points saved in: ./' + model_dir + '/theta.dat')
	i = 0
	lines = ['#!/bin/bash', '\n']
	#myparams = np.vstack(np.loadtxt(model_dir + "/theta.dat", unpack = True)).T
	print(myparams)
	for qs02, gamma, ec, c2 in myparams[:, 0:4]:
		#cmd = 'OMP_NUM_THREADS=2 ../rcbk/build/bin/rcbk -ic MV {} {} 0.01 {} -rc BALITSKY -alphas_scaling {} -maxy 12 -fast -output {}/bks/{}.dat'.format(str(qs02), str(ec), str(c2), model_dir, str(i))
		#os.system(cmd)
		line = "sbatch -J bk submitmv5.sh {0} {1} {2} {3} {4}".format(str(qs02), str(gamma) ,str(ec), str(c2),str(i))
		lines.append(line)
		i += 1


np.savetxt('submit_bk_jobs_{}_{}_{}d.sh'.format(model, type_lhs_name, n_paramvectors), lines, newline = '\n', fmt='%s')