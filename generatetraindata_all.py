import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy import interpolate, integrate
from scipy.special import k0,k1
from numba import njit, float64
import time


#folder = str(sys.argv[1]) # pathname to folder containing theta
#theta_file = folder + "/theta.dat"
theta_file = 'test.dat'
myparams = np.vstack(np.loadtxt(theta_file, unpack = True)).T
n_params_list = list(range(len(myparams)))
#folder = "../../Downloads/practice/mve100d"

def get_file(index):
    #filename = '{0}/bks/{1}.dat'.format(folder,index)
    return str(filename)


def ReadBKDipole(index): # 
    '''Read the dipole amplitude from the given datafile produced running Heikki's BK code
    
    Returns an interpolator for the dipole: N(Y, r), where r is in GeV^-1, and x = x_0*exp(Y)
    
    Note: as this interpolates in r and not in log r, at very small r there are some small interpolation errors
    '''
    thefile =  'test_bk.dat' #get_file(index)
    
    with open(thefile) as f:
        content = f.read().split("###")
    
    content = content[1:]   # gets rid of the stuff at the beginning
    content = [i.split() for i in content] # cleans up lines
    NrY_data = []
    pars = []
    for i in content:
        '''Separates and sorts the lines in the file
        
        Takes values in the beginning of the file to 'pars' list
        and every Y value with associated N(Y,r) values to 'NrY_data' list
        '''

        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

        
    rmYs = np.array(NrY_data).T[1:]     # removes Y values
    N_values = rmYs.T
    Y_values = np.array(NrY_data).T[0]

    pars = np.ndarray.flatten(np.array(pars))
    minr = pars[0]
    mult = pars[1]
    n = int(pars[2])
    r_values = np.array([minr*mult**i for i in range(n)])
    
    rgrid=[]
    ygrid=[]
    for y in Y_values:
        for r in r_values:
            rgrid.append(r)
            ygrid.append(y)

    interpolator = interpolate.CloughTocher2DInterpolator((ygrid, rgrid), N_values.flatten(), fill_value=0)
    
    return interpolator

interpolators = [ReadBKDipole(index) for index in n_params_list]

# physics

# shortcut
def sumZ2():
    Z   = [2/3,-1/3,-1/3] # u d s
    mat = [(Zi**2) for Zi in Z]
    return np.sum(mat)

Z2 = sumZ2()

# Define cross section integrand
    
@njit(float64(float64, float64, float64, float64, float64), cache=True)
def sigmaT(bess0, bess1, ai, z, m):
    return Z2*((((bess1)*(ai))**2)*(z**2 + (1-z)**2) + (m**2)*(bess0)**2)

@njit(float64(float64, float64, float64), cache=True)
def sigmaL(bess0, Q2, z):
    return Z2*4*Q2*(z**2*(1-z)**2)*(bess0**2)

@njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
def F2(bess0, bess1, ai, z, m, Q2):
    return (sigmaT(bess0, bess1, ai, z, m) + sigmaL(bess0, Q2, z))

#def FL(Q2, bess0, ai, z, m, Z):
#    return (sigmaL(bess0, Q2, z, Z))

def integrand(z, r, xbj, Q2, sqrt_s): # contains only kinematical points
    pi, m = np.pi, 0.14 # contants
    ai = np.sqrt((Q2)*z*(1-z) + m**2)
    ai_inp = ai*r
    bess0 = k0(ai_inp)
    bess1 = k1(ai_inp)
    y = Q2/(xbj*sqrt_s**2)
    mult = (y**2)/(1+(1-y)**2)
    return (1/(4*pow(pi,3))) * (F2(bess0, bess1, ai, z, m, Q2) - mult * sigmaL(bess0, Q2, z))

def intz(r, xbj, Q2, sqrt_s): # this is where the fitting parameters go into the intergrand
    return integrate.quad(integrand, 0.0, 1.0, args = (r, xbj, Q2, sqrt_s))[0] # integrates over z, returns only a function of r

def intzr(r, xbj, Q2, sqrt_s, index):
    x0 = 0.01
    y_int = np.log(x0/xbj)
    dip = interpolators[index]
    return r * dip(y_int,r) * intz(r, xbj, Q2, sqrt_s)

def get_sigmar(xbj, Q2, sqrt_s, index, sigma0_half):
    sigma0 = 2.56819 * 2 * sigma0_half # converts to 1/GeV²
    Nc = 3.0
    func = Q2 * Nc * sigma0 * integrate.quad(intzr, 0.0, 50.0, args = (xbj, Q2, sqrt_s, index))[0] 
    print(xbj, Q2, sqrt_s, myparams[:,][index], func)
    return func # integrates over r

def p2f(array):
    return np.array(array.str.rstrip('%').astype('float'))/100

# make the training set
def generate_training_set(xbj_list, Q2_list, sqrt_s_list, amplitude_params_list_index, sigma0_half_list): 
    trainingset = [] # different parameters
    supplement = [] # contain list of 
    for i in range(len(amplitude_params_list_index)):
        #print("currently at {}th parameter out of {} design points".format(i+1, len(amplitude_params_list_index)))
        #st = time.time()
        
        diff_kinematics = [get_sigmar(xbj_list[j], 
                                      Q2_list[j], 
                                      sqrt_s_list[j], 
                                      amplitude_params_list_index[i], 
                                      sigma0_half_list[i]) for j in range(len(xbj_list))]
        
        #et = time.time()
        #print(diff_kinematics)
        trainingset.append(diff_kinematics)
        #print("time taken at design point number {}: {} seconds".format(i+1, et-st))    
    return trainingset

xbj_list, Q2_list, sqrt_s_list = np.loadtxt('exp_all.dat', usecols = (0,1,2), unpack = True)
amplitude_params_list_index, sigma0_half_list = n_params_list, myparams[:,-1]

# generate and then save the training file
my_array = generate_training_set(xbj_list, Q2_list, sqrt_s_list, amplitude_params_list_index, sigma0_half_list)
np.savetxt('aa.dat', my_array, delimiter = " ", newline = "\n")