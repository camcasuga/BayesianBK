
import numpy as np
import pandas as pd

# make data containing only kinematical points (xbj, qs², and sqrt(s) energy with sigma_r as data and data error)

def p2f(array): 
    return np.array(array.str.rstrip('%').astype('float'))/100

# reading data file and storing separate data and errors into file
NCdata = pd.read_csv("hera_wcomp.csv", header = 0)
filt1 = (NCdata['Q**2 [GEV**2]'] <= 50.0) & (NCdata['Q**2 [GEV**2]'] >= 2.0) # get Q2 relevant range # (NCdata['XB'] < 0.01) &
exp = NCdata[filt1] 
data = np.array(exp['$\\sigma_{r,\\rm NC}^{+}$'])
statp = p2f(exp["stat +"])
statuncorp = p2f(exp["sys,uncor +"])
statcorp = p2f(exp["sys,cor +"])
data_err = np.sqrt(statp**2 + statuncorp**2 + statcorp**2) * np.array(exp['$\\sigma_{r,\\rm NC}^{+}$'])
xbj_list = np.array(exp['XB'])
Q2_list = np.array(exp['Q**2 [GEV**2]'])
sqrt_s_list = np.array(exp['sqrt_s'])
np.savetxt('exp_all.dat', np.c_[xbj_list, Q2_list, sqrt_s_list, data, data_err], delimiter = " ", newline = "\n")

# xbj_list, Q2_list, sqrt_s_list= np.loadtxt('exp_all.dat', usecols = (0,1,2), unpack = True)
# print('xbj = ', xbj_list)
# print('Q2 = ', Q2_list)
# print('sqrt_s = ', sqrt_s_list)