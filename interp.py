import numpy as np
from scipy.interpolate import *
import txt 

Unit1 = 1.0
Unit2 = 1.0

D_Z = np.array(txt.retxt('/home/friede/python/d_z_WMAP5.txt', 2, 0, 0))
D_Z = np.array([D_Z[0]*Unit1, D_Z[1]*Unit2])
ZD = interp1d(D_Z[0],D_Z[1], kind='cubic')
DZ = interp1d(D_Z[1],D_Z[0], kind='cubic')
H_Z = np.array(txt.retxt('/home/friede/python/H_z_WMAP5.txt', 2, 0, 0))
H_Z = np.array([H_Z[0]/Unit1, H_Z[1]*Unit2])
HZ = interp1d(H_Z[1],H_Z[0], kind='cubic')
ZH = interp1d(H_Z[0],H_Z[1], kind='cubic')




