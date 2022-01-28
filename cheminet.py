import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double

array_1d_double = npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

libcd = npct.load_library("libcheminet", ".")

libcd.chemsl.restype = None
libcd.chemsl.argtypes = [array_1d_double, array_1d_double, array_1d_double, c_double, c_double, c_double, c_double, c_int, c_double, c_double]

def chemistry1(T, nin, dt0, epsH, J_21, Ns, xnh, xnhe, xnd, xnli, nmax = 100, z  =5, T0 = 2.726):
	out = np.zeros(Ns+2,dtype='float')
	xl = np.array([xnh, xnhe, xnd, xnli])
	libcd.chemsl(nin, xl, out, T, dt0, epsH, J_21, nmax, z, T0)
	dt_cum = out[17]
	total = int(out[18]+0.1)
	ny = out[0:17]
	return [ny, dt_cum, total]

"""
def test():
	n0_default = [1., 5e-4, 1e-18, 1e-11, 5e-16] + \
			 [5e-4, 1.0, 1e-19, 0.0] + \
			 [1.0, 5e-4, 2.5e-11] + \
			 [1.0, 0, 0, 0, 0]
	n0_default = np.array(n0_default, dtype=np.float64)
	n0_default[0:9] = n0_default[0:9]*0.93
	n0_default[9:12] = n0_default[9:12]*4e-5
	n0_default[12:] = n0_default[12:]*4.6e-10
	a = chemistry1(1e3, n0_default, 1e6*3.14e7, 1e-2, 0.0, 17, 0.93, 0.07, 4e-5, 4.6e-10, 100, 5.0, 2.726)
	print(a[0])
	print(a[1], a[2])
	return a

a = test()
"""

"""
libcd.cos_doubles.restype = None
libcd.cos_doubles.argtypes = [array_1d_double, array_1d_double, c_int]

def cos_doubles_func(in_array):
	out_array = np.zeros(in_array.shape)
	libcd.cos_doubles(in_array, out_array, len(in_array))
	return out_array

import pylab

x = np.arange(0, 2*np.pi, 0.1)
#y = np.zeros(x.shape) #np.empty_like(x)
y = cos_doubles_func(x)

pylab.plot(x, y)
pylab.show()
"""


