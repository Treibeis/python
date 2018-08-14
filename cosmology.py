import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Constants
GRA = 6.672e-8
#GRV = 1.0
BOL = 1.3806e-16
#BOL = 1.0
PROTON = 1.6726e-24
#PROTON = 1.6726e-24/(2.176e-5)
ELECTRON = 9.10938356e-28
HBAR = 1.05457266e-27
CHARGE = 4.80320451e-10
H_FRAC = 0.76
SPEEDOFLIGHT = 2.99792458e+10

SIGMATH = 8*np.pi*(CHARGE**2/ELECTRON/SPEEDOFLIGHT**2)**2/3

Hfrac = 0.76
XeH = 0.0
XeHe = 0.0

# Units
UL = 3.085678e21
#UL = 1.0
UM = 1.989e43
#UM = 1.0
UV = 1.0e5
#UV = 1.0

UT = UL/UV
UD = UM/UL**3
UP = UM/UL/UT**2
UE = UM*UL**2/UT**2

G = GRA*UM*UT**2/UL**3

YR = 3600*24*365
# Cosmology
def H(a, Om = 0.315, h = 0.6774):
	H0 = h*100*UV/UL/1e3
	H = H0*(Om/a**3+(1-Om))**0.5
	return H

def DZ(z, Om = 0.315, h = 0.6774):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2)/H(a, Om, h)
	I = quad(integrand, 1/(1+z), 1, epsrel = 1e-8)
	return I[0]

def dt_da(a, Om = 0.315, h = 0.6774):
	return 1/a/H(a, Om, h)

def TZ(z, Om = 0.315, h = 0.6774):
	#def integrand(a):
	#	return 1/a/H(a, Om, h)
	I = quad(dt_da, 0, 1/(1+z), args = (Om, h), epsrel = 1e-8)
	return I[0]

#lz0 = np.linspace(0,3300,3301)
lz0 = np.hstack([[0],10**np.linspace(-2, 4, 1000)])
lt0 = [np.log10(TZ(x)/1e9/YR) for x in lz0]

ZT = interp1d(lt0, lz0)