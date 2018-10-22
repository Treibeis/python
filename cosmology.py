import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Constants
GRA = 6.672e-8
#GRV = 1.0
BOL = 1.3806e-16
#BOL = 1.0
PROTON = 1.6726e-24
ELECTRON = 9.10938356e-28
HBAR = 1.05457266e-27
PLANCK = HBAR*2*np.pi
CHARGE = 4.80320451e-10
SPEEDOFLIGHT = 2.99792458e+10
eV = 1.60218e-12
YR = 3600*24*365
PC = 3.085678e18
SIGMATH = 8*np.pi*(CHARGE**2/ELECTRON/SPEEDOFLIGHT**2)**2/3

# Primordial abundances
Hfrac = 0.76
XeH = 0.0
XeHe = 0.0
def mmw(xeH = XeH, xeHe = XeHe, X = Hfrac):
	xh = 4*X/(1+3*X)
	return 4.0/(1.0+3*X)/(xh*(1+xeH)+(1-xh)*(1+xeHe))

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

# Cosmology
def H(a, Om = 0.315, h = 0.6774, OR = 9.54e-5):
	H0 = h*100*UV/UL/1e3
	H = H0*(Om/a**3+(1-Om)+OR/a**4)**0.5
	return H

def DZ(z, Om = 0.315, h = 0.6774, OR = 9.54e-5):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2)/H(a, Om, h, OR)
	I = quad(integrand, 1/(1+z), 1, epsrel = 1e-8)
	return I[0]

def dt_da(a, Om = 0.315, h = 0.6774, OR = 9.54e-5):
	return 1/a/H(a, Om, h, OR)

def TZ(z, Om = 0.315, h = 0.6774, OR = 9.54e-5):
	I = quad(dt_da, 0, 1/(1+z), args = (Om, h, OR), epsrel = 1e-8)
	return I[0]

#lz0 = np.linspace(0,3300,3301)
lz0 = np.hstack([[0],10**np.linspace(-2, 4, 1000)])
lt0 = [np.log10(TZ(x)/1e9/YR) for x in lz0]
ld0 = [DZ(x)/UL/1e3 for x in lz0]

ZT = interp1d(lt0, lz0)
ZD = interp1d(ld0, lz0)

def rhom(a, Om = 0.315, h = 0.6774):
	H0 = h*100*UV/UL/1e3
	rho0 = Om*H0**2*3/8/np.pi/GRA
	return rho0/a**3

def T_cosmic(z, alpha = -4, beta = 1.27, z0 = 189.6, zi = 1020, T0 = 2.726*1021):
	def integrand(logt):
		return alpha/3.0-(2+alpha)/3.0*(1-np.exp(-(ZT(logt)/z0)**beta))
	I = quad(integrand, np.log10(TZ(zi)/1e9/YR), np.log10(TZ(z)/1e9/YR), epsrel = 1e-6)[0]
	temp = T0*10**I
	return temp

def T_b(z, a1=1./119, a2=1./115, T0=2.726):
	a = 1./(1+z)
	return T0/(a*(1+a/(a1*(1+(a2/a)**1.5))))

def T_dm(z, m = 1., T0=2.726):
	zc = m*eV / ((3./2)*BOL*T0) - 1
	Tc = T0*(1+zc)
	return T0*(1+z) * (z>zc) + Tc*((1+z)/(1+zc))**2 * (z<=zc)

# DM haloes
def tff(z = 10.0, delta = 200):
	return (3*np.pi/(32*GRA*delta*rhom(1/(1+z))))**0.5

def Lvir(m = 1e10, z = 10.0, delta = 200):
	M = m*UM/1e10
	Rvir = (M/(rhom(1/(1+z))*delta)*3/4/np.pi)**(1/3)
	return 3*GRA*M**2/Rvir/tff(z, delta)/5

def Tvir(m = 1e10, z = 10.0, delta = 200):
	M = m*UM/1e10
	Rvir = (M/(rhom(1/(1+z))*delta)*3/4/np.pi)**(1/3)
	return 3*GRA*M*mmw()*PROTON/Rvir/5/(3*BOL)
