import numpy as np
#from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from colossus.cosmology import cosmology

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
KPC = PC*1e3
MPC = PC*1e6
SIGMATH = 8*np.pi*(CHARGE**2/ELECTRON/SPEEDOFLIGHT**2)**2/3
STEFAN = 2*np.pi**5*BOL**4/(15*PLANCK**3*SPEEDOFLIGHT**2)
RC = 4*STEFAN/SPEEDOFLIGHT

AU = 1.495978707e13
Lsun = 3.828e33
Rsun = 6.96342e10

H00 = 1e7/MPC

cosname = 'planck15'
cosmo = cosmology.setCosmology(cosname)
print(cosmo)
h = cosmo.H0/100
#print('log10(h) = {:.3f}'.format(np.log10(h)))
H0 = H00
rho0 = cosmo.Om0*H0**2*3/8/np.pi/GRA
# Units
UL = 3.085678e24/h
Msun = 1.989e33
UM = 1.989e33/h
UV = 1.0e5
# The internal units have h built-in because 
# this is what colossus uses for the power spectrum. 

Om = cosmo.Om0
Ob = cosmo.Ob0
lo = np.array([Om, 1-Om, 1./24000./h**2])
lw = np.array([0., -1., 1./3.])

def Hubble_a(a, lo=lo, lw=lw):
	assert len(lw)==len(lo)
	H0 = H00
	out = 0
	for o, w in zip(lo, lw):
		out += o * a**(-3*(1+w))
	return H0*out**(0.5)

def dt_da(a, lo, lw, h=h):
	return 1/a/Hubble_a(a, lo, lw)/h

def age_a(a, lo=lo, lw=lw, h=h):
	def dt_da(a):
		return 1./(a*h*Hubble_a(a, lo, lw))
	I = quad(dt_da, 0., a, epsrel = 1e-4)
	return I[0]

def delta_plus(z, Om = Om, Ol = 1-Om):
	lo = np.array([Om, Ol, 1./24000./h**2])
	lw = np.array([0., -1., 1./3.])
	a = 1/(1+z)
	def integrand(a):
		return 1./(a*Hubble_a(a, lo, lw))**3
	I = quad(integrand, 0., a, epsrel = 1e-4)
	return I[0]*Hubble_a(a, lo, lw)

lz0 = np.hstack([[0],10**np.linspace(-2, 4, 1000)])
lt0 = [np.log10(age_a(1/(x+1))/1e9/YR) for x in lz0]
ZT = interp1d(lt0, lz0)

def rhom(a, Om = Om, h = h):
	H0 = h*H00
	rho0 = Om*H0**2*3/8/np.pi/GRA
	return rho0/a**3
	
def RV(m = 1e10, z = 10.0, delta = 200):
	M = m*UM
	return (M/(rhom(1/(1+z))*delta)*3/4/np.pi)**(1/3)

def Tvir(m = 1e10, z = 10.0, delta = 200):
	M = m*UM
	Rvir = (M/(rhom(1/(1+z))*delta)*3/4/np.pi)**(1/3)
	return GRA*M*mmw()*PROTON/Rvir/(3*BOL)

def Mk(k):
	R = 1/k*UL
	M = R**3*4*np.pi/3*rho0/UM
	return M

def trans_FDM(k, ma = 1e-22, h = h):
	kJeq = 9*(ma/1e-22)**0.5/h
	xJ = 1.61*(ma/1e-22)**(1/18)*(k/kJeq)
	Tfdm = np.cos(xJ**3)/(1+xJ**8)
	return Tfdm

def FDM(ps, ma, h):
	lk, lPk = ps[0], ps[1]
	TFDM = trans_FDM(lk, ma, h)
	lPk_ = lPk * TFDM**2
	return [lk, lPk_]

#"""
k1, k2 = 1e-4, 1e2
lk = np.geomspace(k1, k2, 10000)
lPk0 = cosmo.matterPowerSpectrum(lk)
norm = 1.0
lPk = lPk0/norm
ps0 = [lk, lPk]

import camb
from camb import model, initialpower
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.74, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
zps = 0
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[zps], kmax=k2)
pars.NonLinear = model.NonLinear_both
results = camb.get_results(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=k2, npoints = 10000)
pars.NonLinear = model.NonLinear_none
results.calc_power_spectra(pars)
kh, zlin, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=k2, npoints = 10000)

ma1 = 1e-21
ps1 = FDM(ps0, ma1, h)
ma2 = 1e-22
ps2 = FDM(ps0, ma2, h)

klm = [1e-4, 1e2] #[1e-4, 1e2]
Plm = [1e-3, 3e4] #[1e-3, 5e4]
plt.figure()
plt.loglog(*ps0, label='CDM (z=0), '+cosname)
plt.loglog(*ps1, '--', label=r'FDM: $m_{\mathrm{a}}='
			+'{:.1f}'.format(ma1*1e21)+r'\times 10^{-21}\ \mathrm{eV}/c^{2}$')
plt.loglog(*ps2, '-.', label=r'FDM: $m_{\mathrm{a}}='
			+'{:.1f}'.format(ma2*1e22)+r'\times 10^{-22}\ \mathrm{eV}/c^{2}$')
plt.loglog(kh, pk[0], color='gray', lw=4.5, alpha=0.5, label='CDM (linear), CAMB')
plt.loglog(kh_nonlin, pk_nonlin[0], ':', label='CDM (non-linear), CAMB')
plt.xlabel(r'$k\ [h\ \mathrm{Mpc^{-1}}]$', size=14)
plt.ylabel(r'$P(k)\ [h^{-3}\ \mathrm{Mpc^{3}}]$', size=14)
plt.xlim(klm)
plt.ylim(Plm)
#plt.title('z={}'.format(zps))
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('powspec.pdf')
plt.close()

def Dgrow_(z, Om = Om, Ol = 1-Om):
	return delta_plus(z, Om, Ol)/delta_plus(0, Om, Ol)

def Dgrow(z, Om = Om, Ol = 1-Om):
	Omz = Om*(1+z)**3/(Om*(1+z)**3 + Ol)
	Olz = 1-Omz
	gz = 2.5*Omz*(Omz**(4/7)-Olz+(1+Omz/2)*(1+Olz/70))**-1
	gz0 = 2.5*Om*(Om**(4/7)-Ol+(1+Om/2)*(1+Ol/70))**-1
	return gz/(1+z)/gz0

from colossus.lss import peaks
deltac0 = peaks.collapseOverdensity(corrections = True, z = 0)
def deltac_z(z, Om = Om, Ol = 1-Om, mode=1):
	if mode==0:
		deltac = deltac0
	else:
		deltac = peaks.collapseOverdensity(corrections = True, z = z)
	Dz = Dgrow(z, Om, Ol)
	return deltac/Dz

#lz = np.linspace(0, 20, 1000)
#plt.plot(1/(1+lz), [deltac_z(z) for z in lz])
#plt.show()

def fPS(nu0, mode = 0, A = 0.322, q = 0.3, fac = 0.84, norm=1):
	nu = nu0*fac
	if mode>0:
		return (2/np.pi)**0.5*nu*np.exp(-nu**2/2) * A * (1+1/nu**(2*q)) /norm
	else:
		return (2/np.pi)**0.5*nu0*np.exp(-nu0**2/2) /norm
	
gf1 = 4*np.pi/3
def wf1(k, R):
	return 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3
	
gf2 = 6*np.pi**2
def wf2(k, R):
	return 1*(k*R<=1) + 0*(k*R>1)
	
def sigma2_M(M, ps, wf = wf1, gf = gf1, Om = Om, norm=1.0):
	rho0 = Om*H0**2*3/8/np.pi/GRA
	R = (M*UM/rho0/gf)**(1/3)/UL
	x, y = ps[0], ps[1]
	#sel = lk<kc
	#x, y = lk[sel], lPk[sel]
	return np.trapz(y*x**2*wf(x, R)**2, x)/2/np.pi**2/norm

corr = lambda z: 1

from scipy.misc import derivative
def halomassfunc(lm, z, ps, wf=wf1, gf=gf1, Om = Om, Ol = 1-Om, dx = 5e-2, corr = corr, mode=0):
	dc = deltac_z(z, Om, Ol)
	lognu = np.log([dc/sigma2_M(m, ps, wf, gf)**0.5 for m in lm])
	logm = np.log(lm)
	func = interp1d(logm, lognu)
	rho0 = Om*H0**2*3/8/np.pi/GRA
	def hmf(M):
		nu = dc/sigma2_M(M, ps, wf, gf)**0.5
		logM = np.log(M)
		return rho0/M**2*fPS(nu, mode)*np.abs(derivative(func, logM, dx))*UL**3/UM / corr(z)
	return hmf
	
lm0 = np.geomspace(1e3, 2e16, 10000)

exit()
	
mode = 1
m1, m2 = 1e4*h, 1e16*h
lm = np.geomspace(m1, m2, 100)
lz = [0, 6, 10, 20, 50]
lhmf = []
for z in lz:
	hmf = halomassfunc(lm0, z, ps0, wf1, gf1, mode=mode)
	lhmf.append(np.array([hmf(m) for m in lm]))
	
plt.figure()
for z, f in zip(lz, lhmf):
	plt.loglog(lm/h, lm*f*h**3, label=r'$z={:.0f}$'.format(z))
plt.legend()
plt.xlim(m1/h, m2/h)
plt.ylim(1e-9, 1e6)
plt.xlabel(r'$M\ [\mathrm{M_{\odot}}]$', size=14)
plt.ylabel(r'$Mdn/dM\ [\mathrm{Mpc^{-3}}]$', size=14)
plt.tight_layout()
plt.savefig('mhmf.pdf')
plt.close()

plt.figure()
for z, f in zip(lz, lhmf):
	plt.loglog(lm/h, lm**2*f/(rho0/UM*UL**3)*h**2, label=r'$z={:.0f}$'.format(z))
plt.legend()
plt.xlim(m1/h, m2/h)
plt.ylim(1e-16, 0.1)
plt.xlabel(r'$M\ [\mathrm{M_{\odot}}]$', size=14)
plt.ylabel(r'$M^{2}dn/dM/\rho_{\rm m}$', size=14)
plt.tight_layout()
plt.savefig('m2hmf.pdf')
plt.close()
