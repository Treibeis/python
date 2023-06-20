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
plt.style.use('test2')
plt.style.use('tableau-colorblind10')
import sys
import os

# Thu, Fri, morning, 9-12

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
YR = 3600*24*365.25
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

cosname = 'planck18-only'
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
Or = cosmo.Or0
aeq = Or/Om
lo = np.array([Om, 1-Om, 1./24000./h**2])
lw = np.array([0., -1., 1./3.])

#Mdown = lambda z: 3e6*((1+z)/10)**-1.5
Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5

def totxt(s, l, ls = 0, t = 0, k = 0):
	j = 0
	with open(s, 'w') as f:
		if t!=0:
			for r in range(len(ls)):
				f.write(ls[r])
				f.write(' ')
			f.write('\n')
		for i in range(len(l[0])):
			if j<k:
				print (l[0][i])
			else:
				for s in range(len(l)):
					f.write(str(l[s][i]))
					f.write(' ')
				f.write('\n')
			j = j+1

def retxt(s, n, k = 0, t = 0): # s: file name, n: num of columns, 
							   # k: num of head lines to skip, t: ifornot reverse
	out = []
	for i in range(n):
		out.append([])
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				for i in range(n):
					out[i].append(float(lst[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
	out[i] = np.array(out[i])
	return out

def mbind(z, mpbh, fpbh, aeq=aeq, seed=0):
	if seed>0:
		y0 = mpbh/aeq/(1+z)
	else:
		y0 = fpbh*mpbh/aeq**2/(1+z)**2
	y = y0*(y0<mpbh/fpbh) + mpbh/fpbh * (y0>=mpbh/fpbh)
	return y

# F200W: 8, F444W: 12
def findmag(fn, z, ind=12):
	data = np.array(retxt(fn, 13, 1))
	lz = data[0]
	sel = lz==z
	return np.min(data[ind][sel])

def Hubble_a(a, lo=lo, lw=lw):
	assert len(lw)==len(lo)
	#H0 = H00
	out = 0
	for o, w in zip(lo, lw):
		out += o * a**(-3*(1+w))
	return H00*out**(0.5)

def dt_da(a, lo, lw, h=h):
	return 1/a/Hubble_a(a, lo, lw)/h

def age_a(a, lo=lo, lw=lw, h=h):
	def dt_da(a):
		return 1./(a*h*Hubble_a(a, lo, lw))
	I = quad(dt_da, 0., a, epsrel = 1e-4)
	return I[0]

def distance(z, lo=lo, lw=lw, h=h):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2*h*Hubble_a(a, lo, lw))
	I = quad(integrand, 1/(1+z), 1, epsrel = 1e-8)
	return I[0]

def horizon(a, lo=lo, lw=lw, a0=1e-10):
	def integrand(loga):
		return SPEEDOFLIGHT/np.exp(loga)/Hubble_a(np.exp(loga),lo,lw)
	I = quad(integrand, np.log(a0), np.log(a), epsrel=1e-8)
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


def Dgrow_(z, Om = Om, Ol = 1-Om):
	return delta_plus(z, Om, Ol)/delta_plus(0, Om, Ol)

def Dgrow(z, Om = Om, Ol = 1-Om):
	Omz = Om*(1+z)**3/(Om*(1+z)**3 + Ol)
	Olz = 1-Omz
	gz = 2.5*Omz*(Omz**(4/7)-Olz+(1+Omz/2)*(1+Olz/70))**-1
	gz0 = 2.5*Om*(Om**(4/7)-Ol+(1+Om/2)*(1+Ol/70))**-1
	return gz/(1+z)/gz0 * (z>=0) + 1*(z<0)

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

gf3 = (2*np.pi)**1.5
def wf3(k, R):
	return np.exp(-(k*R)**2/2)
	
def sigma2_M(M, ps, wf = wf1, gf = gf1, Om = Om, norm=1.0, sigmax=18*np.pi**2):
	rho0 = Om*H00**2*3/8/np.pi/GRA
	R = (M*UM/rho0/gf)**(1/3)/UL
	x, y = ps[0], ps[1]
	#sel = lk<kc
	#x, y = lk[sel], lPk[sel]
	return min(np.trapz(y*x**2*wf(x, R)**2, x)/2/np.pi**2/norm, sigmax**2)

corr = lambda z: 1

from scipy.misc import derivative
def halomassfunc(lm, z, ps, wf=wf1, gf=gf1, Om = Om, Ol = 1-Om, dx = 5e-2, corr = corr, mode=0):
	dc = deltac_z(z, Om, Ol)
	lognu = np.log([dc/sigma2_M(m, ps, wf, gf, Om)**0.5 for m in lm])
	logm = np.log(lm)
	func = interp1d(logm, lognu)
	rho0 = Om*H00**2*3/8/np.pi/GRA
	def hmf(M):
		nu = dc/sigma2_M(M, ps, wf, gf)**0.5
		logM = np.log(M)
		return rho0/M**2*fPS(nu, mode)*np.abs(derivative(func, logM, dx))*UL**3/UM / corr(z)
	return hmf
	
#lm0 = np.geomspace(1e3, 2e16, 10000)

def ngem(m, z, ps, lm0, mmax=1e15, nm=100, wf=wf3, gf=gf3, Om = Om, Ol = 1-Om, dx = 5e-2, corr = corr, mode=0, mass=0):
	lm = np.geomspace(m, mmax, nm+1)
	hmf = halomassfunc(lm0, z, ps, wf, gf, Om, Ol, dx, corr, mode)
	if mass==0:
		ln = np.array([hmf(x) for x in lm])
	else:
		ln = np.array([hmf(x)*x for x in lm])
	return np.trapz(ln, lm)

from colossus.lss import mass_function
#mdef, model = '200m', 'tinker08'

def ngem0(m, z, mmax=1e15, nm=100, model='sheth99', mdef='fof', mass=0):
	lm = np.geomspace(m, mmax, nm+1)
	dndm = mass_function.massFunction(lm, z, mdef=mdef, model=model, q_out='dndlnM')/lm
	if mass==0:
		return np.trapz(dndm, lm)
	else:
		return np.trapz(lm*dndm, lm)

def msdens(lms, z, eps, ps, lm0, gf=gf3, wf=wf3, Om = Om, Ol = 1-Om, dx = 5e-2, corr = corr, mode=1, h=h, Ob=Ob):
	fb = Ob/Om
	#hmf = halomassfunc(lm0, z, ps, wf, gf, Om, Ol, dx, corr, mode)
	#lm = np.geomspace(lm0[0]*10, lm0[-1]/10, int(len(lm0)/10))
	#ln = np.array([hmf(x) for x in lm])
	#lsel = [lm > m/fb/eps*h for m in lms]
	#intg = ln*lm
	#out = np.array([np.trapz(intg[sel], lm[sel]) for sel in lsel])*fb*eps*h**2
	y = np.array([ngem(m/fb/eps*h, z, ps, lm0, wf=wf, gf=gf, Om=Om, Ol=Ol, dx=dx, corr=corr, mode=mode, mass=1) for m in lms])
	out = y*h**2*fb*eps
	return out 
	
def sfrtf(z, a, b, c, d):
	#t = 1/(z+1)
	#return a*(t**b*np.exp(-t/c)+d*np.exp(d*(t-1)/c)) 
	return a*(1+z)**b/(1+((1+z)/c)**d)

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

"""
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
"""

def PBH(ps, mpbh, fpbh, aeq, h, mode=1, mfac=1.0, alpha=0, Ob=Ob, Om=Om, iso=1, seed=0, dmax=178, out=0, cut=1):
	npbh = fpbh*rhom(1, Om, h)*(Om-Ob)/Om/mpbh/Msun
	if out>0:
		print('PBH number density: {:.2e} Mpc^-3'.format(npbh*(MPC)**3))
	ks = (2*np.pi**2*npbh)**(1.0/3.0)*MPC/h*mfac #4*1e3*(fpbh*(20/h/mpbh))**(1.0/3.0)
	kcut = 3*ks
	#print('PBH scale: {:.2e} h/Mpc'.format(ks))
	lk, lPk = ps[0], ps[1]
	kH = MPC/horizon(aeq)
	#print('Horizon size at aeq: {:.2e} kpc/h'.format(1/kH))
	selH = lk<kH
	aeq_ = np.ones(len(lk))*aeq
	aeq_[selH] = aeq * (kH/lk[selH])**2
	kts = max((2*np.pi**2*npbh/fpbh*aeq)**(1.0/3.0)*MPC/h, ks)
	ksub = (2*np.pi**2*npbh/fpbh)**(1.0/3.0)*MPC/h*mfac
	#gam = Om*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
	#q = lk*1e3/gam
	#T_k = (1+(15*q+(0.9*q)**1.5+(5.6*q)**2)**1.24)**(-1.24)
	grow0 = Dgrow(0)/Dgrow(1.0/aeq_-1)
	fc = (Om-Ob)/Om
	a_ = 0.25*((1+24*fc)**0.5-1)
	grow = ( (1.0+1.5*fc/a_/aeq_)**a_ - 1.0 ) #* T_k *(lk/kH)**2
	#print('Ratio of growth factor at z=0:', np.max(grow)/np.max(grow0), np.max(grow), np.max(grow0))
	#print(grow, grow0)
	if mode==0:
		sel = lk>ks
	else:
		sel = lk>0 #lPk < pkpbh
	delta2 = lk[sel]**3/ks**3 * (grow[sel])**2 * fpbh**2
	if seed>0:
		delta2 = delta2*(lk[sel]<kts) + delta2*(lk[sel]>=kts)*(lk[sel]/kts)**3
	if dmax>0:
		delta2[delta2>dmax**2] = dmax**2
	pkpbh =  2*np.pi**2 * delta2/lk[sel]**3
	lPk_ = np.copy(lPk)
	kcrit = ks #kcut #(ks*kcut)**0.5
	delt = (lk[sel]/kcrit)**3
	sel0 = lk[sel]>kcut
	delt[sel0] *= 0 #(lk[sel][sel0]/(kcut))**-3 #fpbh
	#delt[delt<1] = 1
	#delt[delt>mfac] = mfac
	#if seed>0:
	#	pkpbh = pkpbh*mfac*(lk[sel]<ks) + pkpbh*mfac*(lk[sel]>=ks)*(lk[sel]/ks)**3
	if iso>0:
		piso = pkpbh*mfac #+ lPk[sel] * delt * grow**2 * fpbh / grow0
	else:
		piso = pkpbh*mfac + lPk[sel] * delt * grow**2 * fpbh / grow0
	#piso[lk>ks*fpbh*1e4] = 0
	if cut>0:
		piso[lk[sel]>ksub] = 0
	lPk_[sel] += piso #* 1e3
	return [lk, lPk_]

#"""

ma1 = 1e-21
#ps1 = FDM(ps0, ma1, h)
ma2 = 1e-22
#ps2 = FDM(ps0, ma2, h)

def sfrd_func(hmf0, hmf1, lmass, lz, i, Vcom):
	dnh = np.array([hmf0(m) for m in lmass]) - np.array([hmf1(m) for m in lmass]) # dn(m, z_i)
	dnh[dnh<0] = 0
	dnhalo = np.trapz(dnh*flw(lmass/h,lz[i]), lmass)*Vcom # num of pop3 hosts formed in this z bin
	lms = flw(lmass/h,lz[i],1)*eta0*Ob/Om*lmass/h # stellar masses of pop3 hosts
	lms[lms>msmax] = msmax # Apply the maximum stellar mass
	dmstar = np.trapz(dnh*lms, lmass)*Vcom # total stellar mass of pop3 stars formed in this z bin
	nhalo = dnhalo
	mstar = dmstar
	print('z~{}-{}: delta nhalo: {}, delta mstar: {:.2e} Msun'.format(lz[i], lz[i+1], dnhalo, dmstar))
	sfrd = dmstar*h**3/Vcom*YR/(age_a(1/(1+lz[i]))-age_a(1/(1+lz[i+1])))
	return sfrd, dnhalo, dmstar

nf = 11
def rockstar(hd):
	return np.array([hd[2], hd[4], *hd[8:11]*1e3])

#lls = ['-', (0,(10,5)), '--', '-.', ':', (0,(3,3,1,1,1,3))]
lls = ['-', '--', '-.', ':', (0,(10,5)), (0,(3,3,1,1,1,3))]
#llc = ['k', 'b', 'g', 'r']
llc = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
lmk = ['o', '^', 'D', 's']

def fcol_z(lz, ps, fac, lm, maxm, lm0, mode=1, Om=Om, h=h):
	nz = len(lz)
	lf = np.zeros(nz)
	for i in range(nz):
		z = lz[i]
		hmf0 = halomassfunc(lm0, z, ps, wf1, gf1, mode=mode)
		mf0 = np.array([hmf0(m)*m*h**3 for m in lm])
		mcut = Mdown(z)*fac
		if fac==0:
			mcut = 64
		sel = lm/h>mcut
		sel *= lm/h<maxm[i] #10*mcut
		rho = rhom(1/(1+z), Om, h)*MPC**3/Msun
		lf[i] = np.trapz(mf0[sel], lm[sel])*h**2/rho
	return lf

def findfpbh(ref, z, eps, ps0, lm0, m1=1e8, m2=1e12, nm=10, fmin = -10, rerr=0.1, iso=1, gf=gf3, wf=wf3, seed=0, dmax=0):
	lf = np.ones(nm+1)
	lm = np.geomspace(m1, m2, nm+1)
	for i in range(nm+1):
		m = lm[i]
		f1, f2 = fmin, 0
		ps3 = PBH(ps0, m, 1, aeq, h, mfac=1, iso=iso, seed=seed, dmax=dmax)
		y = msdens([ref[0]], z, eps, ps3, lm0, gf=gf, wf=wf)[0]
		if y<ref[1]:
			continue
		ps3 = PBH(ps0, m, 10**fmin, aeq, h, mfac=1, iso=iso, seed=seed, dmax=dmax)
		y = msdens([ref[0]], z, eps, ps3, lm0, gf=gf, wf=wf)[0]
		if y>ref[1]:
			lf[i] = 10**fmin
			continue
		f = 0.5*(f1+f2)
		ps3 = PBH(ps0, m, 10**f, aeq, h, mfac=1, iso=iso, seed=seed, dmax=dmax)
		y = msdens([ref[0]], z, eps, ps3, lm0, gf=gf, wf=wf)[0]
		while abs(np.log10(y/ref[1]))>rerr:
			if y<ref[1]:
				f1 = f
			else:
				f2 = f
			f = 0.5*(f1+f2)
			ps3 = PBH(ps0, m, 10**f, aeq, h, mfac=1, iso=iso, seed=seed, dmax=dmax)
			y = msdens([ref[0]], z, eps, ps3, lm0, gf=gf, wf=wf)[0]
		lf[i] = 10**f
	return lm, lf
		
mc = horizon(aeq)**3*rhom(1)/Msun*h
print('Horizon mass scale: {:.2e} Msun'.format(mc))
#lm0 = np.hstack([np.linspace(1e4, mc), np.geomspace(mc, 1e16, 1000)])
lm0 = np.hstack([np.geomspace(1e4, 1e10, 1000)[:-1], np.geomspace(1e10, 1e16, 1000)])
fb = Ob/Om
	
if __name__=="__main__":
	mode = 1
	#mdef, model = 'fof', 'press74'
	mdef, model = 'fof', 'sheth99'
	#mdef, model = 'vir', 'seppi20'
	gf, wf = gf3, wf3
	comp = 1
	plothmf = 1
	#fac = 0 #0.1
	
	eps = 1

	mpbh, fpbh = 1e10, 1e-5
	
	#sigma8 = 0.8159
	#sigma8 = 2.5
	mfac0 = 1
	
	dz = 2
	z0 = 10
	read = 1
	z = 10
	read1 = 1
	iso = 1
	seed = 0
	dmax = 0
	#dmax = (3*np.pi/4)**2+1
	#dmax = 18*np.pi**2+1
	nmd = 3
	nm = 100
	fmin = -10
	corr=10**1.6
	#repo = 'seed/'
	#repo = 'iso_v2/'
	repo = 'cut/'
	#repo = 'cut_seed/'
	#repo = 'old/final/'

	mlim = 28.04
	fn3 = 'mag_pop3.txt'
	fn2 = 'mag_pop2.txt'
	#fn1 = 'mag_pop1.txt'
	if z<=15:
		mag3 = findmag(fn3, z)
		mag2 = findmag(fn2, z)
		norm = 1
	else:
		mag3 = findmag(fn3, 10, 8) 
		mag2 = findmag(fn2, 10, 8)
		norm = (distance(z)*(1+z)/(distance(10)*11))**2
	ms3 = 1e6*10**((mag3-mlim)/2.5)*norm
	ms2 = 1e6*10**((mag2-mlim)/2.5)*norm
	print('Detection limit: {:.2e}/{:.2e} Msun (Pop III/II)'.format(ms3, ms2))
	Vcom = 40*(np.pi/(180*60))**2 * (distance(z)/MPC)**2 * (distance(z+dz/2)-distance(z-dz/2))/MPC
	
	Hp = horizon(aeq)/MPC/h
	Hp1 = horizon(1/(1+z0))/MPC/h
	print('Horizon size at a_eq: {:.2e} Mpc'.format(Hp))
	print('Horizon size at z={}: {:.2e} Mpc'.format(z0, Hp1))
	
	khp = 1/(horizon(1/(1+z0))/MPC)
	k1, k2, kc = 1e-6, 1e4, 1e-2
	lk0 = np.linspace(k1, kc*0.99, 10000)
	lk1 = np.geomspace(kc, k2, 10000)
	#lk1 = np.linspace(kc, k2, 100000)
	lk = np.hstack([lk0, lk1])
	lPk0 = cosmo.matterPowerSpectrum(lk)
	norm = 1 #(sigma8/cosmo.sigma8)**2
	print('Normalization: {:.2f}'.format(norm))
	lPk = lPk0*norm
	ps0 = [lk, lPk]

	#"""
	ps3 = PBH(ps0, mpbh, fpbh, aeq, h, mfac=mfac0, iso=iso, seed=seed, dmax=dmax)
	ms = 1e11
	mh = ms/fb
	test0 = ngem(mh*h, z0, ps0, lm0, gf=gf, wf=wf, mode=mode)*h**3
	test1 = ngem0(mh*h, z0, mdef=mdef, model=model)*h**3
	test3 = ngem(mh*h, z0, ps3, lm0, gf=gf, wf=wf, mode=mode)*h**3

	print('Number density: {:.2e} ({:.2e})/{:.2e} Mpc^-3 (CDM/PBH)'.format(test0, test1, test3))
	
	ms = ms/10**0.5
	mh = ms/fb/eps
	test0 = msdens(np.array([ms]), z0, eps, ps0, lm0, gf=gf, wf=wf)[0] #ngem(mh*h, z, ps0, lm0, gf=gf, wf=wf, mode=mode, mass=1)*h**3*fb*eps/h
	test1 = ngem0(mh*h, z0, mdef=mdef, model=model, mass=1)*h**3*fb*eps/h
	test3 = msdens(np.array([ms]), z0, eps, ps3, lm0, gf=gf, wf=wf)[0] #ngem(mh*h, z, ps3, lm0, gf=gf, wf=wf, mode=mode, mass=1)*h**3*fb*eps/h
	
	print('Stellar mass density: {:.2e} ({:.2e})/{:.2e} Msun Mpc^-3 (CDM/PBH)'.format(test0, test1, test3))
	#"""
	
	if (not os.path.exists(repo)):
		os.mkdir(repo)
	
	ref = np.array(retxt('labbe2022_z{}.txt'.format(z0), 2))
	lmsref = np.array([np.mean(ref[0][:3]), np.mean(ref[0][3:])])
	lrhoref = np.array([ref[1][0], ref[1][3]])
	ldown = np.array([ref[1][0]-ref[1][1], ref[1][3]-ref[1][4]])
	lup = np.array([ref[1][2]-ref[1][0], ref[1][5]-ref[1][3]])
	
	mudis = np.array(retxt('mu_distortion.txt',2))
	allcos = np.array(retxt('pbh_cos.txt',2))
	
	m1, m2 = 1e5, 1e12
	rerr = 0.01

	y1, y2 = 1e-10, 1e8
	if z0>8:
		eps1, eps2, eps3 = 1, 0.1, 0.05
		y2 = 3e5
	else:
		eps1, eps2 = 0.33, 0.1
		y2 = 1e10
	refd = np.array([lmsref, [ref[1][1], ref[1][4]]])
	refd = refd.T
	ref = np.array([lmsref, lrhoref])
	ref = ref.T
	refc = np.array([lmsref/corr, lrhoref/corr])
	refc = refc.T
	
	#print(ref)
	#print(msdens([ref[0][0]], z, 1, ps3, lm0)[0], msdens([ref[0][0]], z, 1, ps0, lm0)[0])
	#test = findfpbh(ref[0], z, 1, ps0, lm0, nm=3, m1=m1, m2=m2, rerr=rerr, fmin=fmin)
	#print(test)

	md1 = [3e5, 3e-4]
	#md1 = [3e5, 1e-2]
	md2 = [1e9, 1e-5]
	md3 = [1e10, 1e-4]
	md4 = [1e11, 1e-5]
	lmd0 = np.array([md1, md2, md3])#, md4])
	lmd = lmd0.T
	
	llab0 = [r'M1, $m_{\rm PBH}=3\times 10^{5}\ \rm M_{\odot}$, $f_{\rm PBH}=3\times 10^{-4}$', r'M2, $m_{\rm PBH}=10^{9}\ \rm M_{\odot}$, $f_{\rm PBH}=10^{-5}$', r'M3, $m_{\rm PBH}=10^{10}\ \rm M_{\odot}$, $f_{\rm PBH}=10^{-4}$', r'M4, $m_{\rm PBH}=10^{11}\ \rm M_{\odot}$, $f_{\rm PBH}=10^{-5}$']
	#if seed>0:
	#	llab = llab0
	#else:
	#llab = [r'M1, $m_{\rm PBH}f_{\rm PBH}=90\ \rm M_{\odot}$', r'M2, $m_{\rm PBH}f_{\rm PBH}=10000\ \rm M_{\odot}$', r'M3, $m_{\rm PBH}f_{\rm PBH}=10^{6}\ \rm M_{\odot}$']
	llab = [r'M1', r'M2', r'M3', r'M4']

	rhomax = rhom(1)*fb*Om/(Om-Ob)/Msun*MPC**3
	print('Maximum stellar mass density in collapsed PBH clusters: rho/f_PBH/eps = {:.2e} Msun Mpc^-3'.format(rhomax))

	x1, x2 = 1, 21
	lz = np.geomspace(x1, x2, 100)-1
	plt.figure()
	for i in range(nmd):
		lmbp = mbind(lz, *lmd0[i])
		lmbs = mbind(lz, *lmd0[i], seed=1)
		plt.loglog(lz+1, lmbp, ls=lls[i+1], color=llc[i], label=llab0[i])
		plt.loglog(lz+1, lmbs, ls=lls[i+1], color=llc[i], lw=4.5, alpha=0.3)
	plt.xlabel(r'$1+z$')
	plt.ylabel(r'$M_{\rm B}\ [\rm M_{\odot}]$')
	plt.legend()
	plt.xlim(x1, x2)
	plt.tight_layout()
	plt.savefig(repo+'mb_z.pdf')
	plt.close()
	
	if read==0:
		lm, lf00 = findfpbh(ref[0], z0, eps1, ps0, lm0, nm=nm, m1=m1, m2=m2, rerr=rerr, fmin=fmin, iso=iso, gf=gf, wf=wf, seed=seed, dmax=dmax)
		lm, lf01 = findfpbh(ref[1], z0, eps1, ps0, lm0, nm=nm, m1=m1, m2=m2, rerr=rerr, fmin=fmin, iso=iso, gf=gf, wf=wf, seed=seed, dmax=dmax)
		lm, lf10 = findfpbh(ref[0], z0, eps2, ps0, lm0, nm=nm, m1=m1, m2=m2, rerr=rerr, fmin=fmin, iso=iso, gf=gf, wf=wf, seed=seed, dmax=dmax)
		lm, lf11 = findfpbh(ref[1], z0, eps2, ps0, lm0, nm=nm, m1=m1, m2=m2, rerr=rerr, fmin=fmin, iso=iso, gf=gf, wf=wf, seed=seed, dmax=dmax)
		#lm, lf20 = findfpbh(refc[0], z0, eps3, ps0, lm0, nm=nm, m1=m1, m2=m2, rerr=rerr, fmin=fmin, iso=iso, gf=gf, wf=wf, seed=seed)
		#lm, lf21 = findfpbh(refc[1], z0, eps3, ps0, lm0, nm=nm, m1=m1, m2=m2, rerr=rerr, fmin=fmin, iso=iso, gf=gf, wf=wf, seed=seed)
		totxt(repo+'f_m_z{}.txt'.format(z0), [lm, lf00, lf01, lf10, lf11])#, lf20, lf21])
	else:
		lm, lf00, lf01, lf10, lf11 = np.array(retxt(repo+'f_m_z{}.txt'.format(z0),5))
	if iso>0 and seed==0 and nm<10:
		lf00[0] = lf00[-1]*lm[-1]/lm[0]
		lf01[0] = lf01[-1]*lm[-1]/lm[0]
		lf10[0] = lf10[-1]*lm[-1]/lm[0]
		lf11[0] = lf11[-1]*lm[-1]/lm[0]
		#lf20[0] = lf20[-1]*lm[-1]/lm[0]
		#lf21[0] = lf21[-1]*lm[-1]/lm[0]
	plt.figure()
	ax1 = plt.subplot(111)
	if np.max(lf00)>max(y1,10**fmin):# and (lf00[-1]*lm[-1]>100):
		plt.loglog(lm, lf01, ls='-', label=r'$\epsilon={}$'.format(eps1)+r', $M_{\star}\sim 10^{10-10.5}\ \rm M_{\odot}$')
	if np.max(lf01)>max(y1,10**fmin):# and (lf01[-1]*lm[-1]>100):
		plt.loglog(lm, lf00, ls='-', color=llc[0])#, label=r'$\epsilon={}$'.format(eps1))#+r', $M_{\star}\sim 10^{10}\ \rm M_{\odot}$')
	plt.loglog(lm, lf11, ls='--', color=llc[1], label=r'$\epsilon={}$'.format(eps2)+r', $M_{\star}\sim 10^{10-10.5}\ \rm M_{\odot}$')#, lw=4.5, alpha=0.5)
	plt.loglog(lm, lf10, ls='--', color=llc[1])#, label=r'$\epsilon={}$'.format(eps2)+r', $M_{\star}\sim 10^{10}\ \rm M_{\odot}$', lw=4.5, alpha=0.5)
	#if np.max(lf21)>max(y1,10**fmin) and (lf21[-1]*lm[-1]>100):
	#	plt.loglog(lm, lf21, ls=':', color=llc[7], label=r'$\epsilon={}$'.format(eps3)+r', $M_{\star}\sim 10^{8.9}\ \rm M_{\odot}$')#, lw=4.5, alpha=0.5)
	#if np.max(lf20)>max(y1,10**fmin) and (lf20[-1]*lm[-1]>100):
	#	plt.loglog(lm, lf20, ls=':', color=llc[7])#, label=r'$\epsilon={}$'.format(eps2)+r', $M_{\star}\sim 10^{10}\ \rm M_{\odot}$', lw=4.5, alpha=0.5)

	plt.loglog(*allcos, '-.', color=llc[2], label='XB+DF+LSS (Carr 2021)')
	plt.loglog(*mudis, color=llc[3], ls=lls[-2], label='FIRAS, $p=0.5$ (Nakama 2018)')
	plt.scatter(*lmd, marker='^', color=['k', llc[6], llc[6]], s=[48,48,96])#, 'k'], s=[48,48,96,96])
	[plt.plot([m1,m2], [10**(-x)]*2, 'k:', alpha=0.2) for x in range(10)]
	[plt.plot([10**x]*2, [1e-10, 1], 'k:', alpha=0.2) for x in range(6,12)]
	plt.fill_between([m1, m2], [y2]*2, [1]*2, fc='gray', alpha=0.2)
	plt.text(m1*2, y1*10, r'$z={}$'.format(z0))
	for i in range(len(lmd0)):
		md = lmd0[i]
		plt.text(md[0], md[1]/20, 'M{}'.format(i+1))
	plt.xlabel(r'$m_{\rm PBH}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$f_{\rm PBH}$')
	ax1.set_yticks(np.geomspace(1e-10, 1, 6))
	ax1.set_xticks(np.geomspace(1e5, 1e12, 8))
	plt.xlim(m1, m2)
	plt.ylim(y1, y2)
	plt.legend(loc=1, ncol=1)
	plt.tight_layout()
	plt.savefig(repo+'fpbh_mpbh_z{}.pdf'.format(z0))
	plt.close()

	print(lf00[-1]*lm[-1])
	print(lf01[-1]*lm[-1])
	print(lf10[-1]*lm[-1])
	print(lf11[-1]*lm[-1])
	#print(lf20[-1]*lm[-1])
	#print(lf21[-1]*lm[-1])

	#exit()
	
	lps = [PBH(ps0, md[0], md[1], aeq, h, mfac=mfac0, iso=iso, seed=seed, dmax=dmax) for md in lmd0]
	if read1>0:
		#if seed>0:
		#	out = np.array(retxt(repo+'rhoms_eps1_z{}.txt'.format(z),6))
		#	out0 = np.array(retxt(repo+'rhoms_eps0.1_z{}.txt'.format(z),6))
		#else:
		out = np.array(retxt(repo+'rhoms_eps1_z{}.txt'.format(z),2+nmd))
		out0 = np.array(retxt(repo+'rhoms_eps0.1_z{}.txt'.format(z),2+nmd))
		lms = out[0]
		lrho0 = out[1]
		lrho00 = out0[1]
		x1, x2 = np.min(lms), np.max(lms)
	else:
		nms = 100
		x1, x2 = 1e5, 1e11
		lms = np.geomspace(x1, x2, nms+1)
		out = [lms]
		out0 = [lms]
		lrho0 = msdens(lms, z, 1, ps0, lm0, gf=gf, wf=wf)
		lrho00 = msdens(lms, z, 0.1, ps0, lm0, gf=gf, wf=wf)
		out.append(lrho0)
		out0.append(lrho00)

	y1, y2 = 1e3, 1e11
	plt.figure()
	ax1 = plt.subplot(111)
	#plt.loglog([x1, x2], [rhom(1)*fpbh*fb*Om/(Om-Ob)/Msun*MPC**3]*2, 'k', lw=0.5)
	plt.loglog(lms, lrho0, 'k-', label=r'$\Lambda$CDM, $\epsilon=1$')
	plt.loglog(lms, lrho00, 'k-', label=r'$\epsilon=0.1$', lw=4.5, alpha=0.5)
	if z==10 or z==8:
		plt.errorbar(lmsref, lrhoref, [ldown,lup], fmt='o',color='k',label=r'$\rm Labb\'e$ 2022 ($z={}$)'.format(z))
		#plt.errorbar(lmsref/corr, lrhoref/corr, [ldown/corr,lup/corr], fmt='o',color=llc[6])#,label=r'$\rm Labb\'e$ 2022 ($z={}$)'.format(z))
	for j in range(nmd):
		#if nmd>3:
		#if seed>0:
		#	i = 2-j
		#	j += 1
		#else:
		i = j
		if read1==0:
			lrho3 = msdens(lms, z, 1, lps[i], lm0, gf=gf, wf=wf)
			lrho30 = msdens(lms, z, 0.1, lps[i], lm0, gf=gf, wf=wf)
			out.append(lrho3)
			out0.append(lrho30)
		else:
			lrho3 = out[2+j]
			lrho30 = out0[2+j]
		plt.loglog(lms, lrho3, color=llc[i], ls=lls[i+1], label=llab[i])
		#=r'$m_{\rm PBH}=10^{'+str(np.log10(lmpbh[i]))+r'}\ \rm M_{\odot}$')
		plt.loglog(lms, lrho30, color=llc[i], ls=lls[i+1], lw=4.5, alpha=0.5)
	plt.fill_between(lms, lrho0, out[nmd+i-1], fc='k', alpha=0.1)
	plt.plot([ms3]*2, [y1,y2], color=llc[7], ls=(0,(10,5)), alpha=0.5)
	plt.plot([ms2]*2, [y1,y2], color=llc[7], ls=(0,(10,5)), alpha=0.5)
	#if z==15:
	plt.loglog(lms, lms/Vcom, color=llc[5], ls=lls[-1], lw=4.5)
	plt.text(ms3*1.2, y2/3, 'Pop III', color=llc[7])
	plt.text(ms2*1.2, y2/3, 'Pop II', color=llc[7])
	plt.text(x2/20, lms[-1]/Vcom/200, r'40 arcmin$^2$'+'\n$\Delta z={}$'.format(dz), color=llc[5])
	plt.xlabel(r'$M_{\star}=\epsilon f_{\rm b}M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$\rho_{\star}(>M_{\star})=\epsilon f_{\rm b}\rho(>M_{\rm halo})\ [\rm M_{\odot}\ Mpc^{-3}]$')
	plt.ylim(y1, y2)
	plt.xlim(x1, x2)
	ax1.set_yticks(np.geomspace(1e3, 1e11, 9))
	#if nmd<4:
	ax1.set_xticks(np.geomspace(1e5, 1e11, 7))
	#else:
		#ax1.set_xticks(np.geomspace(1e4, 1e11, 8))
	if (z==10 or z==8):# and (nmd<4):
		plt.legend(ncol=1, loc=3)
	plt.text(x2/10, y2/5, r'$z={}$'.format(z))
	#plt.text(2*x1, 5*y1, r'$f_{\rm PBH}=10^{'+str(np.log10(fpbh))+r'}$')
	plt.tight_layout()
	plt.savefig(repo+'rhoms_z{}.pdf'.format(z))
	plt.close()
	if read1==0:
		totxt(repo+'rhoms_eps1_z{}.txt'.format(z), out)
		totxt(repo+'rhoms_eps0.1_z{}.txt'.format(z), out0)

	#"""
	#kH = MPC/horizon(aeq)
	
	#if seed<=0:
	#llab = [r'$m_{\rm PBH}f_{\rm PBH}=90\ \rm M_{\odot}$', r'$m_{\rm PBH}f_{\rm PBH}=10000\ \rm M_{\odot}$', r'$m_{\rm PBH}f_{\rm PBH}=10^{6}\ \rm M_{\odot}$']
	llab =  [r'$m_{\rm PBH}=3\times 10^{5}\ \rm M_{\odot}$, $f_{\rm PBH}=0.0003$', r'$m_{\rm PBH}=10^{9}\ \rm M_{\odot}$, $f_{\rm PBH}=10^{-5}$', r'$m_{\rm PBH}=10^{10}\ \rm M_{\odot}$, $f_{\rm PBH}=10^{-4}$', r'$m_{\rm PBH}=10^{11}\ \rm M_{\odot}$, $f_{\rm PBH}=10^{-5}$']

	klm = [k1, k2] #[1e-4, 1e4] #[1e-4, 1e2]
	Plm = [1e-4, 1e5] #[1e-3, 5e4]
	plt.figure()
	ax1 = plt.subplot(111)
	plt.loglog(*ps0, 'k-', label=r'$\Lambda$CDM')
	for j in range(nmd):
		#if seed>0:
		#	i = nmd-j-1
		#else:
		i = j
		ps3 = lps[i]
		plt.loglog(*ps3, color=llc[i], ls=lls[i+1], label=llab[i])
		#r'$m_{\rm PBH}=10^{'+str(np.log10(lmpbh[i]))+r'}\ \rm M_{\odot}$')
	#plt.loglog(kh, pk[0], color='gray', lw=4.5, alpha=0.5, label='CDM (linear), CAMB')
	#plt.loglog(kh_nonlin, pk_nonlin[0], ':', label='CDM (non-linear), CAMB')
	#plt.plot([kH]*2, Plm, 'k-', lw=0.5)
	plt.xlabel(r'$k\ [h\ \mathrm{Mpc^{-1}}]$', size=16)
	plt.ylabel(r'$P(k)\ [h^{-3}\ \mathrm{Mpc^{3}}]$', size=16)
	#plt.xticks(fontsize= 16)
	#plt.yticks(fontsize= 16)
	plt.ylim(Plm)
	ax1.set_yticks(np.geomspace(1e-4, 1e5, 10))
	if nmd<4:
		plt.xlim(1e-3, 1e3)
		ax1.set_xticks(np.geomspace(1e-3, 1e3, 7))
	else:
		plt.xlim(klm)
		ax1.set_xticks(np.geomspace(1e-6, 1e4, 11))
	#plt.title('z={}'.format(zps))
	#plt.text(k2/10, Plm[1]/3, r'$z={}$'.format(z))
	plt.legend(loc=3)#fontsize=16, loc=3)
	plt.tight_layout()
	plt.savefig(repo+'powspec.pdf')
	plt.close()
	
	exit()


	
	#read = 1

	fpbh = 1e-5
	lmpbh = [1e8, 1e9, 1e10, 1e11, 1e12]
	lps = [PBH(ps0, mpbh, fpbh, aeq, h, mfac=mfac0, iso=iso, seed=seed, dmax=dmax) for mpbh in lmpbh]

	if read>0:
		out = np.array(retxt(repo+'rhoms_mpbh_eps1_z{}.txt'.format(z),2+len(lmpbh)))
		out0 = np.array(retxt(repo+'rhoms_mpbh_eps0.1_z{}.txt'.format(z),2+len(lmpbh)))
		lms = out[0]
		lrho0 = out[1]
		lrho00 = out0[1]
		x1, x2 = np.min(lms), np.max(lms)
	else:
		nms = nm
		x1, x2 = 1e7, 1e11
		lms = np.geomspace(x1, x2, nms+1)
		out = [lms]
		out0 = [lms]
		lrho0 = msdens(lms, z, 1, ps0, lm0, gf=gf, wf=wf)
		lrho00 = msdens(lms, z, 0.1, ps0, lm0, gf=gf, wf=wf)
		out.append(lrho0)
		out0.append(lrho00)

	y1, y2 = 100, 3e13
	plt.figure()
	ax1 = plt.subplot(111)
	#plt.loglog([x1, x2], [rhom(1)*fpbh*fb*Om/(Om-Ob)/Msun*MPC**3]*2, 'k', lw=0.5)
	plt.loglog(lms, lrho0, 'k-', label=r'$\Lambda$CDM, $\epsilon=1$')
	plt.loglog(lms, lrho00, 'k-', label=r'$\epsilon=0.1$', lw=4.5, alpha=0.3)
	for i in range(len(lmpbh)):
		if read==0:
			lrho3 = msdens(lms, z, 1, lps[i], lm0, gf=gf, wf=wf)
			lrho30 = msdens(lms, z, 0.1, lps[i], lm0, gf=gf, wf=wf)
			out.append(lrho3)
			out0.append(lrho30)
		else:
			lrho3 = out[2+i]
			lrho30 = out0[2+i]
		plt.loglog(lms, lrho3, color=llc[i], ls=lls[i+1], label=r'$m_{\rm PBH}=10^{'+str(np.log10(lmpbh[i]))+r'}\ \rm M_{\odot}$')
		plt.loglog(lms, lrho30, color=llc[i], ls=lls[i+1], lw=4.5, alpha=0.3)
	plt.errorbar(lmsref, lrhoref, [ldown,lup], fmt='o',color='r',label=r'$\rm Labb\'e$ 2022 ($z={}$)'.format(z))
	plt.xlabel(r'$M_{\star}=\epsilon f_{\rm b}M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$\rho(>M_{\star})=\epsilon f_{\rm b}\rho(>M_{\rm halo})\ [\rm M_{\odot}\ Mpc^{-3}]$')
	plt.ylim(y1, y2)
	plt.xlim(x1, x2)
	ax1.set_yticks(np.geomspace(1e2, 1e13, 12))
	plt.legend(ncol=2, loc=2)
	plt.text(2*x1, 5*y1, r'$f_{\rm PBH}=10^{'+str(np.log10(fpbh))+r'}$')
	plt.tight_layout()
	plt.savefig(repo+'rhoms_mpbh_z{}.pdf'.format(z))
	plt.close()
	if read==0:
		totxt(repo+'rhoms_mpbh_eps1_z{}.txt'.format(z), out)
		totxt(repo+'rhoms_mpbh_eps0.1_z{}.txt'.format(z), out0)

	#"""
	kH = MPC/horizon(aeq)
	klm = [k1, k2] #[1e-4, 1e4] #[1e-4, 1e2]
	Plm = [1e-8, 3e5] #[1e-3, 5e4]
	plt.figure()
	plt.loglog(*ps0, 'k-', label=r'$\Lambda$CDM')
	for i in range(len(lmpbh)):
		ps3 = lps[i]
		plt.loglog(*ps3, color=llc[i], ls=lls[i+1], label=r'$m_{\rm PBH}=10^{'+str(np.log10(lmpbh[i]))+r'}\ \rm M_{\odot}$')
	#plt.loglog(kh, pk[0], color='gray', lw=4.5, alpha=0.5, label='CDM (linear), CAMB')
	#plt.loglog(kh_nonlin, pk_nonlin[0], ':', label='CDM (non-linear), CAMB')
	plt.plot([kH]*2, Plm, 'k-', lw=0.5)
	plt.xlabel(r'$k\ [h\ \mathrm{Mpc^{-1}}]$', size=16)
	plt.ylabel(r'$P(k)\ [h^{-3}\ \mathrm{Mpc^{3}}]$', size=16)
	#plt.xticks(fontsize= 16)
	#plt.yticks(fontsize= 16)
	plt.xlim(klm)
	plt.ylim(Plm)
	#plt.title('z={}'.format(zps))
	plt.text(3*k1, Plm[1]/10, r'$f_{\rm PBH}=10^{'+str(np.log10(fpbh))+'}$')
	plt.legend()#fontsize=16, loc=3)
	plt.tight_layout()
	plt.savefig(repo+'powspec_mpbh.pdf')
	plt.close()
	#"""

	mpbh = 1e10
	lfpbh = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
	lps = [PBH(ps0, mpbh, fpbh, aeq, h, mfac=mfac0, iso=iso, seed=seed, dmax=dmax) for fpbh in lfpbh]

	if read>0:
		out = np.array(retxt(repo+'rhoms_fpbh_eps1_z{}.txt'.format(z),2+len(lfpbh)))
		out0 = np.array(retxt(repo+'rhoms_fpbh_eps0.1_z{}.txt'.format(z),2+len(lfpbh)))
		lms = out[0]
	else:
		out = [lms, lrho0]
		out0 = [lms, lrho00]

	plt.figure()
	ax1 = plt.subplot(111)
	plt.loglog(lms, lrho0, 'k-', label=r'$\Lambda$CDM, $\epsilon=1$')
	plt.loglog(lms, lrho00, 'k-', label=r'$\epsilon=0.1$', lw=4.5, alpha=0.3)
	for i in range(len(lfpbh)):
		if read==0:
			lrho3 = msdens(lms, z, 1, lps[i], lm0, gf=gf, wf=wf)
			lrho30 = msdens(lms, z, 0.1, lps[i], lm0, gf=gf, wf=wf)
			out.append(lrho3)
			out0.append(lrho30)
		else:
			lrho3 = out[2+i]
			lrho30 = out0[2+i]
		plt.loglog(lms, lrho3, color=llc[i], ls=lls[i+1], label=r'$f_{\rm PBH}=10^{'+str(np.log10(lfpbh[i]))+r'}$')
		plt.loglog(lms, lrho30, color=llc[i], ls=lls[i+1], lw=4.5, alpha=0.3)
	plt.errorbar(lmsref, lrhoref, [ldown,lup], fmt='o',color='r',label=r'$\rm Labb\'e$ 2022 ($z={}$)'.format(z))
	plt.xlabel(r'$M_{\star}=\epsilon f_{\rm b}M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$\rho(>M_{\star})=\epsilon f_{\rm b}\rho(>M_{\rm halo})\ [\rm M_{\odot}\ Mpc^{-3}]$')
	plt.ylim(y1, y2)
	plt.xlim(x1, x2)
	ax1.set_yticks(np.geomspace(1e2, 1e13, 12))
	plt.legend(ncol=2, loc=2)
	plt.text(2*x1, 5*y1, r'$m_{\rm PBH}=10^{'+str(np.log10(mpbh))+r'}\ \rm M_{\odot}$')
	plt.tight_layout()
	plt.savefig(repo+'rhoms_fpbh_z{}.pdf'.format(z))
	plt.close()
	if read==0:
		totxt(repo+'rhoms_fpbh_eps1_z{}.txt'.format(z), out)
		totxt(repo+'rhoms_fpbh_eps0.1_z{}.txt'.format(z), out0)

	#"""
	klm = [k1, k2] #[1e-4, 1e4] #[1e-4, 1e2]
	Plm = [1e-8, 3e5] #[1e-3, 5e4]
	plt.figure()
	plt.loglog(*ps0, 'k-', label=r'$\Lambda$CDM')
	for i in range(len(lmpbh)):
		ps3 = lps[i]
		plt.loglog(*ps3, color=llc[i], ls=lls[i+1], label=r'$f_{\rm PBH}=10^{'+str(np.log10(lfpbh[i]))+r'}$')
	#plt.loglog(kh, pk[0], color='gray', lw=4.5, alpha=0.5, label='CDM (linear), CAMB')
	#plt.loglog(kh_nonlin, pk_nonlin[0], ':', label='CDM (non-linear), CAMB')
	plt.plot([kH]*2, Plm, 'k-', lw=0.5)
	plt.xlabel(r'$k\ [h\ \mathrm{Mpc^{-1}}]$', size=16)
	plt.ylabel(r'$P(k)\ [h^{-3}\ \mathrm{Mpc^{3}}]$', size=16)
	#plt.xticks(fontsize= 16)
	#plt.yticks(fontsize= 16)
	plt.xlim(klm)
	plt.ylim(Plm)
	#plt.title('z={}'.format(zps))
	plt.text(3*k1, Plm[1]/10, r'$m_{\rm PBH}=10^{'+str(np.log10(mpbh))+r'}\ \rm M_{\odot}$')
	plt.legend()#fontsize=16, loc=3)
	plt.tight_layout()
	plt.savefig(repo+'powspec_fpbh.pdf')
	plt.close()
	#"""
	
