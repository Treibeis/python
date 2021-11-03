import numpy as np
#from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root
from scipy.optimize import curve_fit
from scipy.misc import derivative
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
plt.xlabel(r'$k\ [h\ \mathrm{Mpc^{-1}}]$', size=18)
plt.ylabel(r'$P(k)\ [h^{-3}\ \mathrm{Mpc^{3}}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.xlim(klm)
plt.ylim(Plm)
#plt.title('z={}'.format(zps))
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('powspec.pdf')
plt.close()

delta_plus0 = delta_plus(0, Om, 1-Om)
def Dgrow_(z, Om = Om, Ol = 1-Om):
	return delta_plus(z, Om, Ol)/delta_plus0

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

lz = np.linspace(0, 50, 200)
d = Om*(1+lz)**3/(1-Om+Om*(1+lz)**3)-1
do = 18*np.pi**2 + 82*d-39*d**2
plt.figure()
plt.plot(1/(1+lz), [deltac_z(z)*Dgrow(z, Om, 1-Om) for z in lz], label=r'$\delta_{\rm c}$')
plt.plot(1/(1+lz), do/100, '--', label=r'$\Delta_{\rm vir}/100$')
plt.xlabel(r'$a$', size=18)
plt.ylabel(r'Overdensity', size=18)
plt.legend(fontsize=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('deltac_a.pdf')
plt.close()

plt.figure()
plt.plot(1/(1+lz), [Dgrow_(z, Om, 1-Om) for z in lz], label='Numerical solution')
plt.plot(1/(1+lz), Dgrow(lz, Om, 1-Om), '--', label='Fit')
plt.plot(1/(1+lz), 1/(1+lz), '-.', label=r'$D(a)\propto a$')
plt.xlabel(r'$a$', size=18)
plt.ylabel(r'$D(a)/D(a=0)$', size=18)
plt.legend(fontsize=18, loc=4)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('Dgrow_a.pdf')
plt.close()

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

mode = 1 # 0: standard PS + LW, 1: PS with elliptical dynamics + LW, 2: PS with... no LW

m1, m2 = 1e4*h, 1e16*h
lm = np.geomspace(m1, m2, 100)
lz = [0, 5, 10, 20, 30]
lhmf = []
for z in lz:
	hmf = halomassfunc(lm0, z, ps0, wf1, gf1, mode=mode)
	lhmf.append(np.array([hmf(m) for m in lm]))
	
plt.figure()
for z, f in zip(lz, lhmf):
	plt.loglog(lm/h, lm*f*h**3*np.log(10), label=r'$z={:.0f}$'.format(z))
plt.legend(fontsize=18)
plt.xlim(m1/h, m2/h)
plt.ylim(1e-9, 1e6)
plt.xlabel(r'$M\ [\mathrm{M_{\odot}}]$', size=18)
plt.ylabel(r'$dn/d\log M\ [\mathrm{Mpc^{-3}}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.tight_layout()
plt.savefig('mhmf.pdf')
plt.close()

plt.figure()
for z, f in zip(lz, lhmf):
	plt.loglog(lm/h, lm**2*f/(rho0/UM*UL**3)*h**2, label=r'$z={:.0f}$'.format(z))
plt.legend(fontsize=18)
plt.xlim(m1/h, m2/h)
plt.ylim(1e-16, 0.1)
plt.xlabel(r'$M\ [\mathrm{M_{\odot}}]$', size=18)
plt.ylabel(r'$M^{2}dn/dM/\rho_{\rm m}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.tight_layout()
plt.savefig('m2hmf.pdf')
plt.close()

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

def sfrtf(z, a, b, c, d):
	#t = 1/(z+1)
	#return a*(t**b*np.exp(-t/c)+d*np.exp(d*(t-1)/c)) 
	return a*(1+z)**b/(1+((1+z)/c)**d)

lsfrd0 = retxt('pop3sfrd_z_0.txt',2)
lsfrd1 = retxt('pop3sfrd_z_1.txt',2)
lsfrd2 = retxt('pop3sfrd_z_2.txt',2)

lz = np.linspace(5, 30, 251)
para = [765.7, -5.92, 12.83, -8.55]
lsfrd = sfrtf(lz, *para)
plt.figure()
#plt.plot(*lsfrd0, label='Standard PS')
plt.plot(*lsfrd1, '-', label='PS + LW feedback')
plt.plot(*lsfrd2, '--', label='PS no LW feedback')
plt.plot(lz, lsfrd, '-.', label='Simulation (LB20)')
plt.legend(loc=3, fontsize=18)
plt.xlabel(r'$z$', size=18)
plt.ylabel(r'$\dot{\rho}_{\star,\rm PopIII}\ [\rm M_{\odot}\ yr^{-1}\ Mpc^{-3}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.yscale('log')
plt.ylim(1e-7, 1e-3)
plt.xlim(5, 30)
plt.tight_layout()
plt.savefig('pop3sfrd_z.pdf')
plt.close()

#exit()

#"""
lwbg1 = np.array(retxt('z_FLW_IMFmin1.0E+00IMFmax1.7E+02eta1.0E-03slope0.0E+00.dat',2,2)) # read F21 data from merger trees
lwbg2 = np.array(retxt('FLW_z.txt',2)) # read F21 data from simulations
lz, lf = lwbg2
ind0, ind = 17, 20
f210 = lf[ind0]
z0 = lz[ind0]
alp = np.log10(lf[ind]/lf[ind0])/(lz[ind]-lz[ind0])
f21ex = lambda z: f210*10**(alp*(z-z0))
sel = lwbg2[1]<=0
lwbg2[1][sel] = f21ex(lwbg2[0][sel])

F21_z1 = interp1d(*lwbg1)
F21_z2 = interp1d(*lwbg2)

plt.figure()
plt.plot(lz[lz<25], lf[lz<25], 'm-', label='Simulation')
plt.plot(lz[lz>21], f21ex(lz[lz>21]), 'm--')
plt.plot(*lwbg1, 'b:', label='Merger trees')
plt.xlabel(r'$z$', size=18)
plt.ylabel(r'$F_{\rm 21}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(fontsize=18)
plt.yscale('log')
plt.xlim(5, 30)
plt.ylim(1e-5, 1e2)
plt.tight_layout()
plt.savefig('F21_z.pdf')
plt.close()


def flw(m,z,mass=0,z1=4,z2=31):
	if z>z1 and z<z2:
		#f21 = max(F21_z2(z),F21_z1(z)) # You may choose one LW background or combine the two
									   # Here I use the minimum to be conservative
		f21 = F21_z2(z)
	else:
		f21 = 0
	if f21<=0:
		return m/m
	mcrit = 1.25e5 + 8.7e5*f21**0.47
	if mass>0: # count the num of pop3 hosts
		y = 0.06*np.log(m/mcrit) # This formula is negative for m<mcrit, which
								 # is unphysical, should set it to 0 in that case
	else: # apply the reduction of pop3 stellar mass by LW feedback
		y = m/m
	y[m<mcrit] = 0
	if mode<2:
		return y
	else:
		return m/m   # turn off LW feedback (mode=2)

lz = np.linspace(5, 30, 251)
lmh = [1e6, 1e7, 1e8]
lls = ['-', '--', '-.']
plt.figure()
dy = np.array([flw(np.array(lmh), z, 1) for z in lz]).T
for i in range(len(lmh)):
	plt.plot(lz, dy[i], ls=lls[i], label=r'$M={:.1e}\ \rm {}$'.format(lmh[i], r'M_{\odot}'))
plt.xlabel(r'$z$', size=18)
plt.ylabel(r'$f_{\rm LW}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(fontsize=18)
plt.ylim(0,0.5)
plt.xlim(5, 30)
plt.tight_layout()
plt.savefig('fLW_M_z.pdf')
plt.close()
	
exit()

print('mode: ',mode)
lmass = np.geomspace(1e6*h, 1e8*h, 100)
Vcom = 30.3*h**3 # in unit of h^-3 Mpc^3 to be consistent with the hmf
nz = 25
lz = np.linspace(5, 30, nz+1)
msmax = 1000 # Maximum pop3 stellar mass in a halo (implied by simulations)
eta0 = 0.001
nhalo = 0
mstar = 0
lsfrd = np.zeros(nz)
for i in range(nz):
	hmf0 = halomassfunc(lm0,lz[i],ps0,wf1,gf1,mode=mode)   # n(m,z_i)
	hmf1 = halomassfunc(lm0,lz[i+1],ps0,wf1,gf1,mode=mode) # n(m,z_{i+1})
	dnh = np.array([hmf0(m) for m in lmass]) - np.array([hmf1(m) for m in lmass]) # dn(m, z_i)
	dnh[dnh<0] = 0
	dnhalo = np.trapz(dnh*flw(lmass/h,lz[i]), lmass)*Vcom # num of pop3 hosts formed in this z bin
	lms = flw(lmass/h,lz[i],1)*eta0*Ob/Om*lmass/h # stellar masses of pop3 hosts
	lms[lms>msmax] = msmax # Apply the maximum stellar mass
	dmstar = np.trapz(dnh*lms, lmass)*Vcom # total stellar mass of pop3 stars formed in this z bin
	nhalo += dnhalo
	mstar += dmstar
	print('z~{}-{}: delta nhalo: {}, delta mstar: {:.2e} Msun'.format(lz[i], lz[i+1], dnhalo, dmstar))
	lsfrd[i] = dmstar*h**3/Vcom*YR/(age_a(1/(1+lz[i]))-age_a(1/(1+lz[i+1])))
print('Total number of pop3 hosts: {:.0f}'.format(nhalo))
print('Total mass of pop3 stars: {:.2e} Msun'.format(mstar))

totxt('pop3sfrd_z_'+str(mode)+'.txt',[lz[:-1], lsfrd])

exit()
#"""
