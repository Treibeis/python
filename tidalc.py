#from cosmology import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.misc import derivative
#plt.style.use('test2')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# This fontstyle looks better than the default one in the context of MNRAS

GRA = 6.672e-8
PROTON = 1.6726e-24
SPEEDOFLIGHT = 2.99792458e+10
ELECTRON = 9.10938356e-28
HBAR = 1.05457266e-27
PLANCK = HBAR*2*np.pi
CHARGE = 4.80320451e-10
SIGMATH = 8*np.pi*(CHARGE**2/ELECTRON/SPEEDOFLIGHT**2)**2/3
Msun = 1.989e33
Rsun = 6.96342e10
Lsun = 3.828e33
YR = 3600*24*365
PC = 3.085678e18

def sumy(l):
	return np.array([np.min(l), np.max(l), np.average(l), np.median(l)])

def retxt(s, n, k = 0, t = 0):
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

# Tanikawa et al. (2020):
#ind = 0
#fac = 1
#m1, m2 = 1, 110
#nb = 10000
#lmi = np.linspace(m1, m2, nb+1)
#lmf = np.array([Mf_Mi(m,fac=fac)[ind] for m in lmi])

# Please use the following initial-remnant mass relation
# See https://iopscience.iop.org/article/10.1086/338487
# Heger & Woosley (2002):
lmi0, lmf0 = hw02 = np.array(retxt('mass_star_remnant.dat', 2))
sel = lmi0<101
sel[4] = 0
lmi = lmi0[sel]
lmf = lmf0[sel]

#lmf[4] = 2.1
mi_mf = interp1d(lmi, lmf)
mf_mi = interp1d(lmf, lmi)

print('Transition mass: {:.1f} - {:.1f} Msun'.format(mf_mi(2), mf_mi(3)))
#print(mi_mf(6))
# Now the NS-BH transition mass will become 18-27 Msun, we can use 20 Msun as an approximation/example in the calculation of f_NS

m1, m2 = 0.57, 65
nb = 10000
lm = np.linspace(m1, m2, nb)
y1, y2 = 1, 120
x1, x2 = 0.5, 80
log = 1
# We can replace update Fig. 2 with the following plot 
plt.figure()
plt.plot(lmf0, lmi0, 'o', label='HW02')
plt.plot(lm, mf_mi(lm), label='Interpolation')
plt.fill_between([2,3], [y1]*2, [y2]*2, fc='gray', alpha=0.5, label='NS-BH transition')
plt.xlabel(r'$m_{\rm f}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$m_{\rm i}\ [\rm M_{\odot}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(loc=4, fontsize=18)
plt.ylim(y1, y2)
plt.xlim(x1, x2)
if log>0:
	plt.xscale('log')
	plt.yscale('log')
plt.tight_layout()
plt.savefig('mi_mf_hw02.pdf')
plt.close()

plt.figure()
plt.plot(lmi0, lmf0, 'o', label='HW02')
plt.plot(mf_mi(lm), lm, label='Interpolation')
plt.fill_between([y1,y2], [2]*2, [3]*2, fc='gray', alpha=0.5, label='NS-BH transition', lw=0.0)
plt.xlabel(r'$m_{\rm i}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$m_{\rm f}\ [\rm M_{\odot}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(loc=2, fontsize=18)
plt.xlim(y1, y2)
plt.ylim(x1, x2)
if log>0:
	plt.xscale('log')
	plt.yscale('log')
plt.tight_layout()
plt.savefig('mf_mi_hw02.pdf')
plt.close()

def T2_eta(eta, mode=0):
	if mode==0:
		c, eta0, g, b, s = 2.58, 1.73, -4.36, 2.82, 9.91 
		r1, k = 4.5, 4
		y = (0.5-0.5*np.tanh(k*(eta/r1-1)))
	else:
		c, eta0, eta1, g = 0.17, 1.07, 1.92, -3.83
		b, b2, s, s2 = 5.5, 3.49, 3.59, 6.68
		y = (1+(eta/eta1)**s2)**((b-b2)/s2)
	y *= c*2**((b-g)/s)*(eta/eta0)**-g*(1+(eta/eta0)**s)**((g-b)/s)
	return y

"""
# reproducing Fig. B1 in Generozov et al. (2018):
x1, x2 = 0.9, 11
leta = np.geomspace(x1, x2, 100)
ly0 = T2_eta(leta, 0)
ly1 = T2_eta(leta, 1)
plt.figure()
plt.loglog(leta, ly0, label=r'$n=3/2$ ($m_{\star}\leq 0.7\ \rm M_{\odot}$)')
plt.loglog(leta, ly1, '--', label=r'$n=3$ ($m_{\star}>0.7\ \rm M_{\odot}$)')
plt.xlabel(r'$\eta=\left(\frac{m_{\star}}{m_{\star}+m_{\rm f}}\right)^{1/2}\left(\frac{r_{\rm p}}{r_{\star}}\right)^{3/2}$', size=18)
plt.ylabel(r'$T_{2}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(fontsize=18)
plt.xlim(x1, x2)
plt.ylim(1e-4, 4)
plt.tight_layout()
plt.savefig('T2_eta.pdf')
plt.close()
"""

# Edd. accretion rate in Msun/yr
def Macc_edd(MBH, eps = 0.125, mu = 1.22): 
	return 4*np.pi*GRA*PROTON*mu/SPEEDOFLIGHT/SIGMATH/eps * YR * MBH
	#return 2.2e-6 * (MBH/100)

# Lifetime when accreting at eta of the Edd. rate
def tacc_m(mbh, ms=0.5, eta=1.0):
	t0 = 1/Macc_edd(1.0)/eta
	t = np.log((mbh+ms)/mbh)*t0
	return t

# Lifetime of a XRB with a compact object mass m and a companion stellar mass ms in yr
def tlf(m, ms, ymax=10e9): 
	y1 = 4e9*2**(-2*np.log10(m))*2**(3*np.log10(ms/0.1))
	y2 = 10e9/ms**3.3
	y = y1*(ms<=1) + y2*(ms>1)
	return y*(y<ymax) + ymax*(y>=ymax)

# Life time of a TDE ...
def ttde(m, ms=0.5, fac=1e-3, beta=1.5, xi=0.2, eta=0.125, kf=0.02):
	t = 1e7/YR * beta**(-6/5)*(eta/0.1)**(3/5)
	t *= (kf/0.02)**0.5 * ms**((4-3*xi)/5)*(m/1e7)**(-2/5)
	return t*fac**(-3/5)

# XRB luminosity with a compact object mass m and a companion stellar mass ms in solar luminosity
def lum_xrb(m, ms):
	t1 = (ms-0.1)/tlf(m, ms)
	t2 = Macc_edd(m)
	return 0.1*(ms-0.1)*Msun*SPEEDOFLIGHT**2/(tlf(m, ms)*YR)/Lsun
	#return 3.2e4* m *t1/t2

# Cross sections of tidal disruption and capture
def sigmatidal(m, vrel, ms=0.5*Msun, rs=0.5*Rsun, rd=0, T2=0, fac=1.2, eps=1e-2, eps0=1e-2):
	vs = (2*GRA*ms/rs)**0.5
	if vrel>vs:
		return [0, 0]
	r0 = (m/ms)**(1/3)*rs
	if rd<=0:
		xd = 1.5*(ms/Msun<=0.7) + 1*(ms/Msun>0.7)
	else:
		xd = rd
	if T2>0:
		x = (m*vs/(ms*vrel))**(1/3)*T2**(1/6)*((m+ms)/m)**(1/6)
	else:
		x1 = (m/ms)**(1/3)*xd
		x2 = fac * (m*vs/(ms*vrel))**(1/3)*((m+ms)/m)**(1/6)
		#x0 = fac * (m/ms)**(1/3)
		x = 0.1
		def func(x):
			r = x*rs
			eta = (ms/(ms+m))**0.5*(r/rs)**(3/2)
			dE = (T2_eta(eta, ms/Msun>0.7)*GRA*ms**2/rs)**(1/6) *(r/r0)**-1
			K = (0.5*vrel**2*m*ms/(m+ms))**(1/6)
			return np.abs(dE/K-1)
		num = (x2-x1)/eps
		if num>0:
			if num<1:
				if func(0.5*(x1+x2))<=eps0:
					x = 0.5*(x2+x1)
			else:
				lx = np.linspace(x1, x2, int(num+1))
				lf = func(lx)
				x = lx[np.argmin(lf)]
			if abs(func(x))>eps0:
				#print(x, func(x), x1, x2, vrel/vs)
				x = 0.1
		#sol = root(func, [x0], method='hybr')#, options={'factor':0.1, 'eps':eps})
		#if (sol.success==False) or (abs(sol.fun)>eps):
		#	print(sol.x, x0, sol.fun, vrel/vs)
		#print(sol.fun)
		#x = sol.x[0]
		#print(x/x0)
	rt = r0*xd
	rp = x*rs
	bd = 1 #vrel**2/2<=(m/ms)**(1/3)*GRA*ms/rs
	y1 = np.pi*rt**2*(1+2*GRA*(m+ms)/(rt*vrel**2))
	y2 = np.pi*rp**2*(1+2*GRA*(m+ms)/(rp*vrel**2))
	#print(y2/y1)
	#return [bd*y1*(y2>=y1) + bd*y2*(y2<y1), (y2-y1)*(y2>y1)]
	return [bd*y1, (y2-y1)*(x>xd)] # disruption and capture cross sections
	
# Rate (per object) of tidal disruption and capture
# m: compact object mass, 
def caprate(m=150, rhos=1e6, sig=110, nb=100, x1=1e-2, x2=10, log=0, ms=0.2, rs=0.28, rd=0, T2=0):
	y = rhos/PC**3/(2*np.pi*(sig*1e5)**2)**1.5
	if log>0:
		lv = np.hstack([[x1*sig], np.geomspace(x1*sig*1e5, x2*sig*1e5, nb)])
	else:
		lv = np.linspace(x1*sig*1e5, x2*sig*1e5, nb+1)
	#s1, s2 = sigmatidal(m*Msun, lv, ms*Msun, rs*Rsun, rd, T2)
	s1, s2 = np.array([sigmatidal(m*Msun, v, ms*Msun, rs*Rsun, rd, T2) for v in lv]).T
	lint = lv*np.exp(-lv**2/(2*(sig*1e5)**2)) * 4*np.pi*lv**2
	y1 = np.trapz(s1*lint, lv)
	y2 = np.trapz(s2*lint, lv)
	return [y*y1*YR, y*y2*YR] # [disruption rate, capture rate] per yr (per object)

#exit()

"""
c1, c2 = caprate(m=150, log=0)
nbh = 50
print(1/c1/1e7/nbh, 1/c2/1e7/nbh)

vs = 600
est = 1e6/PC**3*(vs/180)**(7/3)*np.pi*(0.5*Rsun)**2*180*1e5*(150/0.5)**(4/3)*YR
print(1/est/1e7/nbh)
"""

t0 = 1e10
# Reproducing the GA18 rates:
print('GA18 reference present-day rates:\nNS: 2e-8 (XRB), 9e-8 (TDE) yr^-1, 6e-8 (XRB), 6e-7 (TDE)')
#print('GA18 reference present-day rates:\nNS: 5e-8 (XRB), 9e-8 (TDE) yr^-1, 2e-7 (XRB), 6e-7 (TDE)')
T20 = 0 #1.15 #7.5e-2
T2 = 0 #0.055 #3.5e-3
ms0 = 0.3
ms = 0.3
rd0 = 0 #1
rd = 0 #1 #1.5
sig0 = 125
sig = 192
alp = 1.27
rho0 =  6.03e5 #8.7e5
nns, nbh =  1.25e5, 0.367e4 #1.23e5, 8.55e3
nns0 = 2.3e5
nbh0 = 2e4
c = 0
c10, c20 = caprate(1.5, rho0*(1.5/10)**alp, sig0, ms=ms0, rs=ms0**0.8, rd=rd0, T2=T20)
print('Total NS capture (disruption) rate: {:.2e} ({:.2e}) yr^-1'.format(nns0*c20, nns0*c10))
print(c10/c20)
c = nns*(c10+c20)
c1, c2 = caprate(10, rho0, sig, ms=ms, rs=ms**0.8, rd=rd, T2=T2)
print('Total BH capture (disruption) rate: {:.2e} ({:.2e}) yr^-1'.format(nbh0*c2, nbh0*c1))
print(c1/c2)
c += nbh*(c1+c2)
print('Total tidal capture rate: {:.2e} yr^-1'.format(c))
print('Num. of active NS (BH) XRBs: {:.2e} ({:.2e})'.format(nns*c20*tlf(1.5, ms0), nbh*c2*tlf(10,ms)))
print('Num. of all NS (BH) XRBs: {:.2e} ({:.2e})'.format(nns*c20*t0, nbh*c2*t0))
print('Effective NS(BH)-XRB lifetimes: {:.2f} ({:.2f}) Gyr'.format(tlf(1.5,ms0)/1e9, tlf(10,ms)/1e9))
#beta = np.log(ms/ms0)/np.log(10/1.5)
#print(beta)
delta = np.log(sig/sig0)/np.log(10/1.5)
print('Velocity dispersion powerlaw index: {:.2f}'.format(delta))


#exit()

psi = np.array(retxt('nsc_cap.txt',3,1))
mstar, psif, psip = psi
#rho_mw = 8e4 #2.5e7
#sig_mw = sig
mnsc = 10**(0.48*np.log10(mstar/1e9)+6.51)
logreff = 0.54*(mnsc<2e6) + (0.34*np.log10(mnsc)-1.59)*(mnsc>=2e6)
g = 1.5
cd = 1
rnsc = 4./3.*10**logreff*(2**(1/(3-g))-1)
lrho = cd*3*mnsc/(4*np.pi*rnsc**3)/ms
lsig = (4.3e-3*mnsc/rnsc)**0.5
mr = 10
rMW = 3.2
lcgal = np.array([caprate(mr, rho, s, ms=ms, rs=ms**0.8) for rho,s in zip(lrho, lsig)]).T
gamma_ratio = lcgal[1]/caprate(mr, rho0*(mr/10)**alp, sig*(mr/10)**delta, ms=ms, rs=ms**0.8)[1] #rho/rho_mw*(sig/sig_mw)**(-4./3.)
m0 = 1e10
epsf = np.sum(psif*gamma_ratio)/np.sum(psif[mstar>m0])
epsp = np.sum(psip*gamma_ratio)/np.sum(psip[mstar>m0])

print('epsilon_MW = {:.2f} (F), {:.2f} (P)'.format(epsf, epsp))
print('Reference n_star and sigma_star: ', rho0*(mr/10)**alp, sig*(mr/10)**delta)
print('n_star_MW: ', 2.5e7*3/(4*np.pi*rMW**3)/ms)
#exit()

def virgo(m, alp=-1.33, ms=6.3e10, norm=5):
	return (m/ms)**(alp+1)*np.exp(-m/ms)*norm
	
lmg = np.geomspace(1e6, 1e12, 100)
logmg = np.log(lmg)
plt.figure()
plt.loglog(lmg, virgo(lmg))
plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$\Psi_{\rm Virgo}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.xlim(3e12, 1e4)
plt.ylim(0.1, 1e3)
plt.tight_layout()
plt.savefig('psi_mg_virgo.pdf')
plt.close()

mnsc = 10**(0.48*np.log10(lmg/1e9)+6.51)
logreff = 0.54*(mnsc<2e6) + (0.34*np.log10(mnsc)-1.59)*(mnsc>=2e6)
g = 1.5
cd = 1
rnsc = 4./3.*10**logreff*(2**(1/(3-g))-1)
lrho = cd*3*mnsc/(4*np.pi*rnsc**3)/ms
lsig = (4.3e-3*mnsc/rnsc)**0.5
lcgal = np.array([caprate(mr, rho, s, ms=ms, rs=ms**0.8) for rho,s in zip(lrho, lsig)]).T
gamma_ratio = lcgal[1]/caprate(mr, rho0*(mr/10)**alp, sig*(mr/10)**delta, ms=ms, rs=ms**0.8)[1] 
psi = virgo(lmg)
epsv = np.trapz(psi*gamma_ratio, logmg)/np.trapz(psi[lmg>1e10], logmg[lmg>1e10])
print('epsilon_Virgo = {:.2f}'.format(epsv))

#ms = 0.2 # companion star mass
nsm = 1.4
m1, m2 = nsm, 65 # Range of remnant mass (from the initial mass range 1-100 Msun based on HW02)
nm = 300
lm = np.geomspace(m1, m2, nm+1)

f3pc = 1e-2*lm**0.5
f3pc_mw = 0.02*(lm/150)**0.5

lmi = mf_mi(lm)
fB = 1e-3*lmi**1.35 # binary fraction as a function of remnant mass
fB[fB>1] = 1
fB *= 0.2
#print('fB: ', sumy(fB))

plt.figure()
plt.loglog(lm, fB)
plt.xlabel(r'$m_{\rm f}\ [\rm Msun]$', size=18)
plt.ylabel(r'$f_{\rm B}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.tight_layout()
plt.savefig('fB_m.pdf')
plt.close()

op = 0
# op=0: fully calibrated to Generozov et al. (2018) 
# op=1: for testing
if op>0:
	fnsc = 0 #f3pc
	mfg = 1.2e5
	alp=0
	norm = 1 #0.5/0.22
	y1, y2 = 1e5, 1e16
else:
	fnsc = 0 #0.15*f3pc
	mfg = 1.4e5
	norm = 1
	y1, y2 = 1e5, 1e16

#lnbh = 9*((mfg/150)**0.5*fnsc  +  (lm/150)**0.5*(1-fnsc))
mbar = 12.7
lnbh = 11*(lm/150)**0.5

#lms = (lm/1.5)**beta
#lms[lms<0.2] = 0.2
#lms[lms>1] = 1.0
lms = np.ones(len(lm))*0.2
lsig = sig0*(lm/1.5)**delta
lrho = rho0*(lm/10)**alp*norm # [rhos for x in range(nm+1)]

lcpr = np.array([caprate(m, rho, sig, ms=ms, rs=ms**0.8, rd=0, T2=0) for m,rho,sig,ms in zip(lm,lrho,lsig,lms)]).T
# lcpr[0]: disruption rate per object per yr
# lcpr[1]: capture rate per object per yr
# Here lcpr[1] corresponds to the Gamma' in the paper

tc1 = 1/lcpr[0]/lnbh
tc1_ = tc1/(1-fB)
tc2 = 1/lcpr[1]/lnbh
tc2_ = tc2/(1-fB)
tc = 1/(lcpr[0]+lcpr[1])/lnbh
tc_ = tc/(1-fB)
tc0 = 1e7*50/lnbh/(lm/150)**(4/3) # original rate from Madau & Rees 2001

plt.figure(figsize=(12,5))
plt.subplot(121)
#plt.loglog(lm, tc, '-', label='Total')
#plt.loglog(lm, tc_, 'm-')
plt.loglog(lm, tc2, 'r-', label='Capture')# (w/o binaires)')
#plt.loglog(lm, tc2_, 'r-')#, label='Capture (w binaires)')
plt.loglog(lm, tc1, 'k--', label='Disruption')
#plt.loglog(lm, tc1_, 'k--')
plt.loglog(lm, tc0, 'g:', label=r'MR01$\times 1-25$', lw=4.5, alpha=0.5)
plt.loglog(lm, tc0/25, 'g:', lw=4.5, alpha=0.5)
plt.plot([1.5, 10], [1/c20/nns0, 1/c2/nbh0], 'o', label='Capture (Pop I/II, GA18)', color='r')
plt.plot([1.5, 10], [1/c10/nns0, 1/c1/nbh0], '^', label='Disruption (Pop I/II, GA18)', color='k')
plt.xlabel(r'$m_{\rm f}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$\Gamma^{-1}\ [\rm yr]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(ncol=1, fontsize=14, loc=1)
plt.ylim(y1, y2)
#plt.tight_layout()
#if op>0:
#	plt.savefig('tcap_m_op.pdf')
#else:
#	plt.savefig('tcap_m.pdf')
#plt.close()

eta = 0.001
#t0 = 1e9
ltedd = tacc_m(lm, ms, eta)
lttde = ttde(lm, ms)
ltxrb = tlf(lm, ms)
print('XRB lifetime (Gyr): ', sumy(ltxrb/1e9))
print('TDE lifetime (yr): ', sumy(lttde))
print('Disruption to capture ratio: ',sumy(lcpr[0]/lcpr[1]))
#test = caprate(1.4, rho0*(1.4/10)**alp, sig0*(1.4/1.5)**delta, ms=ms, rs=ms**0.8, rd=0, T2=0)
#print(test[0]/test[1])
#plt.figure()
plt.subplot(122)
plt.loglog(lm, ltxrb/tc2, 'r-', label=r'XRB')# (w/o binaires)')
#plt.loglog(lm, ltxrb/tc2_, 'r-')#, label=r'XRB (w binaires)')
plt.loglog(lm, ltedd/tc2, 'b-.', label=r'Edd. Acc., $L/L_{\rm Edd}='+r'{}$'.format(eta))
#plt.loglog(lm, ltedd/tc2_, 'b-.')
plt.loglog(lm, lttde/tc1, 'k--', label=r'TDE')
#plt.loglog(lm, lttde/tc1_, 'k--')
#plt.loglog(lm, lt/tc0, 'm-.', label=r'TDE (MP01$\times 25$)')
plt.loglog([1.5, 10], [110, 110], 'o', color='r', label=r'Pop I/II XRB, GA18')
plt.xlabel(r'$m_{\rm f}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$N_{\rm XRB/TDE}\sim \Gamma t_{\rm vis}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(loc=6, bbox_to_anchor=(0., 0.175, 1.0, 0.4), fontsize=14, ncol=1)
#plt.title(r'$\eta={}$'.format(eta)+r', $\Delta m={}'.format(ms)+r'\ \rm M_{\odot}$')
plt.tight_layout()
if op>0:
	#plt.savefig('ncap_m_op.pdf')
	plt.savefig('tcap_ncap_m_op.pdf')
else:
	#plt.savefig('ncap_m.pdf')
	plt.savefig('tcap_ncap_m.pdf')
plt.close()


#print(sumy(1-fB))
plt.figure()
#plt.subplot(122)
plt.loglog(lm, ltxrb/tc2, 'r-', label=r'Pop III XRB (active)') # (w/o binaires)')
plt.loglog(lm, t0/tc2, 'r-.', label=r'Pop III XRB (all)') #, label=r'XRB (w binaires)')
#plt.loglog(lm, ltedd/tc2, 'b-.', label=r'Edd. Acc., $L/L_{\rm Edd}='+r'{}$'.format(eta))
#plt.loglog(lm, ltedd/tc2_, 'b-.')
plt.loglog(lm, lttde/tc1, 'k--', label=r'Pop III TDE')
#plt.loglog(lm, lttde/tc1_, 'k--')
#plt.loglog(lm, lt/tc0, 'm-.', label=r'TDE (MP01$\times 25$)')
plt.loglog([1.5, 10], [110, 110], 'o', color='r', label=r'Pop I/II XRB, GA18')
plt.loglog([1.5, 10], [c10*nns0*ttde(1.5,ms), c1*nbh0*ttde(10,ms)], '^', color='k', label=r'Pop I/II TDE, GA18')
plt.xlabel(r'$m_{\rm f}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$N_{\rm XRB/TDE}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
#plt.legend(loc=6, bbox_to_anchor=(0., 0.21, 1.0, 0.4), fontsize=14, ncol=2)
plt.legend(fontsize=14, loc=3)
#plt.title(r'$\eta={}$'.format(eta)+r', $\Delta m={}'.format(ms)+r'\ \rm M_{\odot}$')
plt.tight_layout()
if op>0:
	plt.savefig('ncap_m_op.pdf')
	#plt.savefig('tcap_ncap_m_op.pdf')
else:
	plt.savefig('ncap_m.pdf')
	#plt.savefig('tcap_ncap_m.pdf')
plt.close()
print('N_XRB (min, max, mean, median): ', sumy(ltxrb/tc2))
print('N_TDE (min, max, mean, median): ', sumy(lttde/tc1))

# L - m relation for XRBs:
llum = lum_xrb(lm, ms)
print('L/L_Edd: ', sumy(llum/(3.2e4*lm)))
llum_ = 3.2e4*lm*0.1

plt.figure()
plt.loglog(lm, llum, 'k-', label='GA18')
plt.loglog(lm, 3.2e4*lm/1e3, 'b-.', label=r'Edd. Acc., $L/L_{\rm Edd}='+r'{}$'.format(eta))
plt.xlabel(r'$m_{\rm f}$', size=18)
plt.ylabel(r'$L\ [\rm L_{\odot}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(loc=2, fontsize=18)
plt.tight_layout()
plt.savefig('L_m.pdf')
plt.close()

mall0, mall1 = 1e7, 2.27e+06

lnxrb0 = t0/tc2
NXRB0 = np.trapz(lnxrb0*lmi**-1.35, lmi)/np.trapz(lmi**-1.35,lmi)
print('NXRB_hat (all): {:.2e} - {:.2e}'.format(NXRB0, NXRB0*mall0/mall1))
sel = np.ones(len(lm), dtype=bool) #lmi>10
lnxrb = ltxrb/tc2
NXRB = np.trapz(lnxrb[sel]*lmi[sel]**-1.35, lmi[sel])/np.trapz(lmi[sel]**-1.35, lmi[sel])
print('NXRB_hat (active): {:.2e} - {:.2e}'.format(NXRB, NXRB*mall0/mall1))
#print('Dead ratio: ', sumy((t0-ltxrb)/t0), 1-NXRB/NXRB0)
lntde = lttde/tc1
NTDE = np.trapz(lntde*lmi**-1.35, lmi)/np.trapz(lmi**-1.35,lmi)
print('NTDE_hat: {:.2e} - {:.2e}'.format(NTDE, NTDE*mall0/mall1))
NPopIIIGC = np.trapz(lnbh*lmi**-1.35, lmi)/np.trapz(lmi**-1.35,lmi)
print('NPopIIIGC_hat: {:.2e} - {:.2e}'.format(NPopIIIGC, NPopIIIGC*mall0/mall1))

# Luminosity function:
# dN_dL = M_rem * dN_dmf * Gamma_ / dL_dmf * ltxrb
# M_rem = (M_bulge/100) * (lm/150)**0.5
# M_bulge = M_PopIII * (r_bulge/r_halo) ~ 3.4e4 - 1.5e5 
# dN_dmf = IMF * dmi_dmf
# Gamma_ = lcpr[1]
# dL_dmf = np.array([derivative(lum_xrb, m, dm) for m in lm])
# You can multiply this with (1-fB) to take into account binaries of compact objects

dm = 0.01
m1, m2 = 1, 100
x = 1.35
A = (2-x)/(m2**(2-x)-m1**(2-x))
mibar = (m2**(2-x)-m1**(2-x))/(2-x)*(1-x)/(m2**(1-x)-m1**(1-x))
mc = 10
mbar = (m2**(2-x)-mc**(2-x))/(2-x)*(1-x)/(m2**(1-x)-mc**(1-x))
imf = A*mf_mi(lm)**(-x)
mrem = np.trapz(imf*lm, lmi)/np.trapz(imf, lmi)
gapf = np.trapz(imf[lm>55], lmi[lm>55])/np.trapz(imf[lmi>mc], lmi[lmi>mc])
print('Mass-gap fraction: {:.2e}'.format(gapf))
#print(A)
ef, ep = epsf, epsp #41.07, 15.74
ev = epsv #5.4
ev1, ev2 = 6*(ep+ev-1), 6*(ep+(ev-1)*ep)
le = np.array([ep, ev1, ev2])
mrem0 = (0.02/1e2)*(lm/150)**0.5 * mall0
mrem1 = (0.02/1e2)*(lm/150)**0.5 * mall1
dl_dmf = np.array([derivative(lambda x: lum_xrb(x, ms), m, dm) for m in lm])
dl_dmf[dl_dmf<=0] = dl_dmf[dl_dmf>0][0]
dmi_dmf = np.array([derivative(mf_mi, m, dm) for m in lm])
dn_dl1 = mrem1 * imf * dmi_dmf * lcpr[1]*ltxrb / dl_dmf
dn_dl0 = mrem0 * imf * dmi_dmf * lcpr[1]*ltxrb / dl_dmf
dn_dlnl = mrem1 * imf * dmi_dmf * lcpr[1]*tacc_m(lm, ms=ms-0.1, eta=0.1) * lm

mfxrb = np.trapz(dn_dl1*lm,llum)/np.trapz(dn_dl1,llum)
print('Typical remnant mass in XRBs: {:.2f} Msun'.format(mfxrb))
print('Typical XRB luminosty: {:.2e} erg/s'.format(Lsun*lum_xrb(mfxrb, ms)))
print('Edd. ratio: ', sumy(llum/(3.2e4*lm)))
print('Mass limit to be seen by ATHENA: {:.2f}'.format(np.min(lm[llum*Lsun>3e35])))

gapf = np.trapz(dn_dl1[lm>55], lm[lm>55])/np.trapz(dn_dl1, lm)
print('Mass-gap fraction: {:.2f}'.format(gapf))

selns = (lm>=nsm)*(lm<3)
selbh = lm>=3
nsxrb = np.trapz(dn_dl1[selns], llum[selns])
bhxrb = np.trapz(dn_dl1[selbh], llum[selbh])
nxrb_ = np.trapz(dn_dlnl/lm, lm)

print('eps_MW, Virgo (low), Virgo (up):', le)
print('NS-XRB (LW): ', nsxrb*le)
print('NS-XRB (no LW): ', nsxrb*le*mall0/mall1)
print('BH-XRB (LW): ', bhxrb*le)
print('BH-XRB (no LW): ', bhxrb*le*mall0/mall1)
print('NS to BH ratio: {:.2e}'.format(nsxrb/bhxrb))
print('L/L_Edd=0.1 (LW): {:.2e}'.format(nxrb_*le[-1]))

l1, l2 = lum_xrb(2, ms)*Lsun, lum_xrb(3, ms)*Lsun
y1, y2 = 1e-3, 1e4
x1, x2 = 1e35, 3e36
plt.figure()
#plt.loglog(llum*Lsun, dn_dl1*llum*ef, 'k-')
plt.loglog(llum*Lsun, dn_dl1*llum*ep, 'k-', label='MW + LW')
#plt.loglog(llum*Lsun, dn_dl0*llum*ef, 'b--')
plt.loglog(llum*Lsun, dn_dl0*llum*ep, 'b--', label='MW no LW')
#plt.loglog(llum*Lsun, dn_dl1*llum*ev1, 'r-.')
#plt.loglog(llum*Lsun, dn_dl1*llum*ev2, 'r-.', label='Virgo + LW')
plt.fill_between(llum*Lsun, dn_dl1*llum*ev1, dn_dl1*llum*ev2, fc='r', lw=0, alpha=0.4, label='Virgo + LW')
plt.loglog(llum*Lsun, dn_dl0*llum*ev1, 'g:')
plt.loglog(llum*Lsun, dn_dl0*llum*ev2, 'g:', label='Virgo no LW')
#plt.loglog(llum_*Lsun, dn_dlnl*ev2, 'r-.', label=r'Virgo + LW, $L/L_{\rm Edd}=0.1$')
plt.fill_between([l1, l2], [y1]*2, [y2]*2, fc='gray', alpha=0.5)
plt.xlabel(r'$L\ [\rm erg\ s^{-1}]$', size=18)
plt.ylabel(r'$L\frac{d N_{\rm XRB}}{dL}$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.legend(loc=4, ncol=1, fontsize=14)
plt.ylim(y1, y2)
plt.xlim(x1, x2)
plt.tight_layout()
plt.savefig('dn_dl.pdf')
plt.close()

#exit()

lr, ln = np.array(retxt('bhb_root.dat',2,2))
lsel = [lr<=11*(m/45)**0.5 for m in lm]
ly = np.zeros(nm)
for i in range(nm):
	if np.sum(lsel)==0:
		ly[i] = 0
	else:
		ly[i] = np.sum(ln[lsel[i]])
print('NPOPIII_GC from merger trees: ', 2*sumy(ly)[1]/500/0.7 * 56.47/mbar)
print('Average compact object (progenitor) mass: {:.2f} ({:.2f})'.format(mrem, mbar))
print('Average stellar mass: {:.2f}'.format(mibar))

lms = np.linspace(0.2, 1.0, 300)
r1 = np.array([caprate(1.5, ms=ms, rs=ms**0.8) for ms in lms]).T
r2 = np.array([caprate(10, ms=ms, rs=ms**0.8) for ms in lms]).T
y1, y2 = 1e-12, 3e-9
plt.figure()
plt.plot(lms, r1[0]+r1[1], 'k-', label=r'NS ($m_{\rm f}=1.5\ \rm M_{\odot}$)')
plt.plot(lms, r1[0], 'k--', label=r'NS, disruption')
plt.plot(lms, r1[1], 'k-.', label=r'NS, capture')
plt.plot(lms, r2[0]+r2[1], 'r-', label=r'BH ($m_{\rm f}=10\ \rm M_{\odot}$)')
plt.plot(lms, r2[0], 'r--', label=r'BH, disruption')
plt.plot(lms, r2[1], 'r-.', label=r'BH, capture')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$', size=18)
plt.ylabel(r'$\gamma\ [\rm yr^{-1}]$', size=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.yscale('log')
plt.legend(ncol=2, fontsize=14)
plt.title(r'$n_{\star}=10^{6}\ \rm pc^{-3}$, $\sigma=110\ \rm km\ s^{-1}$', size=18)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig('gamma_ms.pdf')
plt.close()

