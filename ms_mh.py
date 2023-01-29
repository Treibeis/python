import matplotlib.pyplot as plt
from cosmology import *
from txt import *
import matplotlib
#import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import sys
import os
plt.style.use('test2')
plt.style.use('tableau-colorblind10')
from scipy.optimize import curve_fit
#from halomassfunc import *
#from xraylw import *
from colossus.cosmology import cosmology
#from colossus.lss import mass_function

Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5

lls = ['-', '--', '-.', ':', (0,(10,5)), (0,(1,1,3)), (0,(5,1))]
llc = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

cosname = 'planck15'
cosmo = cosmology.setCosmology(cosname)
h = cosmo.H0/100
Om = cosmo.Om0
Ob = cosmo.Ob0
Or = cosmo.Or0

Tmin = 30
fr = 1/3**0.5
tmax = 1e6
#kmdot = 1 
kmdot = 0.7 #(1.09-1.0/3.0)
print('kmdot={:.2f}'.format(kmdot))
mode = 1
steep = 0

def M_t(t, geff=1.09):
	return 400*(t/1e5)**(4-3*geff)

def T_SIS(mh, z, Tmin=Tmin, delta=200, Om=Om, Ob=Ob, h=h):
	r = RV(mh, z, delta, Om, h)/PC
	y0 = 300 * (mh/950)/r*Ob/Om
	#return 300
	return y0*(y0>=Tmin) + Tmin*(y0<Tmin)

def ms_mh(mh, z, beta, fr=fr, vs=40, Tmin=Tmin, geff=1.09, delta=200, Om=Om, Ob=Ob, h=h, mode=mode, ymax=1e6, tmax=tmax, k=kmdot):
	if mode>0:
		norm = mdot_vir(mh, z, fr, Tmin, delta, Om, Ob, h)/mdot_vir(1e6, 20, fr, Tmin, delta, Om, Ob, h)
	else:
		norm = 1
	norm = norm**k
	T1 = T_SIS(mh, z, Tmin, delta, Om, Ob, h)
	A = 0.28*(T1/300)**2/(vs/40)/beta/norm
	tb = A**(1/(5-3*geff))*1e5
	tb = tb*(tb<tmax) + tmax*(tb>=tmax)
	y = M_t(tb, geff)*norm
	#y *= (mh/1e6*((1+z)/21)**1.5)**(1/(5-3*geff))
	return y*(y<ymax) + ymax*(y>=ymax)

def mdot_vir(mh, z, fr=fr, Tmin=Tmin, delta=200, Om=Om, Ob=Ob, h=h):
	rho = delta*rhom(1/(1+z), Om, h)/3 * Ob/Om
	r = RV(mh, z, delta, Om, h)
	T = Tvir(mh, z, delta, 0, Om, h)
	fac = 1 #(T - T_SIS(mh, z, Tmin, delta, Om, Ob, h))/T
	v = Vcir(mh, z, delta, Om, h) * fr * fac
	mdot = 4*np.pi*rho*r**2*v
	if steep>0:
		mdot *= (mh/1e6)*((1+z)/21)**1.5
	return mdot/Msun*YR 

def ms_mh_h14(mh, z, fr=fr, Tmin=Tmin, delta=200, Om=Om, Ob=Ob, h=h, ymax=1e6):
	y = 100* mdot_vir(mh, z, fr, Tmin, delta, Om, Ob, h)/1.2e-3
	return y*(y<ymax) + ymax*(y>=ymax)

def mth_anna(j21, vbc=0.8):
	logM = 5.562*(1+0.279*j21**0.5)
	s = 0.614*(1-0.56*j21**0.5)
	logM += s*vbc
	return 10**logM

#"""
fmatom = 0
SFE = 1e-3
fac = 10
mth = mth_anna(0, 0)
x1, x2 = 1e5, 1e8
lmh = np.geomspace(x1, x2, 100)
lz = [30, 25, 20, 15, 10]
beta = 0.8
lmscr = [723, 796]
dms = []
for i in range(len(lz)):
	z = lz[i]
	#lT = T_SIS(lmh, z, Tmin)
	#print(lT[0], lT[-1])
	dms.append(ms_mh(lmh, z, beta)) #M_t(tacc_func(lT, beta=beta))
	
y1, y2 = 100, 1e5
plt.figure()
for i in range(len(lz)):
	z = lz[i]
	lab = r'$z={}$'.format(z)
	lms = dms[i]
	plt.loglog(lmh, lms, ls=lls[i], color=llc[i], label=lab)
	lmdot = mdot_vir(lmh, z)
	sel = (lmdot>1e-4)*(lmdot<1e-2)
	lms = ms_mh_h14(lmh, z)
	plt.plot(lmh[sel], lms[sel], ls=lls[i], color=llc[i], lw=4.5, alpha=0.5)
	if fmatom>0:
		plt.plot([Mup(z)]*2, [y1, y2], ls=lls[i], color=llc[i], lw=0.5)
plt.fill_between([x1, x2], [lmscr[0]]*2, [lmscr[1]]*2, fc='gray', alpha=0.5, label='EDGES (SA19)')
plt.plot([mth]*2, [y1, y2], 'k-', lw=0.5)
plt.plot(lmh, SFE*lmh*Ob/Om, ls=(0,(5,1)), color='k', label=r'$\rm SFE={}-{}$'.format(SFE, SFE*fac))
plt.plot(lmh, fac*SFE*lmh*Ob/Om, ls=(0,(5,1)), color='k')
#plt.plot([x1, x2], [mscr]*2, 'k-', lw=0.5)
plt.legend(ncol=1)
plt.xlabel(r'$M_{\rm h}\ [\rm M_{\odot}]$')
plt.ylabel(r'$M_{\star}\ [\rm M_{\odot}]$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.title(r'$Q_{\star}='+'{:.2f}'.format(1.5*beta)+r'\times 10^{50}{\ \rm s^{-1}}\ (M_{\star}/100\ \rm M_{\odot})$')
plt.tight_layout()
plt.savefig('ms_mh_z_beta'+str(beta)+'.pdf')
plt.close()

plt.figure()
#for i in range(len(lz)):
i = 0
z = lz[i]
lms = dms[i]
lmdot = mdot_vir(lmh, z)
plt.loglog(lmdot, lms, ls=lls[i], color=llc[i], label=r'This work ($\beta={}$)'.format(beta))
i = 1
lms = ms_mh_h14(lmh, z)
sel = (lmdot>1e-4)*(lmdot<1e-2)
plt.plot(lmdot[sel], lms[sel], ls=lls[i], color=llc[i], lw=4.5, alpha=0.5, label='HS14')
plt.plot([mdot_vir(mth, z)]*2, [y1, y2], 'k-', lw=0.5)
#plt.fill_between([x1, x2], [lmscr[0]]*2, [lmscr[1]]*2, fc='gray', alpha=0.5, label='EDGES (SA19)')
#plt.plot([x1, x2], [mscr]*2, 'k-', lw=0.5)
plt.legend()
plt.xlabel(r'$\dot{M}_{\rm vir}\ [\rm M_{\odot}\ yr^{-1}]$')
plt.ylabel(r'$M_{\star}\ [\rm M_{\odot}]$')
#plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.title(r'$Q_{\star}='+'{:.2f}'.format(1.5*beta)+r'\times 10^{50}{\ \rm s^{-1}}\ (M_{\star}/100\ \rm M_{\odot})$')
plt.tight_layout()
plt.savefig('ms_mdot_z_beta'+str(beta)+'.pdf')
plt.close()
#"""
