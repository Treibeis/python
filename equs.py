from cosmology import *
from scipy.interpolate import interp1d
from txt import *
import numpy.random as nrdm
import matplotlib
plt.style.use('test2')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from scipy.integrate import quad, solve_ivp
import os
lls = ['-', '--', '-.', ':', (0,(10,5)), (0,(5,1))]
llc = ['m', 'b', 'c', 'g', 'orange', 'r']

fnsc0 = 1 #7.71396309e-02
fbn0 = 1 #0.783622146
fbn = fbn0
#fbn = 2.48286963
#fbn = 1.85265231
fnsc = fnsc0
#fnsc = 0.133628279
#fnsc = 0.111784279
labe = 'eta1.0E-03'
fs = '_IMFmin1.0E+00IMFmax1.7E+02'+labe+'slope0.0E+00.dat'
ntree = 500
drep = 'Merger_Tree/dummy500/' 
#repref = 'bhb_stat/'
repref = 'Merger_Tree/dummy51/'
drepi = 'Merger_Tree/dummy500/'
repo = 'bhb_stat/eif0_0/'
eif = 0
Vcom = 30.3
#mrem_file = 'mass_star_remnant.dat'
#mrem_file = 'mrem_mi_comb.dat'
mrem_file = 'mrem_mi.dat'
test = 0
seed = 1314
bhb_min, bhb_max = 3, 200
nb = 8
fgw0 = 1
fub = 0

op = 1
mode = 1

cr = 'comp/'

drep1 = 'Merger_Tree/dummy51/'
drep2 = 'Merger_Tree/dummy52/'

if not os.path.exists(repo):
	os.makedirs(repo)

def PoD(k, mu):
	return np.exp(-mu)*mu**k/np.math.factorial(k)
	
def ModPoD0(k, mu):
	return np.exp(-(mu-k)**2/4)*PoD(k, mu)
	
kmax = 100
mu2 = 10

lmu, lck = retxt('ck_mu.txt',2)
ck_mu = interp1d(lmu, lck)

def ModPoD(k, mu, mu2=mu2, kmax=kmax):
	if mu<=mu2:
		ck = ck_mu(mu)
	else:
		ck = 1/np.sum([ModPoD0(j, mu) for j in range(kmax+1)])
	return ck*ModPoD0(k, mu)

def gdf(e, g, a1=2.63, a2=2.26, a3=0.9):
	return (2-g)*(a1*(1/(2-g)**a2+a3)*(1-e)+e)
	
def tdf_arca(r, m, M, R, e, g, alpha=-0.67, beta=1.76, a1=2.63, a2=2.26, a3=0.9):
	tau = 0.3*(1e11/M)**0.5*(R/1e3)**1.5*1e6
	return tau*gdf(e, g, a1, a2, a3)*(m/M)**alpha*(r/R)**beta
	
def Rg_Mg(M, g=1.5):
	return 2.37e3*(2**(1/(3-g))-1)*(M/1e11)**0.14
	
def R_Mg(M, g=1.5):
	return 31.62*Rg_Mg(M, g)*(M/1e11)**(1./6)
	
def tdf_old(r, m, M, R, lnL=0):
	sigma = (GRA*M*Msun/(1.0*R*PC))**0.5/1e5
	if lnL==0:
		lnL = np.log(M/m*r/R/0.8)
	return 1.9e10/lnL*(r/5e3)**2*(sigma/200)*(1e8/m)
	
def tdf(r, m, M, R, e, g, alpha=-0.67, beta=1.76, a1=2.63, a2=2.26, a3=0.9, lnL=0):
	t1 = tdf_old(r, m, M, R, lnL)
	if g<2:
		t2 = tdf_arca(r, m, M, R, e, g, alpha, beta, a1, a2, a3)
		return min(t1, t2)
	else:
		return t1
	
smhm0 = [-1.435, 1.831, 1.368, -0.217, 12.035, 4.556, 4.417, -0.731, 1.963, -2.316] \
		+ [-1.732, 0.178, 0.482, -0.841, -0.471, 0.411, -1.034, -3.100, -1.055]
smhm1 = [-1.43, 1.796, 1.36, -0.216, 12.04, 4.675, 4.513, -0.744, 1.973, -2.353] \
		+ [-1.783, 0.186, 0.473, -0.884, -0.486, 0.407, -1.088, -3.241, -1.079]
	
Ob, Om = 0.048, 0.3089
fb = Ob/Om
		
def mjion(z, delta=125, Tb=2e4):
	return 6.7e8*((1+z)**3*delta/5**3/125)**-0.5*(Tb/2e4)**1.5
		
def ms_mh(mh, z, e0, ea, elna, ez, m0, ma, mlna, mz, a0, aa, alna, az, b0, ba, bz, d0, g0, ga, gz, eta0=0.003, fb=fb, zr=6):
	a = 1/(1+z)
	logm1 = m0+ma*(a-1)-mlna*np.log(a)+mz*z
	x = np.log10(mh)-logm1
	eps = e0+ea*(a-1)-elna*np.log(a)+ez*z
	alp = a0+aa*(a-1)-alna*np.log(a)+az*z
	beta = b0+ba*(a-1)+bz*z
	gamma = 10**(g0+ga*(a-1)+gz*z)
	delta = d0
	logms = eps-np.log10(10**(-alp*x)+10**(-beta*x))+gamma*np.exp(-0.5*(x/delta)**2)
	#print(x, logm1, logms)
	#print(eps, alp, beta, gamma)
	ms = 10**(logms+logm1)
	ms0 = 0 + mh*eta0*fb*(z>zr)
	ms = ms*(ms>ms0) + ms0*(ms<=ms0)
	if z<=zr:
		ms *= (mh>mjion(z))
	return ms

mcut = 2.133e6

def focc_nsc(m, f1=0.15, f2=0.72, f3=0.35, m1=6.6, m2=8.2, m3=9.5, m4=11.5):
	logm = np.log10(m)
	s1 = (f2-f1)/(m2-m1)
	s2 = (f3-f2)/(m4-m3)
	y = f1 * (logm<m1)
	y += (f1+s1*(logm-m1)) * (logm>=m1)*(logm<m2)
	y += f2 * (logm>=m2)*(logm<m3)
	y += (f2+s2*(logm-m3)) * (logm>=m3)
	return y

def sfrtf(z, a, b, c, d):
	#t = 1/(z+1)
	#return a*(t**b*np.exp(-t/c)+d*np.exp(d*(t-1)/c)) 
	return a*(1+z)**b/(1+((1+z)/c)**d)
	
def sfrbf(z):
	return 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)
	
m = 20
g = 0
e = 0
M1, M2 = 1e5, 1e11
lM = np.geomspace(M1, M2, 100)
lR = Rg_Mg(lM, g)
lr = [1, 3, 10, 30]#, 1e3]
lc = ['r', 'g', 'b', 'k']
y1, y2 = 1, 1e8
#y1, y2 = 1, 1e6
plt.figure()
for i in range(len(lr)):
	lt1 = tdf_old(lr[i], m, lM, lR)/1e6
	lt2 = tdf_arca(lr[i], m, lM, lR, e, g)/1e6
	if 1:
		plt.loglog(lM, lt1, color=lc[i], label=r'$r={}\ \rm pc$ (BT11)'.format(lr[i]))
		plt.loglog(lM, lt2, '--', color=lc[i], label=r'$r={}\ \rm pc$ (AS16)'.format(lr[i]))
		#plt.loglog(lM, lt2, lls[i], color=lc[i], label=r'$r={}\ \rm pc$'.format(lr[i]))
	else:
		plt.loglog(lM, lt1, color=lc[i], label=r'$r={}\ \rm pc$ (BT11)'.format(lr[i]))
		plt.loglog(lM, lt2, '--', color=lc[i], label='AS16')
plt.legend(loc=2, ncol=2)
plt.title(r'${}={}\ {}, \gamma_{}={}$, $e_{}={}$'.format(r'm_{\bullet}', m, r'\rm M_{\odot}', r'{\rm g}', g, r'{\rm if}', e))
plt.xlim(M1, M2)
plt.ylim(y1, y2)
plt.xlabel(r'$M_{\star}\ \rm [M_{\odot}]$')
plt.ylabel(r'$\tau_{\rm DF}\ \rm [Myr]$')
plt.tight_layout()
plt.savefig('tdf_Mg.png')
plt.close()

lx = np.linspace(0,2,100)
print('Minimal tdf at 300 pc: {:.2f} Gyr'.format(np.min([tdf(300,20,1e5,Rg_Mg(1e5,x),0.9,x)/1e9 for x in lx])/1.67))

"""
lmu = np.linspace(0, mu2, 1000)
lck = 1./np.array([np.sum([ModPoD0(k, mu) for k in range(kmax+1)]) for mu in lmu])
totxt('ck_mu.txt', [lmu, lck])
plt.figure()
plt.plot(lmu, lck, label=r'$k_{\max}='+str(kmax)+'$')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$c_{\mu}$')
plt.xlim(0, 10)
plt.ylim(1, 2.5)
plt.legend()
plt.tight_layout()
plt.savefig('ck_mu.pdf')
plt.close()

le0 = [0, 0.5, 0.9, 0.99]
lg = np.linspace(0, 1.8, 100)
plt.figure()
for i in range(len(le0)):
	e0 = le0[i]
	lgdf = gdf(e0, lg)
	plt.plot(lg, lgdf, label=r'$e={:.2f}$'.format(e0), ls=lls[i])
plt.legend()
plt.yscale('log')
plt.xlim(0, 2)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$g(e,\gamma)$')
plt.tight_layout()
plt.savefig('gdf_gamma.pdf')
plt.close()

lg0 = [0, 1, 1.5, 1.8]
le = np.linspace(0, 1, 100)
plt.figure()
for i in range(len(lg0)):
	g0 = lg0[i]
	lgdf = gdf(le, g0)
	plt.plot(le, lgdf, label=r'$\gamma={:.1f}$'.format(g0), ls=lls[i])
plt.legend()
plt.yscale('log')
plt.xlim(0, 1)
plt.ylim(0.1, 30)
plt.xlabel(r'$e$')
plt.ylabel(r'$g(e,\gamma)$')
plt.tight_layout()
plt.savefig('gdf_e.pdf')
plt.close()

drep1 = 'Merger_Tree/test1/' 
drep2 = 'Merger_Tree/test2/' 
ntree = 100
bhbm1 = np.array(retxt(drep1+'bhb_mass_dis.dat',2,2))
print('N_BHB per tree: {:.0f}'.format(np.sum(bhbm1[1])/ntree))
bhbm2 = np.array(retxt(drep2+'bhb_mass_dis.dat',2,2))
print('N_BHB per tree: {:.0f}'.format(np.sum(bhbm2[1])/ntree))
"""

if test>0:
	ntree = 10
	drepi = 'Merger_Tree/dummy/' 
	repref = 'Merger_Tree/dummy/'
	drep = 'Merger_Tree/dummy/' 
	repo = 'bhb_stat/dummy/'

ref0 = np.array(retxt(cr+'SF16.txt', 3, 0, 0))
datarep = '/home/friede/Documents/SimulationCodes/FDbox_Lseed/'
repi = datarep
sfr0III = np.array(retxt_nor(repi+'popIII_sfr.txt', 4, 0, 0))
sfr0II = np.array(retxt_nor(repi+'popII_sfr.txt', 4, 0, 0))

fsfr = drepi+'z_SFRII'+fs
fsfr3 = drepi+'z_SFRIII'+fs
fflw = drepi+'z_FLW'+fs
sfr2 = np.array(retxt(fsfr, 4, 2))
sfr3 = np.array(retxt(fsfr3, 7, 3))
x1, x2 = 0, 40
y1, y2 = 1e-6, 10
lz = np.linspace(x1, x2, 1000)
para = [765.7, -5.92, 12.83, -8.55]
madau17 = [0.01, 2.6, 3.2, 6.2]
madau14 = [0.015, 2.7, 2.9, 5.6]
lsfrd_ = sfrtf(lz, *madau14)
lsfrd = sfrtf(lz, *para)
sfrt = sfr2[1]+sfr3[5]
sfrts = sfr2[2]+sfr3[6]
low = sfrt - 3*sfrts
low[low<0.1*sfrt] = 0.1*sfrt[low<0.1*sfrt]
plt.figure()
plt.plot(sfr2[0], sfrt, '-', label=r'Total ($\eta_{\star,0}=0.003$)')
plt.fill_between(sfr2[0], sfrt+sfrts*3, low, fc='m', alpha=0.2)
plt.plot(sfr3[0], sfr3[5], '--', label=r'Pop III ($\eta_{\star,\rm III}=0.001$)')
plt.fill_between(sfr3[0], sfr3[5]+3*sfr3[6], sfr3[5]-3*sfr3[6], fc='b', alpha=0.2)
plt.plot(1/sfr0II[0]-1, sfr0II[3]+sfr0III[3], ls=(0, (5,1)), label='Total (Liu+2020)')
plt.plot(lz[lz>4], lsfrd[lz>4], 'g:', label='Pop III (Liu+2020)')
plt.plot(sfr2[0], sfr2[3], 'k-.', label='Campisi+2011')# (obs. & sim.)')
plt.plot(lz[lz<10], lsfrd_[lz<10], color='r', ls=(0,(10,5)), label='Madau+2014')
plt.fill_between(lz[lz<10], lsfrd_[lz<10]*10**0.2, lsfrd_[lz<10]/10**0.2, facecolor='r', alpha=0.3)#, label='Madau+2014')
#plt.plot(ref0[0], 10**ref0[1], 'o', label='Finkelstein+2016: observed', color='k')
plt.plot(ref0[0], 10**ref0[2], '^', label='Finkelstein+2016', color='k')
zcol = 4.6
#plt.fill_between([x1, zcol],[y1,y1],[y2,y2],fc='gray',alpha=0.3)
plt.plot([zcol]*2,[y1,y2],'k-',lw=0.5)
plt.yscale('log')
plt.xlabel(r'$z$')
plt.ylabel(r'$\rm SFRD\ [M_{\odot}\ yr^{-1}\ Mpc^{-3}]$')
plt.legend(ncol=1)#,fontsize=14)
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig('sfrd_z.pdf')
plt.close()

lt = np.array([TZ(z) for z in lz])/YR
lt_ = np.array([TZ(z) for z in sfr3[0]])/YR
print('Overall Pop III fraction: {:.4e}-{:.4e}'.format(abs(np.trapz(lsfrd, lt))/abs(np.trapz(lsfrd_, lt)),abs(np.trapz(sfr3[5], lt_))/abs(np.trapz(lsfrd_, lt))))

flwref = np.array(retxt('fdbk/FLW_z.txt',2))
x1, x2 = 0, 25
flw = np.array(retxt(fflw,5,2))
low = flw[1]-flw[2]*3
low[low<flw[1]*0.1] = flw[1][low<flw[1]*0.1]*0.1
y1, y2 = 1e-3, 1e2
plt.figure()
plt.plot(flw[0], flw[1], label='Total')
plt.fill_between(flw[0], flw[1]+flw[2]*3, low, fc='m', alpha=0.2)
plt.plot(flw[0], flw[3], '--', label='Pop III')
plt.fill_between(flw[0], flw[3]+flw[4]*3, flw[3]-flw[4]*3, fc='b', alpha=0.2)
plt.plot([x1,x2], [4*np.pi]*2, 'k:', label=r'$4\pi$')
plt.plot(*flwref, '-.', label='Liu+2020')
plt.yscale('log')
plt.xlabel(r'$z$')
plt.ylabel(r'$F_{\rm LW}\ [10^{-21}\ \rm erg\ s^{-1}\ cm^{-2}\ Hz^{-1}]$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.legend()
plt.tight_layout()
plt.savefig('FLW_z.pdf')
plt.close()

zreion = 5.5
ffion1 = np.array(retxt(drepi+'z_QIII'+fs,3,2))
ffion2 = np.array(retxt(drepi+'z_QII'+fs,3,2))
x1, x2 = 2.5, 20
y1, y2 = 0, 1.1
plt.figure()
plt.plot(ffion1[0],ffion1[1], label='Total')
plt.fill_between(ffion1[0],ffion1[1]-ffion1[2]*3,ffion1[1]+ffion1[2]*3,fc='m',alpha=0.2)
plt.plot(ffion2[0],ffion2[1],'-.',label='Pop I/II only')
plt.fill_between(ffion2[0],ffion2[1]-ffion2[2]*3,ffion2[1]+ffion2[2]*3,fc='b',alpha=0.2)
plt.plot([zreion]*2,[y1,y2],'k-',lw=0.5)
plt.xlabel(r'$z$')
plt.ylabel(r'$Q_{\rm II}$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.legend()
plt.tight_layout()
plt.savefig('fHII_z.pdf')
plt.close()

tau_local=0 #0.01045
tau = 0.0544
dtau = 0.0073

ltau1 = np.array(retxt(drepi+'z_tauIII'+fs,3,2))
ltau2 = np.array(retxt(drepi+'z_tauII'+fs,3,2))
ltaud = np.array(retxt(drep1+'z_tauIII'+fs,3,2))
ltauu = np.array(retxt(drep2+'z_tauIII'+fs,3,2))
x1, x2 = 2.5, 20
y1, y2 = tau_local, 0.09
nsig = 1
plt.figure()
plt.plot(ltau1[0],ltau1[1]+tau_local, label=r'Total, $f_{\rm esc,III}=0.125$')
plt.plot(ltaud[0],ltaud[1]+tau_local, 'm:', label=r'Total, $f_{\rm esc,III}\sim 0.1-0.3$')
plt.plot(ltauu[0],ltauu[1]+tau_local, 'm:')
#plt.fill_between(ltau1[0],ltau1[1]-ltau1[2]*nsig+tau_local,ltau1[1]+ltau1[2]*nsig+tau_local,fc='m',alpha=0.2)
plt.plot(ltau2[0],ltau2[1]+tau_local,'-.',label=r'Pop I/II only, $f_{\rm esc,I/II}=0.1$')
#plt.fill_between(ltau2[0],ltau2[1]-ltau2[2]*nsig+tau_local,ltau2[1]+ltau2[2]*nsig+tau_local,fc='b',alpha=0.2)
plt.plot([x1,x2],[tau]*2,'r--',label='Planck 2018')
plt.fill_between([x1,x2],[tau-dtau*nsig]*2,[tau+dtau*nsig]*2,fc='r',alpha=0.2)
plt.plot([zreion]*2,[y1,y2],'k-',lw=0.5)
plt.xlabel(r'$z$')
plt.ylabel(r'$\tau$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.legend()
plt.tight_layout()
plt.savefig('tau_z.pdf')
plt.close()

#"""
eta0 = 0.003
m1, m2 = 1e8, 1e15
lmh = np.geomspace(m1, m2, 100)
lz = [0, 2, 4, 6, 8, 10]
lmm = [5e14, 1e14, 2e13, 5e12, 7e11, 2e11] #[m2]*6
lmi = [1e10]+[1e8]*5 
#lmi = [1e10, 1e11, 7e10, 4e10, 4e10, 1e8]
y1, y2 = 1e5, 1e12
leta = [1e-3, 0.01, 0.1]
plt.figure()
for i in range(len(lz)):
	z = lz[i]
	sel = (lmh<lmm[i])*(lmh>lmi[i])
	lms = ms_mh(lmh[sel], z, *smhm1, eta0=eta0)#, leta0[i])
	plt.loglog(lmh[sel], lms, label=r'$z={}$'.format(z), ls=lls[i], color=llc[i])
for eta in leta:
	plt.plot(lmh, lmh*fb*eta, '--', lw=3, color='gray', alpha=0.5)
plt.plot(lmh, lmh*fb, '--', lw=3, color='gray', alpha=0.5, label=r'$\eta_{\star}\sim 10^{-3}-1$')
plt.xlabel(r'$M_{\rm h}\ [\rm M_{\odot}]$')
plt.ylabel(r'$M_{\star}\ [\rm M_{\odot}]$')
plt.text(m1*2, y2/3, r'$\eta_{\star,0}='+str(eta0)+'$')
plt.legend()
plt.xlim(m1, m2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig('ms_mh.pdf')
plt.close()


lm = np.geomspace(1e6, 1e12, 100)
l = np.array(retxt(cr+'nsc_occ.txt', 7))
plt.figure()
lf = focc_nsc(lm)
plt.plot(10**l[0], l[1], 'r--', label='Early-type')
plt.fill_between(10**l[0], l[2], l[3], fc='r', alpha=0.2)
sel = l[0]<11
plt.plot(10**l[0][sel], l[4][sel], 'b-.', label='Late-type')
plt.fill_between(10**l[0][sel], l[5][sel], l[6][sel], fc='b', alpha=0.2)
#plt.plot(10**l[0][sel], (l[1][sel]+l[4][sel])*0.5, 'k:', label='Mean')
#plt.plot(lm, lf, '-', label='Approx.', lw=4.5, alpha=0.7, color='gray')
plt.xscale('log')
plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$f_{\rm NSC}$')
plt.legend(loc=2,ncol=2)
plt.xlim(1e6, 1e12)
plt.ylim(0, 1.)
plt.tight_layout()
plt.savefig('focc_nc.pdf')
plt.close()
#"""
	
#l = abhb_t(1e9, 100, 100, 100, 100, 0.5, 2, 1e7, 2, 0.5, 1)

bhbcata = np.array(retxt(drep+'bhb_nsc_cata'+fs,4,2))
bhbroot = np.array(retxt(drep+'bhb_root_cata'+fs,4,2))
rootz = ZT(np.log10((TZ(0)/YR-bhbroot[1])/1e9))
ntot1 = len(bhbcata[0])+len(bhbroot[0])

bhmin = 10
bhmax = 170
drep0 = 'Merger_Tree/Data/'
frem = retxt(drep0+mrem_file,2)
mrem_ms = interp1d(*frem)
fbcata = np.array(retxt(drep0+'bcata_all.txt',5))
sel = (fbcata[0]>=bhmin)*(fbcata[1]>=bhmin)*(fbcata[0]<=bhmax)*(fbcata[1]<=bhmax)
m1 = mrem_ms(fbcata[0][sel])
m2 = mrem_ms(fbcata[1][sel])
sel_ = (m1>0)*(m2>0)
lmbhb = m1+m2
bhbnbody = [lmbhb[sel_],m1[sel_],m2[sel_],fbcata[3][sel][sel_],fbcata[4][sel][sel_]]
totxt(repo+'bhbcata_nbody.txt',bhbnbody)
fco = np.sum(bhbnbody[0])/(np.sum(fbcata[0])+np.sum(fbcata[1]))
fbinary = 0.69
print('Mass fraction of compact object binaries: {:.2e}'.format(fco))
#print(np.sum(m1[sel_]==m2[sel_])/np.sum(sel_))

"""
x1, x2 = 0, 20
lz = np.linspace(x1, x2, 41)
plt.figure()
plt.hist(np.hstack([bhbcata[1],rootz]), lz, alpha=0.5)
plt.xlabel(r'$z_{\rm if}$')
plt.ylabel(r'$dN/dz_{\rm if}\ [\rm a.u.]$')
plt.xlim(x1, x2)
plt.tight_layout()
plt.savefig(repo+'bhbzif_dis.pdf')
plt.close()

y1, y2 = 10*(ntree/100), 1e5*(ntree/100)
x1, x2 = 1e5, 1e8
lmnsc = np.geomspace(x1, x2, 31)
plt.figure()
plt.hist(np.hstack([bhbcata[2], bhbroot[2]]), lmnsc, alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm NSC}\ [\rm M_{\odot}]$')
plt.ylabel(r'$dN/d\log M_{\rm NSC}\ [\rm a.u.]$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig(repo+'bhbmnsc_dis.pdf')
plt.close()

y1, y2 = 10*(ntree/100), 3e4*(ntree/100)
x1, x2 = 1e5, 1e11
lmnsc = np.geomspace(x1, x2, 31)
plt.figure()
his, ed, pat = plt.hist(np.hstack([bhbcata[3],bhbroot[3]]), lmnsc, alpha=0.5, label=r'$f_{\rm NSC}=1$, $M_{\star>10^{5}\ \rm M_{\odot}$')
base = midbin(ed)
mod = his*focc_nsc(base)
plt.plot(base, mod, 'k-', marker='^', label=r'$f_{\rm NSC}\equiv f_{\rm NSC}(M_{\star})$ based on obs.')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$dN/d\log M_{\star}\ [\rm a.u.]$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig(repo+'bhbmg_dis.pdf')
plt.close()

GWrate = ntot1/ntree/Vcom * 1e9*YR/TZ(0)
GWrate1 = np.sum(mod)/ntree/Vcom * 1e9*YR/TZ(0)
print('Average BHB merger rate density ~ {:.2e} ({:.2e}) yr^-1 Gpc^-3'.format(GWrate, GWrate1))

"""

bhbm = np.array(retxt(drep+'bhb_mass_dis'+fs,2,2))
bhbm0 = np.array(retxt(repref+'bhb_mass_dis'+fs,2,2))

r1, r2 = 3, 300
bhbr = np.array(retxt(repref+'bhb_loc_dis'+fs,2,2))
bhbr0 = np.array(retxt(repref+'bhb_loc_dis_root'+fs,2,2))
bhbr1 = np.array(retxt(drep+'bhb_loc_dis_root'+fs,2,2))
ntot = np.sum(bhbm0[1])
ntot10 = np.sum(bhbm[1])
fbhb = np.sum(bhbr[1])/ntot
fbhb0 = np.sum(bhbr0[1])/ntot
fbhb1 = np.sum(bhbr1[1])/ntot10
#print(ntot, np.sum(bhbr0[1]), np.sum(bhbr0[1])/ntot)
#lr = np.geomspace(r1, r2, 8)
#tt = r'$f_{\bullet}'+r'(<{:.1f}\ \rm pc)\simeq {:.1f}\ /\ {:.1f}$ %'.format(r2,100*fbhb,100*fbhb0) \
#	+' (formation sites / target halo)'
tt = r'$f_{\bullet}'+r'(<{:.1f}\ \rm pc)\simeq {:.1f}\ {}\ {:.1f}$ %'.format(r2,100*fbhb1,r'{\rm and}',100*fbhb0) + ', w and w/o DF'
y1, y2 = 1e-6, 3
plt.figure()
plt.loglog(bhbr[0], np.cumsum(bhbr[1])/ntot, label='Formation sites') #, marker='o'
plt.loglog(bhbr1[0],np.cumsum(bhbr1[1])/ntot10, 'k--', label=r'Target halo (w/ DF, $e_{\rm if}='+str(eif)+'$)') #, marker='^'
plt.loglog(bhbr0[0],np.cumsum(bhbr0[1])/ntot, '-.', label='Target halo (w/o DF)') #, marker='x'
x1, x2 = 4, 10
y0, alp = 5e-3, 2
plt.loglog([x1,x2], [y0, y0*(x2/x1)**alp], 'k:')#, label=r'$f_{\bullet}(<r)\propto r^{'+str(alp)+'}$')
plt.text(x1*1.5, y0, r'$\propto r^{'+str(alp)+'}$')
x1, x2 = 10, 40
y0, alp = 2e-4, 1.5
plt.loglog([x1,x2], [y0, y0*(x2/x1)**alp], 'k:')#, label=r'$f_{\bullet}(<r)\propto r^{'+str(alp)+'}$')
plt.text(x1*1.5, y0*1.5**(alp+1)*1.5, r'$\propto r^{'+str(alp)+'}$')
plt.xlim(r1, r2)
plt.ylim(y1, y2)
plt.text(r1*1.1, y2/3,tt)
plt.legend(loc=4,ncol=1)
#plt.title('w/o dynamical friction')
plt.xlabel(r'$r\ [\rm pc]$')
plt.ylabel(r'$f_{\bullet}(<r)$') #\equiv N(<r)/N_{\rm tot}$')
plt.tight_layout()
plt.savefig(repo+'bhbloc_dis.pdf')
plt.close()
print('Initial (final) fraction of BHBs within {:.1f} pc: {:.2e} ({:.2e})'.format(r2, fbhb, fbhb0))

#drep = 'bhb_stat/' 

#nb = 8
nb0 = 8
x1, x2 = bhb_min, bhb_max
ntot0 = np.sum((lmbhb>x1)*(lmbhb<x2))
lm = np.geomspace(x1, x2, nb+1)
lm0 = np.geomspace(x1, x2, nb0+1)
ntot = np.sum(bhbm[1])
print('N_BHB per tree: {:.1f}'.format(ntot/ntree))
plt.figure()
his1, ed = np.histogram(np.hstack([bhbcata[0],bhbroot[0]]), lm)
plt.plot(bhbm[0], his1*ntot0/ntot1*nb/nb0, 'b-', marker='^', label=r'$r<3\ \rm pc$')
his2, ed, pat = plt.hist(lmbhb, lm0, alpha=0.3, label='Formation sites (N-body sim.)')
plt.plot(bhbm[0], bhbm[1]*ntot0/ntot*nb/nb0, 'k--', marker='o', label='Formation sites (merger trees)')
plt.legend()
plt.xlabel(r'$m_{\bullet}\ [\rm M_{\odot}]$')
plt.ylabel(r'$dN/d\log m_{\bullet}\ [\rm a.u.]$')
plt.xscale('log')
#plt.ylim(0, 0.2)
plt.xlim(x1, x2)
plt.tight_layout()
plt.savefig(repo+'bhbmass_dis.pdf')
plt.close()
#"""

#nb = 8
x1, x2 = bhb_min, bhb_max
lm = np.geomspace(x1, x2, nb+1)

def assign_bhb_info(cata1, cata2, lm, seed=233):
	lm1 = cata1[0]
	lm2 = cata2[0]
	d1 = np.array(cata1).T
	d2 = np.array(cata2).T
	out = []
	nrdm.seed(seed)
	nb = len(lm)-1
	for i in range(nb):
		sel1 = (lm1>lm[i])*(lm1<=lm[i+1])
		sel2 = (lm2>lm[i])*(lm2<=lm[i+1])
		l1 = d1[sel1]
		l2 = d2[sel2]
		if len(l1)>0 and len(l2)>0:
			lind = nrdm.choice(len(l2),len(l1))
			l3 = l2[lind]
			out.append(np.hstack([l1,l3]))
	out = np.vstack(out)
	return out.T
	
bhbpost = bhbroot
bhbpost[1] = rootz
bhbcata_all = np.hstack([bhbcata, bhbpost])
bhb = assign_bhb_info(bhbcata_all, bhbnbody, lm, seed=seed)
totxt(repo+'bhbcata.txt',bhb)

def rg_reff(r, g, rmax=100):
	y = 4./3*(2**(1/(3-g))-1)*r
	return y*(y<=rmax) + rmax*(y>rmax)

def assign_g(lm, seed, m1=mcut, m2=1e8, gr1=[0.65, 2.55], gr2=[0.65,1.35], gmax=3):
	nrdm.seed(seed)
	N = len(lm)
	r = nrdm.uniform(size=N)
	sel = (lm>=m1)*(lm<=m2)
	g1 = np.zeros(N)
	g1[lm<m1] = gr1[0]
	g1[lm>m2] = gr2[0]
	g1[sel] = gr1[0] + (gr2[0]-gr1[0])*np.log10(lm[sel]/m1)/np.log10(m2/m1)
	g2 = np.zeros(N)
	g2[lm<m1] = gr1[1]
	g2[lm>m2] = gr2[1]
	g2[sel] = gr1[1] + (gr2[1]-gr1[1])*np.log10(lm[sel]/m1)/np.log10(m2/m1) 
	g = g1+(g2-g1)*r
	g[g>gmax] = gmax
	return g

def mnc_mg(mg, sc=0.6, seed=2333):
	m = np.array(mg)
	logmnc = 0.48*np.log10(m/1e9)+6.51
	if sc>0:
		nrdm.seed(seed)
		logmnc += (2*nrdm.uniform(m.shape)-1)*sc
	return 10**logmnc

logmc = np.log10(mcut)
seed = 2333
msr0 = np.array(retxt(cr+'nsc_early.txt',2))
msr1 = np.array(retxt(cr+'nsc_late.txt',2))
m0, re0 = 10**msr0
m1, re1 = 10**msr1

logm = np.log10(np.hstack([m0, m1]))
logr = np.log10(np.hstack([re0, re1]))
logrm = np.average(logr[logm<logmc])

def to_fit(x, a, b):
	return x*b+a
	
from scipy.optimize import curve_fit
fit, pcov = curve_fit(to_fit, logm[logm>logmc], logr[logm>logmc])
ofst, alp = fit
print('offset={:.2f}, alpha={:.2f}'.format(ofst, alp))
print('Errors:', np.sqrt(np.diag(pcov)))
logrm1 = to_fit(logmc, ofst, alp)

def reff_mnc(m, alp=alp, r0=10**logrm1, r1=10**logrm1, m1=mcut, sct=0, seed=2333):
	y = r0*(m<m1)
	y += r1*10**(alp*np.log10(m/m1)) * (m>=m1)
	if sct>0:
		nrdm.seed(seed)
		ofst = (nrdm.uniform(size=len(m))-0.5) * sct*2 *3**0.5
		y *= 10**ofst
	return y

logrref = np.log10(reff_mnc(10**logm))
sct = np.std(logrref-logr)
fac = 10**(sct*3**0.5)
print(r'r_eff_0={:.2f} ({:.2f}) pc, scatter={:.2f} dex'.format(10**logrm, 10**logrm1, sct))

#"""
lm = np.geomspace(1e5, 1e9, 1000)
lr = reff_mnc(lm)
nr = 1
m0_ = m0 #np.hstack([m0]*nr), 
re0_ = re0 #np.hstack([re0]*nr)
m1_ = m1 #np.hstack([m1]*nr)
re1_ =  re1 #np.hstack([re1]*nr)
rnc0 = rg_reff(re0_, assign_g(m0_, seed))
rnc1 = rg_reff(re1_, assign_g(m1_, seed*3))

#lmmoc = np.geomspace(1e5, 1e9, 100)
#lmmoc = np.hstack([lmmoc for x in range(100)])
lmmoc = 10**(4*nrdm.uniform(size=3000)+5)
lreffmoc = reff_mnc(lmmoc,sct=sct)
lrmoc = rg_reff(lreffmoc, assign_g(lmmoc,seed*2))
x1, x2 = 1e5, 1e9
plt.figure(figsize=(6,8))
plt.subplot(211)
y1, y2 = 0.7, 3e2
plt.loglog(m0, re0, 'o', color='r', label='Early-type', alpha=0.7)
plt.loglog(m1, re1, '^', color='b', label='Late-type', alpha=0.7)
plt.errorbar([2.5e7], [6.5], xerr=[1.8e7], yerr=[2.7], marker='*', label='MW', color='g', markersize=16)
plt.plot(lm, lr, 'k--')
plt.fill_between(lm, lr*fac, lr/fac, fc='gray', alpha=0.5)
plt.legend()
#plt.xlabel(r'$M_{\rm NSC}\ [\rm M_{\odot}]$')
plt.ylabel(r'$R_{\rm eff}\ [\rm pc]$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.subplot(212)
y1, y2 = 0.3, 1e2
plt.loglog(m0_, rnc0, 'o', color='r', label='Early-type')#, alpha=0.5)
plt.loglog(m1_, rnc1, '^', color='b', label='Late-type')#, alpha=0.5)
plt.plot([2.5e7], [4.4], marker='*', label='MW', color='g', markersize=16)
plt.plot(lmmoc, lrmoc, '.', color='gray', alpha=0.2, zorder=0)
#plt.legend()
plt.xlabel(r'$M_{\rm NSC}\ [\rm M_{\odot}]$')
plt.ylabel(r'$R_{\rm NSC}\ [\rm pc]$')
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig('rnc_mnc.pdf')
plt.close()
#"""
def fGW(M, a, e):
	out = (GRA*M*Msun)**0.5/np.pi
	out *= (1+e)**1.1954/(a*(1-e**2))**1.5
	return out
	
def plot_bhb(d, lab, rep='./', mode=0, aflag=0, y1=0, log=1, gyr=0, sep=0):
	l = d['l']
	m1, m2, a0, e0, r0, e, M, R, g, H, lnL, kap, cd, eflag = d['para']
	tdes, tmg, pev, tdf0, aej, tmg_ej = d['out']
	ana = d['ana']
	x2 = np.max(l[0]/1e6*1.01)
	if mode==0:
		xb = l[0]/1e6
		xr = [0, x2]
	else:
		xb = np.log10((tmg-l[0])/1e6) #
		xr = [np.log10(tmg)-6, -6] #np.log10(tmg-l[0][-2])-6]
	if gyr==0:
		tt1 = r'$m_{}={}\ \rm M_{}$, $m_{}={}\ \rm M_{}$, $a_{}={}\ \rm au$, $e_{}={}$, '.format(1,m1,'\odot',2,m2,'\odot',0,a0,0,e0)+'\n'+r'$H={}$, $\kappa={}$: ${}={:.1f}\ \rm Myr$'.format(H,kap,r'\hat{t}_{\rm mg}',tmg/1e6) #$\ln\Lambda={}$
	else:
		tt1 = r'$m_{}={}\ \rm M_{}$, $m_{}={}\ \rm M_{}$, $a_{}={}\ \rm au$, $e_{}={}$, '.format(1,m1,'\odot',2,m2,'\odot',0,a0,0,e0)+'\n'+r'$H={}$, $\kappa={}$: ${}={:.3f}\ ({:.3f})\ \rm Gyr$'.format(H,kap,r'\hat{t}_{\rm mg}',tmg/1e9, tmg_ej/1e9) #$\ln\Lambda={}$
	tt2 = r'$M_{}={:.1e}\ \rm M_{}$, $R_{}={}\ \rm pc$, $\gamma={}$, ${}={}$, '.format(r'{\rm NSC}', M, '\odot', r'{\rm NSC}', R, g, r'\Delta_{\rm c}', cd)+'\n'+r'$r_{}={}\ \rm pc$, $e_{}={}$: ${}={:.1f}\ \rm Myr$'.format(0, r0, r'{\rm if}', e, r'\hat{t}_{\rm DF}', tdf0/1e6)
	plt.figure(figsize=(11,12))
	if y1==0:
		y1 = 0.1*Rsun/AU
	y2 = a0*2
	plt.subplot(322)
	plt.plot(xb, l[1])
	plt.plot(xb, l[-3], 'g--', label=r'$a_{\rm GW}$')
	if aej>y1:
		plt.plot(xr, [aej]*2, color='b', ls=(0, (10,5)), label=r'$a_{\rm ej}$')
	if mode==0 and aflag>0:
		plt.plot(ana[0]/1e6, ana[1], 'r:', label='Arca (2020), w/o GW')
	#if tdes<tmg:
	#	plt.plot([tdes/1e6]*2, [y1, y2], 'k:', label=r'$\min(t+\tau_{\rm ev})$') #t_{\rm des}\equiv 
	plt.yscale('log')
	#plt.xlabel(r'$t\ [\rm Myr]$')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	plt.ylabel(r'$a\ [\rm au]$')
	plt.legend()
	plt.title(tt1)
	plt.subplot(324)
	plt.plot(xb, 1-l[2])
	plt.yscale('log')
	#plt.xlabel(r'$t\ [\rm Myr]$')
	plt.xlim(xr)
	plt.ylim(np.min(1-l[2])*0.5, 1)
	plt.ylabel(r'$1-e$')
	y1, y2 = 1e-7, 1e9
	plt.subplot(326)
	plt.plot(xb, l[4], '-.', label=r'$a/a_{\rm HDB}$')
	plt.plot(xb, l[5], ls=(10, (10, 5)), label=r'$a_{\rm GW}/a_{\rm ej}$')
	if log>0:
		plt.scatter(xb, l[6], c=np.log10(l[-1]/YR), zorder=5)#, label=r'$t_{\rm gw}/t_{\rm ev}$', cmap=plt.cm.cividis)
		plt.clim(1, 11)
		cb = plt.colorbar()#orientation='horizontal')
		cb.ax.set_title(r'$\log(\tau_{\rm GW}/\rm yr)$')
	else:
		plt.scatter(xb, l[6], c=l[-1]/YR/1e9, zorder=5)#, label=r'$t_{\rm gw}/t_{\rm ev}$', cmap=plt.cm.cividis)
		plt.clim(0, 20)
		cb = plt.colorbar()#orientation='horizontal')
		cb.ax.set_title(r'$\tau_{\rm GW}\ [\rm Gyr]$')
	plt.plot(xr, [1]*2, 'k-', lw=0.5)
	#if tdes<np.max(l[0]):
	#	plt.plot([tdes/1e6]*2, [y1, y2], 'k:', label=r'$\min(t+\tau_{\rm ev})$')
	plt.plot(xb, l[7], color='k', ls=':', label=r'$\tau_{\rm ev}/\hat{t}_{\rm mg}$')
	plt.yscale('log')
	if mode==0:
		plt.xlabel(r'$t\ [\rm Myr]$')
	else:
		plt.xlabel(r'$\log[(\hat{t}_{\rm mg}-t)/\rm Myr]$')
	plt.xlim(xr)
	plt.ylabel(r'$\tau_{\rm GW}/\tau_{\rm ev}$')#+r' ($p_{}={:.3f}$)'.format(r'{\rm ev}', pev))
	plt.ylim(y1, y2)
	plt.legend()#loc=2)
	plt.subplot(321)
	plt.plot(l[0]/1e6, l[8])
	plt.yscale('log')
	#plt.xlabel(r'$t\ [\rm Myr]$')
	plt.xlim(0, x2)
	plt.ylabel(r'$\max(r,r_{\rm c})\ [\rm pc]$')
	plt.title(tt2)
	plt.subplot(323)
	plt.plot(l[0]/1e6, l[9])
	plt.yscale('log')
	plt.xlim(0, x2)
	#plt.xlabel(r'$t\ [\rm Myr]$')
	plt.ylabel(r'$\rho_{\star}\ [\rm M_{\odot}\ pc^{-3}]$')
	plt.subplot(325)
	plt.plot(l[0]/1e6, l[10])
	#plt.yscale('log')
	plt.xlim(0, x2)
	plt.xlabel(r'$t\ [\rm Myr]$')
	plt.ylabel(r'$\sigma_{\star}\ [\rm km\ s^{-1}]$')
	plt.tight_layout()
	plt.savefig(rep+'bhb_evo_{}.pdf'.format(lab))
	plt.close()

	if sep==0:
		return 0
	#if y1==0:
	y1 = 0.1*Rsun/AU
	y2 = a0*2
	plt.figure()
	#plt.plot(xb, l[1])
	if log>0:
		plt.scatter(xb, l[1], c=np.log10(l[-1]/YR), zorder=5)#, label=r'$t_{\rm gw}/t_{\rm ev}$', cmap=plt.cm.cividis)
		plt.clim(1, 11)
		cb = plt.colorbar()#orientation='horizontal')
		cb.ax.set_title(r'$\log(\tau_{\rm GW}/\rm yr)$')
	else:
		plt.scatter(xb, l[1], c=l[-1]/YR/1e9, zorder=5)#, label=r'$t_{\rm gw}/t_{\rm ev}$', cmap=plt.cm.cividis)
		plt.clim(0, 20)
		cb = plt.colorbar()#orientation='horizontal')
		cb.ax.set_title(r'$\tau_{\rm GW}\ [\rm Gyr]$')
	#plt.plot(xb, l[-3], 'm--', label=r'$a_{\rm GW}$')
	#plt.plot(xb, l[-4]/1e2, 'k:', label=r'$10^{-2}a_{\rm HDB}$')
	if aej>y1:
		plt.plot(xr, [aej]*2, color='b', ls='-.', label=r'$a_{\rm ej}$')
	if mode==0 and aflag>0:
		plt.plot(ana[0]/1e6, ana[1], 'r:', label='Arca (2020), w/o GW')
	plt.yscale('log')
	plt.xlabel(r'$t\ [\rm Myr]$')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	plt.ylabel(r'$a\ [\rm au]$')
	plt.legend()
	#plt.title(tt1)
	plt.tight_layout()
	plt.savefig(rep+'a_t_{}.pdf'.format(lab))
	plt.close()
	
	plt.figure()
	y1, y2 = max(np.min(l[2])*0.5, 1e-5), 1.1
	lfgw = fGW(m1+m2, l[1]*AU, l[2])
	#print(lfgw)
	plt.scatter(lfgw, l[2], c=np.log10(l[1]))
	cb = plt.colorbar()
	cb.ax.set_title(r'$\log(a\ [\rm au])$')
	plt.plot([0.1]*2,[y1,y2],'k', lw=0.5)
	plt.plot([1.0]*2,[y1,y2],'k', lw=0.5)
	plt.xlabel(r'$f_{\rm GW}\ [\rm Hz]$')
	plt.ylabel(r'$e$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(1e-4, 1e3)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig(rep+'e_fgw_{}.pdf'.format(lab))
	plt.close()

def vesc(M, rho):
	return 40e5*(M/1e5)**(1/3)*(rho/1e5)**(1/6)

def dadtGW(a, e, m1, m2):
	f1 = (1+73./24*e**2+37./96*e**4)
	A = GRA**3*m1*m2*(m1+m2)/SPEEDOFLIGHT**5
	return -64./5*A/(a**3*(1-e**2)**3.5)*f1

def dedtGW(a, e, m1, m2):
	f2 = (1+121./304*e**2)
	A = GRA**3*m1*m2*(m1+m2)/SPEEDOFLIGHT**5
	return -304./15*e*A/(a**4*(1-e**2)**2.5)*f2
	
def dadt3b(a, rho, sigma, H=20):
	return -GRA*H*rho/sigma*a**2
	
def dedt3b(a, rho, sigma, H=20, kap=0.1):
	return kap*GRA*H*rho/sigma*a

def abhb_ana(t2, a0, r0, R, rho0, sigma0, tdf0, g, delta, beta, H):
	ag0 = sigma0/(H*GRA*rho0)*(r0/R)**(g+delta/2)
	ft = 2*tdf0/(2*g+delta-2*beta)*((1-beta*t2*YR/tdf0)**((2*g+delta-2*beta)/2/beta)-1)
	return a0/(1-a0*AU/ag0*ft)

def sigmafac(g):
	return 0.2 + 0.11*g

def abhb_t(t2, nt, m1, m2, a0, e0, r0, M, R, e=0.5, g=1, alpha=-0.67, beta=1.76, H=20, kap=0.1, emax=1-1e-8, mfield=1, cd=100, lnL=0, wt=0, dehnen=1, fac=1):
	rho0 = (3-g)*M*Msun/(4*np.pi*(R*PC)**3) # cgs
	if fac>0:
		sigma0 = (GRA*M*Msun/(R*PC))**0.5*fac # cgs
	else:
		sigma0 = (GRA*M*Msun/(R*PC))**0.5*sigmafac(g)
	lnL0 = np.log(R*PC*sigma0**2/GRA/(m1+m2)/Msun)
	if lnL==0:
		lnL = lnL0
	tdf0 = tdf(r0, m1+m2, M, R, e, g, alpha, beta, lnL=lnL0)*YR # cgs
	rmin = R*(2*(m1+m2)/M)**(1/(3-g)) # pc
	if g>0 and cd>0:
		rc = R/cd**(1.0/g)
	else:
		rc = 0
	rmin = max(rmin, rc)
	def r_t(t):
		x = 1.0-beta*t/tdf0
		y = x*(x>(rmin/r0)**beta) + (rmin/r0)**beta*(x<=(rmin/r0)**beta) # pc
		return r0*y**(1.0/beta)
	delta = (2-g)*(g>=1) + g*(g<1)
	if dehnen>0:
		rho_r = lambda r: rho0*(r/R)**-g*(1+r/R)**(g-4) # cgs
		sigma_r = lambda r: sigma0*(r/R)**(delta/2)#/(1.0+r/R)**(3.0-g)
	else:
		rho_r = lambda r: rho0*(r/R)**-g
		sigma_r = lambda r: sigma0*(r/R)**(delta/2) # cgs
	y0 = [a0*AU, e0]
	def fun(t, y):
		a = y[0]
		e = max(min(y[1], emax), 0)
		r = r_t(t)
		dadt = dadtGW(a, e, m1*Msun, m2*Msun) + dadt3b(a, rho_r(r), sigma_r(r), H)
		dedt = dedtGW(a, e, m1*Msun, m2*Msun) + dedt3b(a, rho_r(r), sigma_r(r), H, kap)
		if e==emax:
			dedt = min(dedt, 0)#emax-y[1])
		return [dadt, dedt]
	amin = 6*GRA*(m1+m2)*Msun/SPEEDOFLIGHT**2
	def merger(t, y):
		a = y[0]
		return amin-a
	merger.terminal = True
	merger.direction = 1.0
	#tr = np.linspace(0, t2*YR, nt)
	if nt>0:
		sol = solve_ivp(fun, [0, t2*YR], y0, events=merger, max_step=t2*YR/nt)#, t_eval=tr)
	else:
		sol = solve_ivp(fun, [0, t2*YR], y0, events=merger)
	lt = sol.t/YR #np.hstack([sol.t, sol.t_events[0]])/YR
	la = sol.y[0]/AU #np.hstack([sol.y[0], sol.y_events[0][0]])/AU
	le = sol.y[1] #np.hstack([sol.y[1], sol.y_events[0][1]])
	le[le<0] = 0
	eflag = np.sum(le>=1)
	le[le>emax] = emax
	tmg = np.max(lt)
	lr = r_t(sol.t)
	lrho = rho_r(lr)
	lsigma = sigma_r(lr)
	sel = lt*YR<tdf0/beta
	lt_ana = lt[sel]
	la_ana = abhb_ana(lt_ana, a0, r0, R, rho0, sigma0, tdf0, g, delta, beta, H)
	if lnL==lnL0:
		llnL = np.log(lr*PC*lsigma**2/2/GRA/(m1+m2)/Msun)
	else:
		llnL = lnL
	lahd = GRA*m1*m2*Msun/mfield/lsigma**2/AU#*(1-le)/(1+le)
	#lahd = GRA*(m1+m2)*Msun/lsigma**2/AU/2
	vesc0 = vesc(M, rho0/Msun*PC**3)
	aej = GRA*H/np.pi*mfield**2*m1*m2*Msun/(m1+m2)**3/vesc0**2/AU
	lagw = (64*GRA**2/5/H/SPEEDOFLIGHT**5*lsigma/lrho*m1*m2*(m1+m2)*Msun**3/(1-le**2)**3.5*(1+73./24*le**2+37./96*le**4))**0.2/AU
	rat1 = la/lahd
	rat2 = lagw/aej
	ltev = 3**0.5*lsigma/(32*np.pi**0.5*GRA*lrho*llnL*la*AU)*(m1+m2)/mfield
	ltev0 = 3**0.5*lsigma/(32*np.pi**0.5*GRA*lrho*llnL*a0*AU)*(m1+m2)/mfield
	ltgw = 5./256*SPEEDOFLIGHT**5*(la*AU)**4*(1-le**2)**3.5/(GRA**3*m1*m2*(m1+m2)*Msun**3*(1+73./24*e**2+37./96*e**4)) / (1.0-0.8*le)
	rat3 = ltgw/ltev
	rat4 = ltev/YR/(np.max(lt))#-lt)
	tdes = np.min(lt+ltev/YR)
	sel = (ltev0<ltgw[0])#*(rat1>1)
	pev = 0#1-np.exp(-np.trapz(YR/ltev0[sel], lt[sel]))
	if sum((la<aej)*(rat2<1))>0:
		sel = (la<aej)*(rat2<1)
		tmg_ej = np.min(lt[sel])+np.max(ltgw[sel])/YR
	else:
		tmg_ej = tmg
	out = [tdes, tmg, pev, tdf0/beta/YR, aej, tmg_ej]
	para = [m1, m2, a0, e0, r0, e, M, R, g, H, lnL, kap, cd, eflag]
	l = lt, la, le, la*AU/amin, rat1, rat2, rat3, rat4, lr, lrho/Msun*PC**3, lsigma/1e5, lahd, lagw, ltev, ltgw
	d = {}
	d['l'] = l
	d['para'] = para
	d['out'] = out
	d['ana'] = [lt_ana, la_ana]
	if wt>0:
		print('Parameters (m1 [Msun], m2 [Msun], a0 [au], e0, r0 [pc], e, M [Msun], R [pc], g, H, lnL, kap, cd):',para)
		print('Surviving probability (if soft): {:.2e}, tev={:.2e} Myr'.format(1-pev, -tmg/np.log(1-pev)/1e6))
		print('tmg={:.2e} ({:.2e}) Myr, tdf0/beta={:.2e} Myr'.format(tmg/1e6, tmg_ej/1e6, tdf0/beta/YR/1e6)) 
		print(r'rmin={:.2e} pc, amin={:.2e} au, aej={:.2e} au, vesc={:.2f} km/s'.format(rmin, amin/AU, aej, np.max(vesc0)/1e5))
	return d

def evo_bhb_nsc(cata, kap=0.1, cd=100, ro = 3.0, sct=0.32, seed=2333, mfield=1, edis=0,tend=15e9,fac=1,fgw0=1,fub=1):
	raw = np.array(cata).T
	lm = cata[4]
	lm1, lm2 = cata[5], cata[6]
	la = cata[7]
	le = cata[8]
	lmc = cata[2]
	lmg = cata[3]
	lzif = cata[1]
	lreff = reff_mnc(lmc, sct=sct, seed=seed)
	lg = assign_g(lmc, seed*2)
	lrc = rg_reff(lreff, lg)
	nrdm.seed(seed*3)
	n = len(lm)
	out = []
	ndes = 0
	for i in range(n):
		if i%1000==0 or i==n-1:
			print('{:.4f}%'.format(100*(i+1)/n))
		if fac==0:
			sigma = (GRA*lmc[i]*Msun/(lrc[i]*PC))**0.5*sigmafac(lg[i])
		else:
			sigma = (GRA*lmc[i]*Msun/(lrc[i]*PC))**0.5*fac
		ahd = GRA*lm1[i]*lm2[i]*Msun/mfield/sigma**2/AU
		#if lm[i]>55 and lm[i]<70:
		#	print('pre: ',la[i],le[i],lzif[i],lmc[i],lmg[i])
		#if la[i]>ahd or la[i]*AU>lrc[i]*PC/cd**(1/lg[i]):
		if la[i]>ahd or la[i]*AU>lrc[i]*PC/cd**(1/lg[i]):
			continue
		#if lrc[i]>ro:
		#	r0 = ro
		#else:
		#	r0 = nrdm.uniform()*(ro-lrc[i]) + lrc[i]
		r0 = ro
		if edis>0:
			e = nrdm.uniform()**0.5
		else:
			e = 0.0
		d = abhb_t(tend,0,lm1[i],lm2[i],la[i],le[i],r0,lmc[i],lrc[i],e,lg[i],kap=kap,cd=cd)
		if d['para'][-1]>0:
			#print('Destroyed!')
			ndes += 1
			if fub>0:
				continue
		tif = TZ(lzif[i])/YR
		tmg = d['out'][-1]
		tmg0 = d['out'][1]
		#if lm[i]>55 and lm[i]<70:
		#	print('post: ', tmg0/1e6, tmg/1e6)
		if tmg+tif>TZ(0)/YR:
			z = -1
		else:
			z = min(ZT(np.log10((tmg+tif)/1e9)), lzif[i])
		lfgw = fGW(lm1[i]+lm2[i], d['l'][1]*AU, d['l'][2])
		#selfgw = lfgw<=fgw0
		#eout = d['l'][2][selfgw]
		selfgw = lfgw>fgw0
		if np.sum(selfgw)==0:
			eout = d['l'][2][-1]
		else:
			eout = np.max(d['l'][2][selfgw])
		ly = [eout, z, tmg0, tmg, d['out'][3]]
		#print(i, ly)
		out.append(np.hstack([raw[i],ly]))
	print('Destroyed fraction: {:.2e}'.format(ndes/n))
	return np.array(out).T

def nsc_regu(dmg, seed=1314):
	nrdm.seed(seed)
	raw = np.array(dmg).T
	n = len(raw)
	out = []
	for i in range(n):
		mg = raw[i][3]
		f = focc_nsc(mg)
		r = nrdm.uniform()
		if r<f:
			out.append(raw[i])
	return np.array(out).T

def gw190521_sel(dmg,m1d=71,m1u=106,md=133,mu=179):
	raw = np.array(dmg).T
	n = len(raw)
	out = []
	for i in range(n):
		m1, m2 = raw[i][5], raw[i][6]
		ma = max(m1, m2)
		m = m1 + m2
		if ma>=m1d and ma<=m1u and m>=md and m<=mu:
			out.append(raw[i])
	return np.array(out).T
	
def ppisn_sel(dmg,md=55,mu=85):
	raw = np.array(dmg).T
	n = len(raw)
	out = []
	for i in range(n):
		m1, m2 = raw[i][5], raw[i][6]
		#ma = max(m1, m2)
		#m = m1 + m2
		#if ma>=m1d and ma<=m1u and m>=md and m<=mu:
		if (m1>=md and m1<=mu) or (m2>=md and m2<=mu):
			out.append(raw[i])
	return np.array(out).T

def mchirp(m1, m2):
	return (m1*m2)**0.6/(m1+m2)**0.2

def event_dis(x, xp, xm, lx, ed, fac=1.7):
	n = len(x)
	y = np.zeros(lx.shape[0])
	for i in range(n):
		sel1 = lx>x[i]
		sel2 = lx<=x[i]
		s1 = xp[i]/fac
		s2 = xm[i]/fac
		y[sel1] += np.exp(-(lx[sel1]-x[i])**2/(2*s1**2))/s1/(2*np.pi)**0.5
		y[sel2] += np.exp(-(lx[sel2]-x[i])**2/(2*s2**2))/s2/(2*np.pi)**0.5
	n0 = len(ed)
	out = np.zeros(n0-1)
	for i in range(n0-1):
		sel = (lx>ed[i])*(lx<=ed[i+1])
		out[i] = np.trapz(y[sel],lx[sel])
	return out

#ligo = np.array(retxt(cr+'LIGO_O3a.txt',6))
ligo = np.array(retxt(cr+'ogc3.txt',22,1))
#95.3 28.7 18.9 69.0 22.7 23.1
#ind0 = np.argmax(ligo[2])
#print(ligo[0][ind0])

if __name__=='__main__':
	fac = 1
	cata = np.array(retxt(repo+'bhbcata.txt',9))
	cata1 = nsc_regu(cata)
	llab = ['CS', 'OP']
	
	"""
	bhbnbody = np.array(bhbnbody)
	sel = (bhbnbody[0]>55)*(bhbnbody[0]<70)
	la = bhbnbody[3][sel]
	la0 = bhbnbody[3]
	ab = np.geomspace(1, 1e6, 51)
	plt.figure()
	plt.hist(la0, ab, alpha=0.5, label='All')
	plt.hist(la, ab, histtype='step', lw=1.5, label=r'$m\sim 30-60\ \rm M_{\odot}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(r'$a_{0}\ [\rm au]$')
	plt.ylabel(r'$dN/d\loga_{0}\ [\rm a.u.]$')
	plt.tight_layout()
	plt.savefig(repo+'a0_dis.pdf')
	plt.close()
	print(np.sum(sel)/len(bhbnbody[0]),sumy(la))
	sel = (cata[4]>55)*(cata[4]<70)
	print(np.sum(sel)/len(cata[0]),sumy(cata[7][sel]))
	print(sumy(cata[2][sel]))
	print(sumy(cata[3][sel]))
	"""
	
	if op==0:
		cd = 20.0
		kap = 0.01
		repo = repo+'lowdens/'
		y0, b = 3, 1/5
		z01, y01, b1 = 6, 4.5*fbn0, 1/6
		nt = 50
		t1, t2 = 1e-1, 1.5e4
		red = 5
	else:
		cd = 100.0
		kap = 0.1
		y0, b = 2, 1/5
		z01, y01, b1 = 6, 1, 1/6
		nt = 50
		t1, t2 = 1e-1, 1.5e4
		red = 5.5
	if not os.path.exists(repo):
		os.makedirs(repo)
		
	if mode==0:
		dmg = evo_bhb_nsc(cata,cd=cd,kap=kap,seed=seed,fac=fac,fgw0=fgw0,fub=fub)
		#print(dmg.shape)
		totxt(repo+'bhbmerger.txt',dmg)
	else:
		dmg = np.array(retxt(repo+'bhbmerger.txt',14))
	
	dmg1 = nsc_regu(dmg)
	dmg2 = gw190521_sel(dmg)
	dmg3 = gw190521_sel(dmg1)
	
	zcol = 4.6
	nz = 30
	x1, x2 = 0, 15
	zed = np.linspace(x1, x2, nz+1)
	zbase = midbin(zed)
	dt = np.array([(TZ(zed[i])-TZ(zed[i+1]))/YR for i in range(nz)])
	his, ed = np.histogram(dmg[-4], zed)
	his1, ed = np.histogram(dmg1[-4], zed)
	his2, ed = np.histogram(dmg2[-4], zed)
	his3, ed = np.histogram(dmg3[-4], zed)
	mrd = his/dt/ntree/Vcom * 1e9 * fbn*fnsc0/fnsc
	mrd1 = his1/dt/ntree/Vcom * 1e9 * fbn*fnsc0/fnsc
	mrd2 = his2/dt/ntree/Vcom * 1e9 * fbn*fnsc0/fnsc
	mrd3 = his3/dt/ntree/Vcom * 1e9 * fbn*fnsc0/fnsc
	print('Peak MRD = {:.2e} yr^-1 Gpc^-3 at z = {:.1f}'.format(np.max(mrd), zbase[np.argmax(mrd)]))
	print('Peak MRD = {:.2e} yr^-1 Gpc^-3 at z = {:.1f}'.format(np.max(mrd1), zbase[np.argmax(mrd1)]))
	y1, y2 = 1e-2, 100/(11-op*10)
	plt.figure()
	plt.plot(zbase, mrd, label=llab[op]+'_F')#r'$f_{\rm NSC}=1$, $M_{\star}>10^{5}\ \rm M_{\odot}$')
	plt.plot(zbase, mrd1, '--', label=llab[op]+'_P')#r'$f_{\rm NSC}\equiv f_{\rm NSC}(M_{\star})$ based on obs.')
	plt.plot(zbase, mrd2, 'k-.', label='GW190521-like')
	plt.plot(zbase, mrd3, 'k-.')#, label='GW190521-like')
	plt.fill_between([x1, zcol],[y1,y1],[y2,y2],fc='gray',alpha=0.3)
	plt.fill_between([0.48, 1.1], [0.02, 0.02], [0.43, 0.43], color='g', alpha=0.5, label='GW190521')
	z0, z1 = zcol, 0
	#plt.plot([z0, z1], [y0, y0*10**(b*(z1-z0))], 'm:')#, label='Extrapolation')
	if op>0:
		plt.plot([z01, z1], [y01, y01*10**(b*(z1-z01))], 'k:')
		plt.plot([z01, z1], [y01/red, y01*10**(b1*(z1-z01))/red], 'k:')
	plt.legend(ncol=2)
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\dot{n}_{\rm GW}\ [\rm yr^{-1}\ Gpc^{-3}]$')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(repo+'mrd_z.pdf')
	plt.close()
	
	print('Occ reduction factor:', len(cata[0])/len(cata1[0]))
	sel = dmg[-4]>0
	print('HDB (merge) fraction: {:.4e} ({:.8e})'.format(np.sum(sel)/len(cata[0]), np.sum(sel)/len(dmg[0])))
	print('Ejection fraction: {:.4e}'.format((len(dmg[0])-np.sum(dmg[-3]==dmg[-2]))/len(cata[0])))
	sel = dmg1[-4]>0
	print('HDB (merge) fraction: {:.4e} ({:.8e})'.format(np.sum(sel)/len(cata[0]), np.sum(sel)/len(dmg1[0])))
	print('Ejection fraction: {:.4e}'.format((len(dmg1[0])-np.sum(dmg1[-3]==dmg1[-2]))/len(cata1[0])))
	sel = dmg2[-4]>0
	print('HDB (merge) fraction: {:.4e} ({:.8e})'.format(np.sum(sel)/len(cata[0]), np.sum(sel)/len(dmg2[0])))
	#print('Ejection fraction: {:.4e}'.format(1-np.sum(dmg2[-3]==dmg2[-2])/len(dmg2[0])))
	sel = dmg3[-4]>0
	print('HDB (merge) fraction: {:.4e} ({:.8e})'.format(np.sum(sel)/len(cata[0]), np.sum(sel)/len(dmg3[0])))
	#print('Ejection fraction: {:.4e}'.format(1-np.sum(dmg3[-3]==dmg3[-2])/len(dmg3[0])))

	
	log = 1
	#t1, t2 = 1e-3, 1e3
	if log>0:
		ted = np.geomspace(t1, t2, nt+1)
	else:
		ted = np.linspace(t1, t2, nt+1)
	y1, y2 = 0.5, 5e4*(60/nt)*(ntree/100)
	plt.figure()
	plt.hist(dmg[-2]/1e6,ted, alpha=0.5, label=llab[op]+'_F')#r'$f_{\rm NSC}=1$, $M_{\star}> 10^{5}\ \rm M_{\odot}$')
	plt.hist(dmg1[-2]/1e6,ted, histtype='step',lw=1.5, label=llab[op]+'_P')#r'$f_{\rm NSC}\equiv f_{\rm NSC}(M_{\star})$ based on obs.')
	plt.legend()
	plt.xlim(t1,t2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$t_{\rm delay}\ [\rm Myr]$')
	if log>0:
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel(r'$dN/d\log t_{\rm delay}\ [\rm a.u.]$')
	else:
		plt.ylabel(r'$dN/dt_{\rm delay}\ [\rm a.u.]$')
	plt.tight_layout()
	plt.savefig(repo+'tdelay_dis.pdf')
	plt.close()
	
	sel = (dmg[-4]>0)*(dmg[1]>=dmg[-4])
	sel1 = (dmg1[-4]>0)*(dmg1[1]>=dmg1[-4])
	
	leo = dmg[-5][sel*(dmg[-5]>=0)]
	leo1 = dmg1[-5][sel1*(dmg1[-5]>=0)]
	eref = 0.1
	print('Fraction with e data: {:.2e} (F)'.format(len(leo)/np.sum(sel)))
	print('Fraction of e({} Hz)>{}: {:.2e} (F)'.format(fgw0, eref, np.sum(leo>eref)/np.sum(sel)))
	print('Fraction with e data: {:.2e} (P)'.format(len(leo1)/np.sum(sel1)))
	print('Fraction of e({} Hz)>{}: {:.2e} (P)'.format(fgw0, eref, np.sum(leo1>eref)/np.sum(sel1)))
	e1, e2 = 1e-5, 1
	ne = 50
	eed = np.geomspace(e1, e2, ne+1)
	y1, y2 = 0.5, 100*ntree
	plt.figure()
	plt.hist(leo, eed, alpha=0.3, label=llab[op]+'_F')
	plt.hist(leo1, eed, histtype='step', lw=1.5, ls='-', color='b', label=llab[op]+'_P')
	plt.xlim(e1, e2)
	plt.ylim(y1, y2)
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(r'$e (f_{\rm gw}='+str(fgw0)+r'\ \rm Hz)$')
	plt.ylabel(r'$dN/d\log e\ [\rm a.u.]$')
	plt.tight_layout()
	plt.savefig(repo+'eend_dis.pdf')
	plt.close()
	
	log = 1
	nt = 35
	t1, t2 = 3, 1e4
	if log>0:
		ted = np.geomspace(t1, t2, nt+1)
	else:
		ted = np.linspace(t1, t2, nt+1)
	y1, y2 = 0.5, 1e5*(35/nt)*(ntree/100)
	plt.figure()
	plt.hist(dmg[-1]/1e6,ted, alpha=0.3, label=llab[op]+'_F (all in NSCs)')
	plt.hist(dmg[-1][sel]/1e6,ted, histtype='step',lw=1.5, ls='-', color='b', label=llab[op]+r'_F (merged at $z>0$)')
	plt.hist(dmg1[-1]/1e6,ted, histtype='step',lw=1.5, ls='--', color='k', label=llab[op]+'_P (all in NSCs)')
	plt.hist(dmg1[-1][sel1]/1e6,ted, histtype='step',lw=1.5, ls='-.', color='r', label=llab[op]+r'_P (merged at $z>0$)')
	plt.legend()
	plt.xlim(t1,t2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$\hat{t}_{\rm DF}\ [\rm Myr]$')
	if log>0:
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel(r'$dN/d\log \hat{t}_{\rm DF}\ [\rm a.u.]$')
	else:
		plt.ylabel(r'$dN/d\hat{t}_{\rm DF}\ [\rm a.u.]$')
	plt.tight_layout()
	plt.savefig(repo+'tdf_dis.pdf')
	plt.close()
	
	log = 0
	lm1, lm2 = dmg[5][sel], dmg[6][sel]
	lm = dmg[4][sel] #lm1+lm2
	#sel = (lm>55)*(lm<70)
	#print(np.sum(sel))
	#lm = bhbnbody[0]
	#print(np.sum(lm1==lm2)/len(lm1))
	lma = np.copy(lm1)
	lma[lm1<lm2] = lm2[lm1<lm2]
	lmb = np.copy(lm2)
	lmb[lm2>lm1] = lm1[lm2>lm1]
	lmc = mchirp(lm1, lm2)
	lm1, lm2 = dmg1[5], dmg1[6]
	lmc1 = mchirp(lm1, lm2)
	lmcligo = ligo[7] #mchirp(ligo[0],ligo[3])
	lmligo = ligo[1] + ligo[4]
	#print('Chirp mass of GW190521: ', mchirp(85, 66))
	nm = 40
	nm0 = 20
	nm1 = 20
	if log>0:
		y1, y2 = 1, 3e4*(ntree/100)*(40/nm)
		x1 = 1
		x2 = 100
		med = np.geomspace(x1, x2, nm+1)
		med0 = np.geomspace(x1, x2, nm0+1)
		med1 = np.geomspace(x1, x2, nm1+1)
	else:
		y1, y2 = 1, 5e3*(ntree/100)*(40/nm)*(2+op)/3
		x1 = 0
		x2 = 100
		med = np.linspace(x1, x2, nm+1)
		med0 = np.linspace(x1, x2, nm0+1)
		med1 = np.linspace(x1, x2, nm1+1)
	#mbase = midbin(med1)
	#lmdis = np.linspace(x1, x2, 10000)
	#mcdisligo = event_dis(ligo[7], ligo[8], ligo[9], lmdis, med1)*len(lm)/len(lmcligo)*nm1/nm
	#mdisligo = event_dis(lmligo/2.0, (ligo[2]**2+ligo[5]**2)**0.5/2.0, (ligo[3]**2+ligo[6]**2)**0.5, lmdis, med1)*len(lm)/len(lmcligo)*nm1/nm
	plt.figure()
	plt.hist(lmc, med, label=r'$m_{\rm chirp}$, OP_F', alpha=0.3)
	#plt.hist(lmc1, med, histtype='step', lw=1.5, ls='-', label=r'$m_{\rm chirp}$, $f_{\rm NSC}(M_{\star})<1$', color='b', weights=np.ones(len(lmc1))*len(lm)/len(lm1))
	plt.hist(lm/2.0, med, histtype='step', lw=1.5, ls='-', label=r'$m_{\bullet}/2$, OP_F', color='b')
	plt.hist(lmcligo, med0, histtype='step', lw=1.5, ls='--', label=r'$m_{\rm chirp}$, 3-OGC', color='k', weights=np.ones(len(lmcligo))*len(lm)/len(lmcligo)*nm0/nm)
	#plt.plot(mbase, mcdisligo, 'r:')
	plt.hist(lmligo/2.0, med0, histtype='step', lw=1.5, ls='-.', label=r'$m_{\bullet}/2$, 3-OGC', color='r', weights=np.ones(len(lmligo))*len(lm)/len(lmligo)*nm0/nm)
	#plt.plot(mbase, mdisligo, 'k:')
	#plt.hist(lma, med, histtype='step', lw=1.5, ls='-.', label=r'$m_{1}$', color='orange')
	#plt.hist(lmb, med, histtype='step', lw=1.5, ls=':', label=r'$m_{2}$',color='g')
	plt.legend(loc=1,ncol=2)
	if log>0:
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel(r'$dN/d\log m\ [\rm a.u.]$')
	else:
		#plt.yscale('log')
		plt.ylabel(r'$dN/dm\ [\rm a.u.]$')
	plt.xlabel(r'$m\ [\rm M_{\odot}]$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig(repo+'mass_dis.pdf')
	plt.close()
	
	y1, y2 = 0, 0.05
	m1ref0 = np.array(retxt(cr+'m1_3ogc_0.txt',2))
	m1ref1 = np.array(retxt(cr+'m1_3ogc_1.txt',2))
	plt.figure()
	plt.hist(lma, med, label=r'OP_F', alpha=0.3, density=True)
	plt.plot(*m1ref0, label='Observed, 3-OGC')
	plt.plot(*m1ref1, 'k--', label='Reweighted, 3-OGC')
	plt.legend(loc=1,ncol=1)
	if log>0:
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel(r'p(m_{1}) [$\rm M_{\odot}^{-1}$]')
	else:
		#plt.yscale('log')
		plt.ylabel(r'$p(m_{1})\ [\rm M_{\odot}^{-1}$]')
	plt.xlabel(r'$m_{1}\ [\rm M_{\odot}]$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig(repo+'m1_dis.pdf')
	plt.close()
	
	#sel = dmg[-4]>0
	print('NS-BH fraction: {:.4e}'.format(np.sum((lmb<=3)*(lma>3))/np.sum(sel)))
	print('NS-NS fraction: {:.4e}'.format(np.sum((lmb<=3)*(lma<=3))/np.sum(sel)))
	
	mlow = 11
	mlows = 20
	#nlow = np.sum((lmb<=mlow)*(lma<=mlow))
	#print('Reducrion factor for mlow={:.1f} Msun: {:.4e}'.format(mlow,nlow/np.sum(sel)*(bhmax-1)/(mlow-1)))
	"""
	his, ed = np.histogram(lm, np.geomspace(bhb_min, bhb_max, nb+1))
	print('Mass base:', bhbm[0])
	f1 = np.sum(his)/np.sum(bhbm[1])
	f2 = np.sum(his[bhbm[0]<mlow])/np.sum(bhbm[1][bhbm[0]<mlow])
	print(f1, f2, f1/f2*(mlows-1)/(bhmax-1)*(bhmax-bhmin)/(bhmax-1)*(mlows-1)/(mlows-bhmin), (mlows-1)/(bhmax-1)*(bhmax-bhmin)/(bhmax-1)*(mlows-1)/(mlows-bhmin))
	print(his/bhbm[1])
	"""
	selligo = (lma<50)*(lmb<20)*(lmb>3)
	print('Fraction of low mass BH-BH mergers: {:.4e}'.format(np.sum(selligo)/len(lma)))
	
	nq = 1000
	qed = np.linspace(0, 1, nq+1)
	lq = lmb/lma
	lq0 = cata[6]/cata[5]
	lq0[lq0>1] = 1.0/lq0[lq0>1]
	print('Median mass ratio: {:.2f}'.format(np.median(lq)))
	hisq, ed = np.histogram(lq, qed)
	hisq0, ed = np.histogram(lq0, qed)
	plt.figure()
	#plt.plot(ed[:-1], np.cumsum(his)/np.sum(his), 'b-', label='$x=q$ (merged at $z>0$)')
	#plt.plot([0.5]*2,[0,1], 'k-', lw=0.5)
	plt.plot([0,1],[0.5]*2, 'k-', lw=0.5)
	plt.plot([0,1],[0,1], 'k:')
	#plt.yscale('log')
	#plt.xlim(0, 1)
	#plt.ylim(0, 1)
	#plt.xlabel(r'$q\equiv m_{2}/m_{1}$')
	#plt.ylabel(r'$N(<q)/N_{\rm tot}$')
	#plt.tight_layout()
	#plt.savefig(repo+'q_dis.pdf')
	#plt.close()

	nq = 1000
	qed = np.linspace(0, 1, nq+1)
	le = dmg[8][sel]
	le0 = cata[8]
	print('Median initial eccentricity: {:.2f}'.format(np.median(le)))
	his, ed = np.histogram(le, qed)
	his0, ed = np.histogram(le0, qed)
	plt.plot(ed[:-1], np.cumsum(his0)/np.sum(his0), 'm-', lw=4.5, alpha=0.5, label='$x=e_{0}$ (all in NSCs)')
	plt.plot(ed[:-1], np.cumsum(his)/np.sum(his), 'b-', label='$x=e_{0}$ (merged at $z>0$)')
	plt.plot(ed[:-1], np.cumsum(hisq0)/np.sum(hisq0), 'k--', label='$x=q$ (all in NSCs)')
	plt.plot(ed[:-1], np.cumsum(hisq)/np.sum(hisq), 'r-.', label='$x=q$ (merged at $z>0$)')
	#plt.plot([0,1],[0.5]*2, 'k--')
	#plt.yscale('log')
	plt.legend()
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel(r'$x=e_{0}$, $q\equiv m_{2}/m_{1}$')
	plt.ylabel(r'$N(<x)/N_{\rm tot}$')
	plt.tight_layout()
	plt.savefig(repo+'q_e_dis.pdf')
	plt.close()	
	print('q_min={:.4e}'.format(np.min(lq)))

	y1, y2 = 10*(ntree/100), 5e4*(ntree/100)
	x1, x2 = 1e5, 1e8
	lmnsc = np.geomspace(x1, x2, 25)
	plt.figure()
	plt.hist(cata[2], lmnsc, alpha=0.3, label='All in NSCs')
	plt.hist(dmg[2][sel], lmnsc, histtype='step', lw=1.5, ls='-', label='Merged at $z>0$')
	#plt.hist(cata[2], lmnsc, alpha=0.3, label=llab[op]+'_F (all in NSCs)')
	#plt.hist(dmg[2][sel], lmnsc, histtype='step', lw=1.5, ls='-', label=llab[op]+'_F (merged at $z>0$)')
	#plt.hist(cata1[2], lmnsc, histtype='step', lw=1.5, ls='--', color='k', label=llab[op]+'_P (all in NSCs)')
	#plt.hist(dmg1[2][sel1], lmnsc, histtype='step', lw=1.5, ls='-.', color='r', label=llab[op]+'_P (merged at $z>0$)')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(r'$M_{\rm NSC}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$dN/d\log M_{\rm NSC}\ [\rm a.u.]$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig(repo+'bhbmnsc_dis.pdf')
	plt.close()
	q = [50, 90]
	print('mnsc percentile: ', np.percentile(dmg[2][sel], q), np.percentile(dmg1[2][sel1], q))
	print('mg percentile: ', np.percentile(dmg[3][sel], q), np.percentile(dmg1[3][sel1], q))

	y1, y2 = 10*(ntree/100), 3e4*(ntree/100)
	x1, x2 = 1e5, 1e11
	lmnsc = np.geomspace(x1, x2, 33)
	plt.figure()
	his, ed, pat = plt.hist(cata[3], lmnsc, alpha=0.3, label=llab[op]+'_F (all in NSCs)')#r'$f_{\rm NSC}=1$, $M_{\star}>10^{5}\ \rm M_{\odot}$')
	plt.hist(dmg[3][sel],lmnsc, histtype='step', lw=1.5, ls='-', label=llab[op]+r'_F (merged at $z>0$)')#r'$f_{\rm NSC}=1$ (Merged at $z>0$)')
	base = midbin(ed)
	mod = his*focc_nsc(base)
	his_, ed, pat = plt.hist(cata1[3], lmnsc, histtype='step', lw=1.5, ls='--', color='k', label=llab[op]+'_P (all in NSCs)')#r'$f_{\rm NSC}\equiv f_{\rm NSC}(M_{\star})$ based on obs.')
	plt.hist(dmg1[3][sel1],lmnsc, histtype='step', lw=1.5, ls='-.', color='r', label=llab[op]+r'_P (merged at $z>0$)')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$dN/d\log M_{\star}\ [\rm a.u.]$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig(repo+'bhbmg_dis.pdf')
	plt.close()
	fmw1 = len(cata[3])/np.sum(cata[3]>1e10)
	fmw2 = len(cata1[3])/np.sum(cata1[3]>1e10)
	print('All to MW-like: {:.2e} ({:.2e})'.format(fmw1, fmw2))
	totxt(repo+'nsc_cap.txt', [base, his, his_])

	nx = 40
	x1, x2 = 0, 20
	lz = np.linspace(x1, x2, nx+1)
	plt.figure()
	plt.hist(cata[1], lz, alpha=0.5, label='All in NSCs')
	plt.hist(dmg[1][sel],lz, histtype='step', lw=1.5, label='Merged at $z>0$')
	plt.xlabel(r'$z_{\rm if}$')
	plt.ylabel(r'$dN/dz_{\rm if}\ [\rm a.u.]$')
	plt.xlim(x1, x2)
	plt.legend()
	plt.tight_layout()
	plt.savefig(repo+'bhbzif_dis.pdf')
	plt.close()
	
	zcut = 4.5
	print('Fraction of mergers with infalls at z>{}: {:.4e}'.format(zcut, np.sum(dmg[1][sel]>zcut)/np.sum(sel)))

	zed = np.linspace(x1, x2, nx+1)
	zbase = midbin(zed)
	dt = np.array([(TZ(zed[i])-TZ(zed[i+1]))/YR for i in range(nx)])
	his, ed = np.histogram(cata[1], zed, weights=cata[4])
	his1, ed = np.histogram(dmg[1][sel], zed, weights=dmg[4][sel])
	mrd = his/dt/ntree/Vcom #* 1e9
	mrd1 = his1/dt/ntree/Vcom #* 1e9
	y1, y2 = 1e-10, 1e-4
	plt.figure()
	plt.plot(zbase, mrd, label=r'All in NSCs')
	plt.plot(zbase, mrd1, '--', label=r'Merged at $z>0$')
	plt.plot(sfr3[0], sfr3[5]*fco*fbinary, 'k-.', label=r'Formation sites')
	#plt.fill_between(sfr3[0], sfr3[5]+3*sfr3[6], sfr3[5]-3*sfr3[6], fc='b', alpha=0.2)
	plt.legend(ncol=1)
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\dot{\rho}_{\bullet}\ [\rm M_{\odot}\ yr^{-1}\ Mpc^{-3}]$')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(repo+'ifrd_z.pdf')
	plt.close()

	#acb = 3e3*Rsun/AU
	acb = 1e4*Rsun/AU
	#bhbnbody = np.array(bhbnbody)
	la0 = cata[7] #bhbnbody[3]
	la = dmg[7][sel]
	lans1 = dmg[7][sel][(lmb<=3)*(lma>3)]
	lans2 = dmg[7][sel][(lmb<=3)*(lma<=3)]
	a1, a2 = 1, 1e6
	ab = np.geomspace(a1, a2, 61)
	y1, y2 = 1, 5e4*(ntree/100)
	plt.figure()
	plt.hist(la0, ab, alpha=0.3, label='All in NSCs')# (N-body sim.)')
	plt.hist(la, ab, histtype='step', lw=1.5, label='Merged at $z>0$')#, weights=np.ones(len(la))*len(la0)/len(la))
	#plt.hist(lans1, ab, histtype='step', lw=1.5, ls='--', color='k', label='NS-BH mergers')
	#plt.hist(lans2, ab, histtype='step', lw=1.5, ls='-.', color='r', label='NS-NS mergers')
	plt.plot([acb]*2,[y1,y2],'k-',lw=0.5)
	plt.xscale('log')
	plt.yscale('log')
	plt.legend(ncol=2)
	plt.xlim(a1, a2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$a_{0}\ [\rm au]$')
	plt.ylabel(r'$dN/d\loga_{0}\ [\rm a.u.]$')
	plt.tight_layout()
	plt.savefig(repo+'a0_dis.pdf')
	plt.close()
	print('Close binary fraction: {:.2e} ({:.2e})'.format(np.sum(la<acb)/len(la), np.sum(la0<acb)/len(la0)) )

	#exit()
	
	"""
	import seaborn as sns
	import pandas
	#sns.set_theme(style="darkgrid")
	xlab, ylab = r'$m_{\rm chirp}\ [\rm M_{\odot}]$', r'$q$'
	data = pandas.DataFrame(data=np.array([lmc,lmb/lma]).T,columns=[xlab,ylab])
	g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hist',xlim=[0,80], ylim=[0,1],bins=41,marginal_kws=dict(alpha=0.5,bins=41),color='k',cmap=plt.cm.Greys)
	#g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',xlim=[0,100], ylim=[0,100],gridsize=(41,41),bins='log',marginal_kws=dict(alpha=0.7),color='k',cmap=plt.cm.Greys)
	selgw = [7, 28, 29, 32, 35, 36, 46]
	g.ax_joint.errorbar(ligo[7],1.0/ligo[10],xerr=[ligo[9],ligo[8]],yerr=[ligo[11]/ligo[10]**2,ligo[12]/ligo[10]**2],fmt='.',
	alpha=0.3,color='r', label='3-OGC')
	g.ax_joint.errorbar(ligo[7][selgw],1.0/ligo[10][selgw],xerr=[ligo[9][selgw],ligo[8][selgw]],yerr=[ligo[11][selgw]/ligo[10][selgw]**2,ligo[12][selgw]/ligo[10][selgw]**2],fmt='^',color='r')
	g.ax_joint.legend()
	plt.savefig('mchirp_q.pdf')
	plt.close()
	
	xlab, ylab = r'$m_{1}\ [\rm M_{\odot}]$', r'$q$'
	data = pandas.DataFrame(data=np.array([lma,lmb/lma]).T,columns=[xlab,ylab])
	g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hist',xlim=[0,100], ylim=[0,1],bins=41,marginal_kws=dict(alpha=0.5,bins=41),color='k',cmap=plt.cm.Greys)
	#g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',xlim=[0,100], ylim=[0,100],gridsize=(41,41),bins='log',marginal_kws=dict(alpha=0.7),color='k',cmap=plt.cm.Greys)
	selgw = [7, 28, 29, 32, 35, 36, 46]
	g.ax_joint.errorbar(ligo[1],1.0/ligo[10],xerr=[ligo[3],ligo[2]],yerr=[ligo[11]/ligo[10]**2,ligo[12]/ligo[10]**2],fmt='.',
	alpha=0.3,color='r', label='3-OGC')
	g.ax_joint.errorbar(ligo[1][selgw],1.0/ligo[10][selgw],xerr=[ligo[3][selgw],ligo[2][selgw]],yerr=[ligo[11][selgw]/ligo[10][selgw]**2,ligo[12][selgw]/ligo[10][selgw]**2],fmt='^',color='r')
	g.ax_joint.legend()
	plt.savefig('m1_q.pdf')
	plt.close()
	
	#xlab, ylab = r'$m_{2}\ [\rm M_{\odot}]$', r'$q$'
	#data = pandas.DataFrame(data=np.array([lmb,lmb/lma]).T,columns=[xlab,ylab])
	#g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hist',xlim=[0,100], ylim=[0,1],bins=41,marginal_kws=dict(alpha=0.5,bins=41),color='k',cmap=plt.cm.Greys)
	#selgw = [7, 28, 29, 32, 35, 36, 46]
	#g.ax_joint.errorbar(ligo[4],1.0/ligo[10],xerr=[ligo[6],ligo[5]],yerr=[ligo[12],ligo[11]],fmt='.',
	#alpha=0.3,color='r', label='3-OGC')
	#g.ax_joint.errorbar(ligo[4][selgw],1.0/ligo[10][selgw],xerr=[ligo[6][selgw],ligo[5][selgw]],yerr=[ligo[12][selgw],ligo[11][selgw]],fmt='^',color='r')
	#g.ax_joint.legend()
	#plt.savefig('m2_q.pdf')
	#plt.close()

	xlab, ylab = r'$m_{1}\ [\rm M_{\odot}]$', r'$m_{2}\ [\rm M_{\odot}]$'
	data = pandas.DataFrame(data=np.array([lma,lmb]).T,columns=[xlab,ylab])
	g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hist',xlim=[0,100], ylim=[0,100],bins=41,marginal_kws=dict(alpha=0.5,bins=41),color='k',cmap=plt.cm.Greys)
	#g = sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',xlim=[0,100], ylim=[0,100],gridsize=(41,41),bins='log',marginal_kws=dict(alpha=0.7),color='k',cmap=plt.cm.Greys)
	selgw = [7, 28, 29, 32, 35, 36, 46]
	g.ax_joint.errorbar(ligo[1],ligo[4],xerr=[ligo[3],ligo[2]],yerr=[ligo[6],ligo[5]],fmt='.',
	alpha=0.3,color='r', label='3-OGC')
	g.ax_joint.errorbar(ligo[1][selgw],ligo[4][selgw],xerr=[ligo[3][selgw],ligo[2][selgw]],yerr=[ligo[6][selgw],ligo[5][selgw]],fmt='^',color='r')
	g.ax_joint.legend()
	plt.savefig('m2_m1.pdf')
	plt.close()

	selm = cata[3]<cata[2]
	cata[2][selm] = cata[3][selm]*0.99
	xlab, ylab = r'$\log(M_{\star}\ [\rm M_{\odot}])$', r'$\log(M_{\rm NSC}\ [\rm M_{\odot}])$'
	data = pandas.DataFrame(data=np.log10([cata[3],cata[2]]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(16,16),
	marginal_kws=dict(bins=16, fill=True,alpha=0.7),marginal_ticks=True,
	xlim=[5,10.5], ylim=[5,7.9],color='k',cmap=plt.cm.Greys)
	plt.savefig('mnsc_mg0.pdf')
	plt.close()

	selm = dmg[3]<dmg[2]
	dmg[2][selm] = dmg[3][selm]*0.99
	xlab, ylab = r'$\log(M_{\star}\ [\rm M_{\odot}])$', r'$\log(M_{\rm NSC}\ [\rm M_{\odot}])$'
	data = pandas.DataFrame(data=np.log10([dmg[3][sel],dmg[2][sel]]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(16,16),
	marginal_kws=dict(bins=16, fill=True,alpha=0.7),marginal_ticks=True,
	xlim=[5,10.5], ylim=[5,7.9],color='k',cmap=plt.cm.Greys)
	plt.savefig('mnsc_mg.pdf')
	plt.close()
	#print(np.sum(dmg[3]<dmg[2])/len(dmg[2]))
	
	selm = dmg1[3]<dmg1[2]
	dmg1[2][selm] = dmg1[3][selm]*0.99
	xlab, ylab = r'$\log(M_{\star}\ [\rm M_{\odot}])$', r'$\log(M_{\rm NSC}\ [\rm M_{\odot}])$'
	data = pandas.DataFrame(data=np.log10([dmg1[3][sel1],dmg1[2][sel1]]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(16,16),
	marginal_kws=dict(bins=16, fill=True,alpha=0.7),marginal_ticks=True,
	xlim=[5,10.5], ylim=[5,7.9],color='k',cmap=plt.cm.Greys)
	plt.savefig('mnsc_mg1.pdf')
	plt.close()
	
	xlab, ylab = r'$\log(M_{\rm NSC}\ [\rm M_{\odot}])$', r'$z_{\rm if}$'
	#data = pandas.DataFrame(data=np.array([np.log10(cata[2]),cata[1]]).T,columns=[xlab,ylab])
	data = pandas.DataFrame(data=np.array([np.log10(dmg[2]),dmg[1]]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(16,16),
	marginal_ticks=True,xlim=[5,7.9], ylim=[0,15],color='k',
	cmap=plt.cm.Greys,marginal_kws=dict(bins=16, fill=True,alpha=0.7))
	plt.savefig('mnscmg_zif.pdf')
	plt.close()
	data = pandas.DataFrame(data=np.array([np.log10(cata[2]),cata[1]]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(16,16),
	marginal_kws=dict(bins=16, fill=True,alpha=0.7),marginal_ticks=True,
	xlim=[5,7.9], ylim=[0,15],color='k',cmap=plt.cm.Greys)
	plt.savefig('mnsc_zif.pdf')
	plt.close()
	
	t0 = np.log10(TZ(0)/1e6/YR)
	t1, t2 = np.log10([TZ(z)/YR/1e6 for z in dmg[1][sel]]), np.log10([TZ(z)/YR/1e6 for z in dmg[-4][sel]])
	xlab, ylab = r'$\log(t_{\rm if}\ [\rm Myr])$', r'$\log(t_{\rm mg}\ [\rm Myr])$'
	data = pandas.DataFrame(data=np.array([t1, t2]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(33,33),
	marginal_kws=dict(bins=33, fill=True,alpha=0.7),
	xlim=[2.5,t0], ylim=[2.5,t0],color='k',cmap=plt.cm.Greys)
	#g.ax_joint.plot([0,16],[0,16],'r-')
	plt.savefig('tmg_tif.pdf')
	plt.close()
	
	xlab, ylab = r'$z_{\rm if}$', r'$z_{\rm mg}$'
	data = pandas.DataFrame(data=np.array([dmg[1][sel],dmg[-4][sel]]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hex',bins='log',gridsize=(33,33),
	marginal_kws=dict(bins=33, fill=True,alpha=0.7),
	xlim=[0,16], ylim=[0,16],color='k',cmap=plt.cm.Greys)
	#g.ax_joint.plot([0,16],[0,16],'r-')
	plt.savefig('zmg_zif.pdf')
	plt.close()

	xlab, ylab = r'$\log(a_{0}\ [\rm au])$', r'$e_{0}$'
	data = pandas.DataFrame(data=np.array([np.log10(la),le]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hist',xlim=[0,5], ylim=[0,1],bins=51,
	marginal_kws=dict(bins=51,alpha=0.5),color='k',cmap=plt.cm.Greys)
	plt.savefig('loga_e.pdf')
	plt.close()
	
	xlab, ylab = r'$q$', r'$e_{0}$'
	data = pandas.DataFrame(data=np.array([lq,le]).T,columns=[xlab,ylab])
	sns.jointplot(data=data,x=xlab,y=ylab,kind='hist',xlim=[0,1], ylim=[0,1],bins=51,
	marginal_kws=dict(bins=51,alpha=0.5),color='k',cmap=plt.cm.Greys)
	plt.savefig('q_e.pdf')
	plt.close()
	"""
	
	#exit()
	
	test = 1
	lab = 'test'
	t2, nt = 14e9, 14000
	lnL = 6.5
	cd = 100
	H = 20 #7.6
	kap = 0.1
	m1, m2 = 20, 2, 
	a0, e0 = 100, 0.1
	r0, M, R, e, g = 3, 1e5, 3, 0, 1.5
	fac = 1#sigmafac(g)
	if test>0:
		d = abhb_t(t2, nt, m1, m2, a0, e0, r0, M, R, e, g, H=H, cd=cd, kap=kap, lnL=lnL, wt=1, dehnen=1, fac=fac)
		#d0 = abhb_t(t2, 0, m1, m2, a0, e0, r0, M, R, e, g, H=H, cd=cd, kap=kap, lnL=lnL, wt=0, dehnen=1, fac=fac)
		#print(d['out'][1], d0['out'][1])
		plot_bhb(d, lab, aflag=0, y1=0, log=1, gyr=1, sep=1)

	t2, nt = 10e9, 100000
	H = 20
	kap = 0.1
	m1, m2 = 70, 50
	a0, e0 = 300, 0.9
	r0, M, R, e, g = 3, 1e6, 3, 0, 1.5
	fac = 1#sigmafac(g)
	
	if test>0:
		#g = 1.8
		cd = 100
		d = abhb_t(t2, nt, m1, m2, a0, e0, r0, M, R, e, g, H=H, cd=cd, kap=kap, lnL=lnL, wt=1, fac=fac)
		#d0 = abhb_t(t2, 0, m1, m2, a0, e0, r0, M, R, e, g, H=H, cd=cd, kap=kap, lnL=lnL, wt=0, dehnen=1, fac=fac)
		#print(d['out'][1], d0['out'][1])
		plot_bhb(d, 'ecc', aflag=0, y1=0, log=1, sep=1)
		exit() #THREE-DIMENSIONAL STELLAR KINEMATICS AT THE GALACTIC CENTER: MEASURING THE NUCLEAR STAR CLUSTER SPATIAL DENSITY PROFILE, BLACK HOLE MASS, AND DISTANCE
	
	t2, nt = 10e9, 100000
	m1, m2 = 85, 66
	cd = 1000
	a0, e0 = 200, 0
	r0, M, R, e, g = 3, 2.5e7, 4.4, 0, 1.8
	rep = 'kap01/'
	if not os.path.exists(rep):
		os.makedirs(rep)
	de0 = [0, 0.5, 0.9]
	de = [0, 0.5, 0.9]
	for i in range(3):
		for j in range(3):
			e0 = de0[j]
			e = de[i]
			print('ebhb0={}, eo0={}'.format(e0, e))
			lab = str(i)+str(j)
			d = abhb_t(t2, nt, m1, m2, a0, e0, r0, M, R, e, g, H=H, cd=cd, kap=kap, lnL=lnL, wt=1, fac=fac)
			if i==0 and j==0 or j==2:
				plot_bhb(d, lab, rep, 1, y1=0, log=1, sep=1)
				plot_bhb(d, lab, rep+'linear/', y1=0, log=1, sep=1)
			plot_bhb(d, lab, rep, 1)
			plot_bhb(d, lab, rep +'linear/', 0)








