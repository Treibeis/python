from pbh_hmf import *
from scipy.special import erfc

Hfrac = 0.76
XeH = 0.0
XeHe = 0.0
def mmw(xeH = XeH, xeHe = XeHe, X = Hfrac):
	xh = 4*X/(1+3*X)
	return 4.0/(1.0+3*X)/(xh*(1+xeH)+(1-xh)*(1+xeHe))

def M_Tvir(T, z = 10.0, delta = 200, xe=0, Om=Om, h=h):
	y = 2*T*BOL*(3/(4*np.pi*delta*rhom(1/(1+z),Om,h)))**(1/3)/GRA/(mmw(xe)*PROTON)
	return y**1.5/Msun

def Tvir(m = 1e10, z = 10.0, delta = 200, xeH=0, Om=Om, h=h):
	M = m*Msun
	Rvir = (M/(rhom(1/(1+z),Om,h)*delta)*3/4/np.pi)**(1/3)
	return GRA*M*mmw(xeH)*PROTON/Rvir/(2*BOL) # *3/5

def sfrbf(z):
	return 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)

def sfrtf(z, a, b, c, d):
	#t = 1/(z+1)
	#return a*(t**b*np.exp(-t/c)+d*np.exp(d*(t-1)/c)) 
	return a*(1+z)**b/(1+((1+z)/c)**d)

Mdown = lambda z: 3e6*((1+z)/10)**-1.5 # T_vi r> 10^3 K, https://doi.org/10.1007/978-3-642-32362-1_3
#Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5 # Trenti M., Stiavelli M., 2009, ApJ, 694, 879

smhm0 = [-1.435, 1.831, 1.368, -0.217, 12.035, 4.556, 4.417, -0.731, 1.963, -2.316] \
		+ [-1.732, 0.178, 0.482, -0.841, -0.471, 0.411, -1.034, -3.100, -1.055]
smhm1 = [-1.43, 1.796, 1.36, -0.216, 12.04, 4.675, 4.513, -0.744, 1.973, -2.353] \
		+ [-1.783, 0.186, 0.473, -0.884, -0.486, 0.407, -1.088, -3.241, -1.079]
		
def mjion(z, delta=125, Tb=2e4):
	return 6.7e8*((1+z)**3*delta/5**3/125)**-0.5*(Tb/2e4)**1.5

fb = Ob/Om
		
def ms_mh(mh, z0, e0, ea, elna, ez, m0, ma, mlna, mz, a0, aa, alna, az, b0, ba, bz, d0, g0, ga, gz, eta0=1e-3, fb=fb, zr=0, zcut=10):
	z = min(z0, zcut)
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
	ms0 = mh*eta0*fb*(z>zr)
	if z<=zr:
		ms *= (mh>mjion(z))
	#ms = ms*(ms>ms0) + ms0*(ms<=ms0)
	ms += ms0
	return ms

def ms_mh0(mh, eta0=1e-3, fb=fb):
	return fb*mh*(0.05/((mh/2.8e11)**0.49+(mh/2.8e11)**-0.61)+eta0)

def smd_z(z, ps, lm0, mmax=1e15, nm=100, gf=gf3, wf=wf3, Om = Om, Ol = 1-Om, dx = 1e-4, corr = corr, mode=1, h=h, Ob=Ob):
	#ms = Mdown(z) 
	ms = Mup(z)
	lm = np.geomspace(ms*h, mmax, nm+1)
	hmf = halomassfunc(lm0, z, ps, wf, gf, Om, Ol, dx, corr, mode)
	ln = np.array([hmf(x)*ms_mh(x/h, z, *smhm1) for x in lm])
	return np.trapz(ln, lm)*h**3

def sfrd_mod(z, ps, lm0, dz=0.1, mmax=1e15, nm=30, gf=gf3, wf=wf3, Om = Om, Ol = 1-Om, dx = 1e-4, corr = corr, mode=1, h=h, Ob=Ob, zcut=100, zr=6, eta0=1e-3):
	#if z>zcut:
	#	ms = Mdown(z) 
	#elif z>zr:
	#	ms = Mup(z)
	#else:
	#	ms = mjion(z)
	ms = Mdown(z)
	#ms = Mup(z)
	lm = np.geomspace(ms*h, mmax, nm+1)
	hmf = halomassfunc(lm0, z, ps, wf, gf, Om, Ol, dx, corr, mode)
	hmf1 = halomassfunc(lm0, z+dz, ps, wf, gf, Om, Ol, dx, corr, mode)
	if z<zcut:
		ln = np.array([(hmf(x)-hmf1(x))*ms_mh(x/h, z, *smhm0, zr=zr, eta0=eta0) for x in lm])
	else:
		ln = np.array([(hmf(x)-hmf1(x))*ms_mh0(x/h, eta0) for x in lm])
	ln[ln<0] = 0
	dt = (age_a(1/(1+z))-age_a(1/(1+z+dz)))/YR
	return np.trapz(ln, lm)*h**3/dt

def fcol_z_hmf(lmh, z, ps, lm0, gf=gf3, wf=wf3, Om = Om, Ol = 1-Om, dx = 1e-4, corr = corr, mode=1, h=h, Ob=Ob, ref=0, norm = 0.5, model='sheth99', mdef='fof'):
	rhoref = rhom(1, Om, h)/Msun*MPC**3
	if ref>0:
		lrho = np.array([ngem0(x*h, z, mass=1, mdef=mdef, model=model) for x in lmh])*h**2
	else:
		lrho = np.array([ngem(x*h, z, ps, lm0, wf=wf, gf=gf, mode=mode, mass=1) for x in lmh])*h**2*norm
	#lrho = msdens(lms, z, 1, ps, lm0, gf, wf, Om, Ol, dx, corr, mode, h, Ob)/fb
	return lrho/rhoref

def fcol_z_raw(lmh, z, ps, wf=wf3, gf=gf3, Om = Om, Ol=1-Om, norm=1.0, h=h):
	lsig = np.array([sigma2_M(x*h, ps, wf, gf, Om, norm)**0.5 for x in lmh])
	dc = deltac_z(z, Om, Ol)
	return 0.5*erfc(dc/2**0.5/lsig)

def fcol_seed(z, mpbh, fpbh, seed=1):
	return fpbh*mbind(z, mpbh, fpbh, seed=seed)/mpbh

def fromseed(mpbh, ng=1e-5):
	return ng/(rhom(1, Om, h)*(Om-Ob)/Om/mpbh/Msun*MPC**3)

def eps_mpbh(mpbh, z=10, fb=fb, mg=1e11):
	mb = mbind(z, mpbh, 1e-10, seed=1)
	return mg/(fb*mb)

def tolog10(x, nd=0):
	logx = int(np.log10(x))
	if logx<0 and logx>np.log10(x):
		logx = logx-1
	if nd==0:
		num = int(x/10**logx)
	else:
		num =  int(x/10**(logx-nd))/10**nd
	if num==1:
		return r'10^{'+str(logx)+'}'
	else:
		return '{}'.format(num)+r'\times 10^{'+str(logx)+'}'

redu = 1

# 1st version obs. data:
#z = 10
#ng = 2e-5
#mg = 1e11

z = 7.5
mg = 1e11
ng = 1e-5
if redu>0:
	z = 8.5
	mg = 1e10
	ng = 5e5/1e10
	#mg = 1e11/10**1.6
	#ng = 2e-5*10
eps0 = 1

m1, m2, nm = 1e6, 1e11, 1001
fcrit = (1+z)*aeq
lm = np.geomspace(m1, m2, nm)
lfc = np.ones(nm)*fcrit
y1, y2 = 1e-8, 1
leps = eps_mpbh(lm, mg=mg, z=z)
lf = fromseed(lm, ng)
md = np.min(lm[leps<eps0])
mu = 1e11
sel = (lm>md)*(lm<mu)
lmf = lf*lm
print('Seed effect: m_pbh * f_pbh > {:.2e} Msun, m_pbh > {:.2e} Msun'.format(np.min(lmf[sel]), md))

lab = r'$M_{\star}='+tolog10(mg)+r'\ \rm M_{\odot}$'+'\n' \
	+r'$n_{\rm g}='+tolog10(ng)+r'\ \rm Mpc^{-3}$'+'\n' \
	+'$z={}$'.format(z)

plt.figure()
ax1 = plt.subplot(111)
plt.loglog(lm, lf, label=r'$f_{\rm PBH}$')
plt.plot(lm, leps, '--', label=r'$\epsilon$')
plt.plot([md]*2,[y1,y2], 'k-.', label='$\epsilon=0.1,\ 1$')
plt.plot([md*10]*2,[y1,y2], 'k-.')
plt.plot([m1, m2], [fcrit]*2, ':', label=r'$(1+z)a_{\rm eq}$')
plt.fill_between(lm[sel], lf[sel], lfc[sel], fc='gray', alpha=0.5)
#plt.text(m1*2, y2/5, lab)
plt.text(m1*1.5, y2/100, lab)
plt.xlabel(r'$m_{\rm PBH}$')
plt.ylabel(r'$f_{\rm PBH}$ or $\epsilon$')
plt.xlim(m1, m2)
plt.ylim(y1, y2)
ax1.set_yticks(np.geomspace(y1, y2, 9))
plt.legend(ncol=1, loc=6)
plt.tight_layout()
if redu==0:
	plt.savefig('fpbh_mpbh_seed_z'+str(z)+'.pdf')
	#plt.savefig('fpbh_mpbh_seed.png', dpi=300)
else:
	plt.savefig('fpbh_mpbh_seed1_z'+str(z)+'.pdf')
plt.close()

exit()

if __name__=="__main__":
	mode = 0
	mdef, model = 'fof', 'press74'
	#mdef, model = 'fof', 'sheth99'
	#mdef, model = 'vir', 'seppi20'
	gf, wf = gf3, wf3
	mfac0 = 1
	iso = 1
	seed = 0
	dmax = 0
	read = 1
	cut = 1
	ppoi = 1
	
	repo = './'
	if seed>0:
		repo = 'seed/'
	if cut>0:
		repo = 'cut/'
		ppoi = 0
		#repo = 'cut_mode1/'

	if (not os.path.exists(repo)):
		os.mkdir(repo)
	
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
	
	md1 = [3e5, 3e-4]
	#md1 = [3e5, 1e-2]
	md2 = [1e9, 1e-5]
	md3 = [1e10, 1e-4]
	md4 = [1e11, 1e-5]
	lmd0 = np.array([md1, md2, md3])#, md4])
	lmd = lmd0.T
	#llab = [r'$\Lambda$CDM', r'M1, $m_{\rm PBH}f_{\rm PBH}=90\ \rm M_{\odot}$', r'M2, $m_{\rm PBH}f_{\rm PBH}=10^{4}\ \rm M_{\odot}$', r'M3, $m_{\rm PBH}f_{\rm PBH}=10^{6}\ \rm M_{\odot}$']
	llab = [r'$\Lambda$CDM', r'M1', 'M2', 'M3']
	
	lps = [ps0] + [PBH(ps0, md[0], md[1], aeq, h, mfac=mfac0, iso=iso, seed=seed, dmax=dmax, out=1, cut=0) for md in lmd0]
	if cut>0:
		lps_ = [ps0] + [PBH(ps0, md[0], md[1], aeq, h, mfac=mfac0, iso=iso, seed=seed, dmax=dmax, out=1, cut=1) for md in lmd0]
	
	ztest = 10
	mh = 1e10
	test0 = ngem(mh*h, ztest, ps0, lm0, gf=gf, wf=wf, mode=1)*h**3
	test1 = ngem0(mh*h, ztest, mdef=mdef, model=model)*h**3
	test3 = ngem(mh*h, ztest, lps[-1], lm0, gf=gf, wf=wf, mode=1)*h**3
	print('Number density: {:.2e} ({:.2e})/{:.2e} Mpc^-3 (CDM/PBH)'.format(test0, test1, test3))
	
	lT = np.array([1e4, 1e5, 1e6, 1e7])
	tlab = [r'$T_{\rm vir}=10^{4}\ \rm K$', r'$T_{\rm vir}=10^{5}\ \rm K$', r'$T_{\rm vir}=10^{6}\ \rm K$', r'$T_{\rm vir}=10^{7}\ \rm K$']

	llc = ['k', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
	
	z1, z2 = 0, 35
	lz = np.linspace(z1, z2, 100)
	#lf0 = np.array([fcol_z_raw(M_Tvir(lT, z), z, ps0, wf, gf) for z in lz])
	#lf0 = lf0.T
	df = []
	for i in range(len(lps)):
		if cut==0:
			ps = lps[i]
		else:
			ps = lps_[i]
		lf1 = np.array([fcol_z_raw(M_Tvir(lT, z), z, ps, wf, gf) for z in lz])
		df.append(lf1.T)
	lf0 = df[0]
	y1, y2 = 1e-5, 1e3
	plt.figure()
	ax1 = plt.subplot(111)
	[plt.plot(lz, lf0[j], color=llc[j], label=tlab[j]) for j in range(len(lT))]
	#plt.plot(lz, lf0[0], color=llc[0], label=r'$\Lambda$CDM')
	plt.plot([], [], color='k', ls=lls[0], lw=4.5, alpha=0.3, label=r'$f_{\rm col,seed}$')
	for i in range(len(lps)):
		if i>0:
			plt.plot(lz, fcol_seed(lz, *lmd0[i-1]), color='k', ls=lls[i], lw=4.5, alpha=0.3)
		for j in range(len(lT)):
			if j==0:
				plt.plot(lz, df[i][j], color=llc[j], ls=lls[i], label=llab[i])
			else:
				plt.plot(lz, df[i][j], color=llc[j], ls=lls[i])
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm col}(>T_{\rm vir},z)$')
	plt.xlim(z1, z2)
	plt.ylim(y1, y2)
	plt.yscale('log')
	ax1.set_yticks(np.geomspace(y1, 1e3, 9))
	plt.legend(ncol=2)
	plt.tight_layout()
	plt.savefig(repo+'fcol_z.pdf')
	plt.close()
	
	lmh = [1e6, 1e8, 1e10, 1e12]
	mlab = [r'$M_{\rm halo}=10^{6}\ \rm M_{\odot}$', r'$M_{\rm halo}=10^{8}\ \rm M_{\odot}$', r'$M_{\rm halo}=10^{10}\ \rm M_{\odot}$', r'$M_{\rm halo}=10^{12}\ \rm M_{\odot}$']
	z1, z2 = 0, 35
	lz = np.linspace(z1, z2, 100)
	#lf0 = np.array([fcol_z_raw(lmh, z, ps0, wf, gf) for z in lz])
	#lf0 = lf0.T
	df = []
	for i in range(len(lps)):
		if cut>0:
			ps = lps_[i]
		else:
			ps = lps[i]
		lf1 = np.array([fcol_z_raw(lmh, z, ps, wf, gf) for z in lz])
		df.append(lf1.T)
	lf0 = df[0]
	y1, y2 = 1e-5, 3e2
	plt.figure()
	ax1 = plt.subplot(111)
	[plt.plot(lz, lf0[j], color=llc[j], label=mlab[j]) for j in range(len(lmh))]
	#plt.plot(lz, lf0[0], color=llc[0], label=r'$\Lambda$CDM')
	plt.plot([], [], color='k', ls=lls[0], lw=4.5, alpha=0.3, label=r'$f_{\rm col,seed}$')
	for i in range(len(lps)):
		if i>0:
			plt.plot(lz, fcol_seed(lz, *lmd0[i-1]), color='k', ls=lls[i], lw=4.5, alpha=0.3)
		for j in range(len(lT)):
			if j==0:
				plt.plot(lz, df[i][j], color=llc[j], ls=lls[i], label=llab[i])
			else:
				plt.plot(lz, df[i][j], color=llc[j], ls=lls[i])
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm col}(>M_{\rm halo},z)$')
	plt.xlim(z1, z2)
	plt.ylim(y1, y2)
	plt.yscale('log')
	ax1.set_yticks(np.geomspace(y1, 1e2, 8))
	plt.legend(ncol=2)
	plt.tight_layout()
	plt.savefig(repo+'fcol_z0.pdf')
	plt.close()
	
	#exit()
	
	m10, m20 = 1e6, 1e14
	m1, m2, nm = 1e4, 1e16, 301
	lmh = np.geomspace(m1, m2, nm)
	dsig = []
	y1, y2 = 1, 1e6
	plt.figure()
	ax1 = plt.subplot(111)
	dsig = []
	for i in range(len(lps)):
		if cut>0:# and i>0:
			ps = lps_[i]
			#lsig = [sigma2_M(m*h, ps, wf = wf, gf = gf)**0.5 for m in lmh]
			#plt.loglog(lmh, lsig, color=llc[i], ls=lls[i])#, label=llab[i])
			#dsig.append(lsig)
		else:
			ps = lps[i]
		lsig = [sigma2_M(m*h, ps, wf = wf, gf = gf)**0.5 for m in lmh]
		dsig.append(lsig)
		plt.loglog(lmh, lsig, color=llc[i], ls=lls[i], label=llab[i])
	deltac_eq = deltac_z(1/aeq-1)
	plt.plot([m1,m2],[deltac0]*2, 'k', lw=4.5, alpha=0.3, label=r'$\delta_{c}(z=0)$')
	plt.plot([m1,m2],[deltac_z(10)]*2, 'k--', lw=4.5, alpha=0.3, label=r'$\delta_{c}(z=10)$')
	plt.plot([m1,m2],[deltac_z(101)]*2, 'k-.', lw=4.5, alpha=0.3, label=r'$\delta_{c}(z=100)$')
	plt.plot([m1,m2],[deltac_eq]*2, 'k:', lw=4.5, alpha=0.3, label=r'$\delta_{c}(z=z_{\rm eq})$')
	plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$\sigma(M_{\rm halo})$')
	plt.xlim(m10, m20)
	plt.ylim(y1, y2)
	ax1.set_xticks(np.geomspace(m10, m20, 9))
	#plt.yscale('log')
	#ax1.set_yticks(np.geomspace(y1, 1e2, 8))
	plt.legend(ncol=2)
	plt.tight_layout()
	plt.savefig(repo+'sigma_m.pdf')
	plt.close()
	#1019607
	nu1, nu2  = 1, 4
	z1, z2 = 0, 100
	lz = np.linspace(z1, z2, 100)
	dmcrit1 = []
	dmcrit2 = []
	for z in lz:
		deltac = deltac_z(z)
		lmcrit = [np.min(lmh[x<deltac/nu1]) for x in dsig]
		dmcrit1.append(lmcrit)
		lmcrit = [np.min(lmh[x<deltac/nu2]) for x in dsig]
		dmcrit2.append(lmcrit)
	dmcrit1 = np.array(dmcrit1).T
	dmcrit2 = np.array(dmcrit2).T
	plt.figure()
	for i in range(len(lps)):
		plt.plot(lz, dmcrit1[i], color=llc[i], ls=lls[i], label=llab[i])
		plt.plot(lz, dmcrit2[i], color=llc[i], ls=lls[i])
		if i>0:
			#plt.plot(lz, mbind(lz, *lmd0[i-1]), color=llc[i], ls=lls[i], lw=4.5, alpha=0.3)
			plt.plot(lz, mbind(lz, *lmd0[i-1], seed=1), color=llc[i], ls=lls[i], lw=4.5, alpha=0.3)
			if ppoi>0:
				plt.plot(lz, mbind(lz, *lmd0[i-1], seed=0), color=llc[i], ls=lls[i], lw=4.5, alpha=0.3)
			#plt.plot([z1, z2], [lmd0[i-1][0]]*2, color=llc[i], ls=lls[i], lw=4.5, alpha=0.3)
	plt.plot([10]*2, [m1,m2], 'k-', lw=0.5)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$M_{\rm crit}'+r'(\nu\sim{}-{})'.format(nu1,nu2)+r'\ [\rm M_{\odot}]$')
	plt.xlim(z1, z2)
	plt.ylim(m1, m2)
	plt.yscale('log')
	plt.legend(ncol=1)
	plt.tight_layout()
	plt.savefig(repo+'mcrit_z.pdf')
	plt.close()
	
	#ps = lps[-1]
	eta0 = 1e-3
	zcut = 10
	
	gf, wf = gf3, wf3
	z1, z2, nz = 0, 35, 36
	lz = np.geomspace(z1+1, z2+1, nz)-1
	lz0 = np.geomspace(1, zcut+1, 100)-1
	#lt = np.array([age_a(1/(1+z)) for z in lz])
	#ldt = (lt[:-1]-lt[1:])/YR
	#lsmd = np.array([smd_z(z, ps, lm0, gf=gf, wf=wf, mode=mode) for z in lz])
	
	if read==0:
		dsfr = []
		if cut>0:
			dsfr_ = []
		for i in range(len(lps)):
			ps = lps[i]
			lsfrd = np.array([sfrd_mod(z, ps, lm0, gf=gf, wf=wf, mode=mode, eta0=eta0) for z in lz])
			dsfr.append(lsfrd)
			if cut>0:
				ps = lps_[i]
				lsfrd = np.array([sfrd_mod(z, ps, lm0, gf=gf, wf=wf, mode=mode, eta0=eta0) for z in lz])
				dsfr_.append(lsfrd)
		totxt(repo+'sfrd.txt', [lz]+dsfr)
		if cut>0:
			totxt(repo+'sfrd_cut.txt', [lz]+dsfr_)
	else:
		data = np.array(retxt(repo+'sfrd.txt',len(lps)+1))
		lz = data[0]
		dsfr = data[1:]
		if cut>0:
			data = np.array(retxt(repo+'sfrd_cut.txt',len(lps)+1))
			dsfr_ = data[1:]
	#(lsmd[:-1]-lsmd[1:])/ldt
	plt.figure()
	for i in range(len(lps)):
		if cut>0:
			lsfrd = dsfr_[i]
		else:
			lsfrd = dsfr[i]
		plt.loglog(lz+1, lsfrd, label=llab[i], ls=lls[i], color=llc[i])
		#if cut>0 and i>0:
		#	plt.loglog(lz+1, dsfr_[i], ls=lls[i], color=llc[i], lw=4.5, alpha=0.3)
	lsfrd_ = sfrbf(lz0)
	plt.plot(lz0+1, lsfrd_, color='r', ls=(0,(10,5)), label='Madau+2014')
	plt.fill_between(lz0+1, lsfrd_*10**0.5, lsfrd_/10**0.5, facecolor='r', alpha=0.3)
	plt.legend()
	plt.xlabel(r'$1+z$')
	plt.ylabel(r'$\rm SFRD\ [\rm M_{\odot}\ yr^{-1}\ Mpc^{-3}]$')
	plt.xlim(1, z2+1)
	plt.ylim(1e-5, 1)
	plt.tight_layout()
	plt.savefig(repo+'sfrd_z.pdf')
	plt.close()
	#exit()
	
	ps = ps0# lps[-3]
	
	z = 10
	lmh = M_Tvir(lT, z)
	lmh0 = M_Tvir(lT, 0)
	
	print('Ratio (M3):', fcol_z_raw(lmh0, 0, lps[-1], wf, gf)/fcol_z_raw(lmh0, 0, ps0, wf, gf))
	print('Ratio (M2):', fcol_z_raw(lmh0, 0, lps[-2], wf, gf)/fcol_z_raw(lmh0, 0, ps0, wf, gf))
	
	#lmh = np.array([1e8, 1e10, 1e12])
	#print(Tvir(lmh))
	print('z={}:'.format(z), lmh)
	print('z=0:', lmh0)
	print(fcol_z_raw(lmh, z, ps, wf, gf))
	print(fcol_z_hmf(lmh, z, ps, lm0, gf, wf, mode=mode)/2)
	print(fcol_z_hmf(lmh, z, ps, lm0, gf, wf, mode=mode, ref=1, mdef=mdef, model=model)/2)
	
	smd0 = smd_z(z, ps0, lm0, gf=gf, wf=wf, mode=mode)
	lsmd = np.array([smd_z(z, ps, lm0, gf=gf, wf=wf, mode=mode) for ps in lps])
	print('z={}:'.format(z), lsmd/smd0, smd0)
	smd0 = smd_z(0, ps0, lm0, gf=gf, wf=wf, mode=mode)
	lsmd = np.array([smd_z(0, ps, lm0, gf=gf, wf=wf, mode=mode) for ps in lps])
	print('z=0:', lsmd/smd0, smd0)
	
	#exit()
	
	
	
