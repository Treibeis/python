from cosmology import *
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('test2')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

geff = 1.09
nad = 1e19
rho0 = nad/1e19

t0 = rho0**-0.5
r0 = rho0**(geff/2-1)
m0 = rho0**(1.5*geff-2)

nu1500 = SPEEDOFLIGHT/(1.5e3*1e-8)

def UVmag(L, R, z, nu = nu1500):
	l = 10**L*Lsun
	r = 10**R*Rsun
	Teff = (l/(4*np.pi*r**2)/STEFAN)**0.25
	print('logTeff = {:.2f}'.format(np.log10(Teff)))
	dL = DZ(z)*(1+z)
	fnu = BB_spectrum(nu, Teff)*np.pi*(1+z)*(r/dL)**2/1e-23
	mag = -2.5*np.log10(fnu)+8.90
	return mag

def tff(rho):
	return (3*np.pi/(32*GRA*rho))**0.5/YR

def tdyn(M, R):
	r = R*AU
	m = M*Msun
	return (r**3/(GRA*m))**0.5/YR

def trelax_(N, M, R, gamma = 0.11, fac = 0.138):
	N_ = gamma*N
	return fac*N/np.log(N_) * tdyn(M, R)

def trelax(N, M, R):
	y1 = 0.138*N/np.log(0.11*N) * tdyn(M, R)
	y2 = 3.33392239*N/np.log(N) * tdyn(M, R)
	return y1 * (N>10) + y2 * (N<=10)

"""
lN = np.geomspace(2, 1000.0, 100)
lt1 = trelax(lN, 1, 1)/tdyn(1, 1)
lt2 = trelax_(lN, 1, 1, 1, 0.1)/tdyn(1,1)
lt3 = trelax_(lN, 1, 1)/tdyn(1, 1)
plt.figure()
plt.plot(lN, lt1)
plt.plot(lN, lt2, '--')
plt.plot(lN, lt3, '-.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$N$')
plt.ylabel(r'$t_{\rm relax}/t_{\rm dyn}$')
plt.tight_layout()
plt.savefig('trelax_N.pdf')
plt.close()
"""

def tdecay(M, R):
	return 14*R**1.5*M**-0.5

def t3b(sig, rho, M, a, m = 1):
	n = rho/(m*Msun)
	return sig/(2*np.pi*GRA*M*Msun*n*a)/YR
	
def tdf(sig, rho, M, lnl = 5):
	return 3/(4*(2*np.pi)**0.5*GRA**2*lnl) * sig**3/(M*Msun*rho)/YR

def N_t(t):
	return 3*(t/t0)**0.3
	
def M_t(t):
	return 400*(t/1e5/t0)**(4-3*geff) * m0
	
def R_t(t):
	return 0.85 * (t/t0)**(2-geff) * r0
	
def Mbar(tf, ta):
	return M_t(ta)/N_t(tf)
	
def Mmax(tf, ta, alp, Mmin = 1, mode=0):
	Mb = Mbar(tf, ta)
	if alp!=-1 and alp!=-2:
		def f(m):
			out = (alp+1)/(alp+2)
			out *= m**(alp+2)-Mmin**(alp+2)
			out /= m**(alp+1)-Mmin**(alp+1)
			return out - Mb
	elif alp==-1:
		def f(m):
			out = (m - Mmin)/(np.log(m/Mmin))
			return out - Mb
	else:
		def f(m):
			out = np.log(m/Mmin)/(1/Mmin - 1/m)
			return out - Mb
	sol = root(f, 10*Mmin)
	if mode>0:
		Mm = min(sol.x[0], M_t(ta))
	else:
		Mm = sol.x[0]
	if alp!=-2:
		A = M_t(ta)*(alp+2)/(Mm**(alp+2)-Mmin**(alp+2))
	else:
		A = M_t(ta)/np.log(Mm/Mmin)
	return Mm, A
	
def IMF(tf, ta, alp, Mmin = 1, nb = 100):
	Mm, A = Mmax(tf, ta, alp, Mmin)
	lm = np.geomspace(Mmin, Mm, nb)
	phi = A*lm**alp
	#print('Total mass: ', np.trapz(phi*lm, lm), M_t(ta))
	return lm, phi, N_t(tf), M_t(ta)
	
def plotIMF(param, i, j, nb = 100):
	lm, phi, Ns, Mtot = IMF(*param, nb)
	llab = []
	labtf = r'$t_{\rm frag}='+str(param[0]/1e3)+'\ \mathrm{kyr}$, '
	labtf += r'$N_{\star}='+'{:.0f}'.format(Ns)+'$'
	llab.append(labtf)
	labta = r'$t_{\rm acc}='+str(param[1]/1e3)+'\ \mathrm{kyr}$, '
	labta += r'$M={:.0f}'.format(Mtot)+'\ \mathrm{M_{\odot}}$'
	llab.append(labta)
	llab.append(r'$\alpha='+str(-param[2])+'$')
	llab.append(r'$M_{\min}='+str(param[3])+'\ \mathrm{M_{\odot}}$')
	if j>=0:
		lab = llab[j]
	else:
		lab = 'Pop II/I'
	plt.loglog(lm, phi*lm, label=lab, ls=lls[i], lw=llw[i])

def popIII_IMF(m, Mcut = 20, alpha=0.17):
	return m**-alpha * np.exp(-Mcut/m**2)

def popII_IMF(m, mc=0.18, m0=2.0, x=1.35, sig=0.579):
	ot = m0**-x * np.exp(-(np.log(m/mc)**2*0.5/sig**2)) * (m<=m0)
	ot += m**-x * (m>m0)
	return ot/m

mcut3 = [0, 8, 40, 140, 150]
mcut2 = [11, 100, 27, 85]

def SP_info(m1 = 1, m2 = 150, imf = popIII_IMF, mcut=mcut3, nb = 10000, mode=1):
	if mode==0:
		lm = np.linspace(m1, m2, nb+1)
	else:
		lm = np.geomspace(m1, m2, nb+1)
	Mtot = np.trapz(imf(lm)*lm, lm)
	Ntot = np.trapz(imf(lm), lm)
	d = {}
	lmf, lnf = [], []
	for i in range(len(mcut)-1):
		x, y = mcut[i], mcut[i+1]
		sel = (lm>x) * (lm<=y)
		lm_ = lm[sel]
		lmf.append(np.trapz(imf(lm_)*lm_, lm_))
		lnf.append(np.trapz(imf(lm_), lm_))
	d['mass'] = np.array(lmf)/Mtot
	d['num'] = np.array(lnf)/Ntot
	d['N/M'] = Ntot/Mtot
	return d

def plotref(l, i, fmt, c, lab = None, fs = 'full', mode = 0, mf = 1, zo = 1):
	d = l['raw']
	if len(d[0])>1 and mode==0:
		dd = l['mean']
		if mf>0:
			a = plt.errorbar(dd[0][0], dd[i][0], xerr=dd[0][1], yerr=dd[i][1], 
				label=lab, fmt=fmt, fillstyle=fs, color=c, zorder=zo)
			plt.scatter(d[0], d[i], marker='.', color=c, alpha=0.5, zorder=1)
		else:
			if fs is 'none':
				a = plt.scatter(d[0], d[i], marker=fmt, label=lab, fc=fs, ec=c, lw=1, zorder=zo)
			else:
				a = plt.scatter(d[0], d[i], marker=fmt, label=lab, fc=c, ec=c, lw=1, zorder=zo)
	else:
		if len(d[i])>1:
			a = plt.errorbar(np.average(d[0]), d[i][0], yerr=d[i][1], fmt=fmt, label=lab, fillstyle=fs, color=c, zorder=zo)
		else:
			if fs is 'none':
				a = plt.scatter(d[0], d[i], marker=fmt, label=lab, fc=fs, ec=c, lw=1, zorder=zo)
			else:
				a = plt.scatter(d[0], d[i], marker=fmt, label=lab, fc=c, ec=c, lw=1, zorder=zo)
	return a

def refdata(d, mode=0):
	l = {}
	if mode==0:
		d.append(np.array(d[2])/np.array(d[1]))
	else:
		d.append([d[2][0]/d[1][0], (d[2][1])/(d[1][0])])
	l['raw'] = d
	if len(d[0])>1 and mode==0:
		dd = []
		for x0 in d:
			x = [y for y in x0 if y>0]
			if len(x)>1:
				dd.append([np.average(x), np.std(x)])
			else:
				dd.append([x[0], x[0]])
		l['mean'] = dd
	else:
		l['mean'] = []
	return l
	
lls = ['-', '--', '-.', ':']*3
llw = [1]*4 + [2]*4 + [3]*4
oi = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']

if __name__=="__main__":
	# time, number, mass, size
	GT12 = refdata([[8.31, 11.38, 9.6, 8.94], [2, 4, 4, 5], [0.15, 0.7, 0.7, 1], [3, 5, 5, 3]])
	mn = 1e6**(1.5*geff-2)
	rn = 1e6**(0.5*geff-1)
	SA13 = refdata([[4.8, 4.4, 4.3, 4.4, 4.5, 4.75, 4.9, 4.55, 4.45, 4.5], [4, 3, 9, 9, 9, 7, 6, 10, 6, 24], np.array([55, 32, 50, 70, 50, 35, 30, 50, 40, 120])*mn, [3e3*rn, 2e3*rn]])
	SA13_ = refdata([np.array([4.8, 4.4, 4.3, 4.4, 4.5, 4.75, 4.9, 4.55, 4.45, 4.5])*1e3, [4, 3, 9, 9, 9, 7, 6, 10, 6, 24], np.array([55, 32, 50, 70, 50, 35, 30, 50, 40, 120]), [3e3, 2e3]])
	ndis = np.array([[1, 2, 3, 4, 5, 6], [21, 11, 14, 8, 1, 4], [125, 55, 60, 50, 30, 70]])
	mn = (1e19/3e13)**(1.5*geff-2)
	rn = (1e19/3e13)**(0.5*geff-1)
	Nb = np.sum(ndis[0]*ndis[1])/np.sum(ndis[1])
	ta = 1e5
	SH14 = refdata([[ta*3e-6**0.5], [Nb, [[Nb-1], [6-Nb]]], [np.sum(ndis[1]*ndis[2]*ndis[0])/np.sum(ndis[1])*mn, np.std(ndis[0]*ndis[2])*mn], [4e4*rn, [[3.5e4*rn], [6e4*rn]]]], 1)
	SH14_ = refdata([[ta], [np.sum(ndis[0]*ndis[1])/np.sum(ndis[1]), [6-Nb]], [np.sum(ndis[1]*ndis[2]*ndis[0])/np.sum(ndis[1]), np.std(ndis[0]*ndis[2])], [4e4, [[3.5e4], [6e4]]]], 1)
	#print(ndis[1]*ndis[2]*ndis[0])
	mn = 1e3**(1.5*geff-2)
	rn = 1e3**(0.5*geff-1)
	SA16 = refdata([np.array([1e3, 2e3, 4.7e3])/1e3**0.5, np.array([25, 35, 45]), np.array([15, 54, 77])*mn, np.array([1500, 1000, 4000])*rn])
	SA16_ = refdata([np.array([1e3, 2e3, 4.7e3]), np.array([25, 35, 45]), np.array([15, 54, 77]), np.array([1500, 1000, 4000])])
	ta = 1e5+1e4
	tf = 1e4
	rat = 1e7
	rat_ = (tf/ta)**2
	mn = (rat/rat_)**(1.5*geff-2)
	rn = (rat/rat_)**(0.5*geff-1)
	rf = PC/AU #R_t(ta)
	SD20 = refdata([[tf/rat**0.5], [4,3], [3e2*mn, 2e2*mn], [rf*rn, 0.9*rf*rn]], 1)
	SD20_ = refdata([[ta], [4,3], [3e2, 2e2], [rf, 0.9*rf]], 1)
	#print(rf*AU/PC, rf*(tf/ta)**(2-geff)*AU/PC, 3e2*(tf/ta)**(4-3*geff))

	rat = 1e19/4e17
	mn = (rat)**(1.5*geff-2)
	rn = (rat)**(0.5*geff-1)
	ln = [rat**-0.5, 1, mn, rn]
	MM15_ = refdata([[104, 114, 114, 114], [17, 9, 13, 7], [6.5, 4, 3, 2], [30, 20, 25, 10]])
	MM15 = refdata([np.array(MM15_['raw'][i])*ln[i] for i in range(4)])

	rat = 1e19/np.array([1e15, 1e12, 1e10])
	mn = (rat)**(1.5*geff-2)
	rn = (rat)**(0.5*geff-1)
	ln = [rat**-0.5, 1, mn, rn]
	HS17_ = refdata([[4e2, 7e3, 7e4], [2, 3, 2], [4.7, 4.7*1e3**(2-1.5*geff), 380], [9*6.2, 7*140, 12*1300]])
	#HS17_ = refdata([[4e2, 7e3, 7e4], [2, 3, 2], [4.7, np.nan, 330], np.array([9.5*6.2, 14*140, 14*1300])])
	dhs17 = [np.array(HS17_['raw'][i])*ln[i] for i in range(4)]
	#dhs17[2] = np.array([4.7, 2.3, 0.5])*mn[0]
	HS17 = refdata(dhs17)
	HS17_ = refdata([[4e2, 7e3, 14e4], [2, 3, 2], [4.7, 4.7*1e3**(2-1.5*geff), 380], [9*6.2, 7*140, 12*1300]])
	#print(HS17_)
		
	rat = 1e19/2e11
	mn = (rat)**(1.5*geff-2)
	rn = (rat)**(0.5*geff-1)
	ln = [rat**-0.5, 1, mn, rn]
	HT20_ = refdata([[1e5-1e4], [4], [145], [2e4+8e2]])
	HT20 = refdata([np.array(HT20_['raw'][i])*ln[i] for i in range(4)])
	ci = 2
		
	rat = 1e19/1e12
	rat = 1e19/2e11
	mn = (rat)**(1.5*geff-2)
	rn = (rat)**(0.5*geff-1)
	ln = [rat**-0.5, 1, mn, rn]
	SA12_ = refdata([[5e3, 5e3, 4.5e3], [5, 3, 2], [60, 30.65, 28.4], [1110, 2330, 440]])
	SA12 = refdata([np.array(SA12_['raw'][i])*ln[i] for i in range(4)])

	more = 1
	lt = np.geomspace(0.1, 1e6, 200)
	plt.figure(figsize=(11,8))
	plt.subplot(221)
	plt.loglog(lt, N_t(lt), 'k', zorder=0)
	plt.loglog(lt, N_t(lt)/3, 'k--', lw=1, zorder=0)
	plt.loglog(lt, N_t(lt)*3, 'k--', lw=1, zorder=0)
	a0 = plotref(SA12, 1, 'x', 'k', 'Stacy+2010', zo=1)
	a1 = plotref(GT12, 1, 's', 'b', 'Greif+2012', zo=2)
	a2 = plotref(SA13, 1, '^', 'g', 'Stacy+2013', 'none', zo=3)
	a3 = plotref(SH14, 1, 'o', 'orange', 'Susa+2014', 'none', 1, zo=4)
	a4 = plotref(MM15, 1, 'v', 'purple', 'Machida+2015', zo=5)
	a5 = plotref(SA16, 1, 'D', 'r', 'Stacy+2016', 'none', zo=6)
	a6 = plotref(HS17, 1, 'H', 'pink', 'Hirano+2017', zo=7)
	a7 = plotref(HT20, 1, '*', oi[ci], 'Hosogawa+2020', 'none', zo=9)
	#a8 = plotref(SD20, 1, 'p', 'cyan', 'Skinner+2020', 'none', mode = 1, zo=8)
	plt.ylabel(r'$N_{\star}$', size=14)
	#plt.xlabel(r'$t,\ \tau(4\pi G\rho_{\rm ad})^{1/2}\ [\mathrm{yr}]$', size=14)
	plt.ylim(1, 1e4)
	#plt.legend(loc=2, fontsize=12)
	llab = ('Stacy+2010, 12', 'Greif+2012', 'Stacy+2013', 'Susa+2014', 'Machida+2015', 'Stacy+2016', 'Hirano+2017', 'Sugimura+2020', 'Skinner+2020')
	#plt.legend((a0, a1, a2, a3, a4, a5, a6, a7, a8), llab,
	plt.legend((a0, a1, a2, a3, a4, a5, a6, a7), llab,
		bbox_to_anchor=(0., 0., .97, .97), loc=2, ncol=2, borderaxespad=0.)#mode="expand"
	plt.subplot(222)
	plt.loglog(lt, M_t(lt), 'k', zorder=0)
	plt.loglog(lt, M_t(lt)/3, 'k--', lw=1, zorder=0)
	plt.loglog(lt, M_t(lt)*3, 'k--', lw=1, zorder=0)
	plotref(SA12, 2, 'x', 'k', zo=1, mf=1)
	plotref(SA12_, 2, 'x', 'k', zo=1, mf=1)
	plotref(GT12, 2, 's', 'b', zo=2)
	plotref(SA13, 2, '^', 'g', fs='none', zo=3)
	plotref(SA13_, 2, '^', 'g', fs='none', zo=3)
	plotref(SH14, 2, 'o', 'orange', fs='none', mode = 1, zo=4)
	plotref(SH14_, 2, 'o', 'orange', fs='none', mode = 1, zo=4)
	plotref(MM15, 2, 'v', 'purple', zo=5)
	plotref(MM15_, 2, 'v', 'purple', zo=5)
	plotref(SA16, 2, 'D', 'r', fs='none', zo=6)
	plotref(SA16_, 2, 'D', 'r', fs='none', zo=6)
	plotref(HS17, 2, 'H', 'pink', mode=0, zo=7)
	plotref(HS17_, 2, 'H', 'pink', mf=0, zo=7)
	#plotref(SD20, 2, 'p', 'cyan', fs='none', mode = 1, zo=8)
	#plotref(SD20_, 2, 'p', 'cyan', fs='none', mode = 1, zo=8)
	plotref(HT20, 2, '*', oi[ci], fs='none', zo=9)
	plotref(HT20_, 2, '*', oi[ci], fs='none', zo=9)
	plt.ylabel(r'$M\ [\mathrm{M}_{\odot}]$', size=14)
	#plt.xlabel(r'$t,\ \tau(4\pi G\rho_{\rm ad})^{1/2}\ [\mathrm{yr}]$', size=14)
	plt.text(5e4, 5e-2, r'$\gamma_{\rm eff}='+str(geff)+'$')
	plt.ylim(1e-2, 3e3)
	plt.subplot(223)
	plt.loglog(lt, R_t(lt), 'k', zorder=0)
	plt.loglog(lt, R_t(lt)/3, 'k--', lw=1, zorder=0)
	plt.loglog(lt, R_t(lt)*3, 'k--', lw=1, zorder=0)
	plotref(SA12, 3, 'x', 'k', zo=1, mf=1)
	plotref(SA12_, 3, 'x', 'k', zo=1, mf=1)
	plotref(GT12, 3, 's', 'b', zo=2)
	plotref(SA13, 3, '^', 'g', fs='none', mode=1, zo=3)
	plotref(SA13_, 3, '^', 'g', fs='none', mode=1, zo=3)
	plotref(SH14, 3, 'o', 'orange', fs='none', mode = 1, zo=4)
	plotref(SH14_, 3, 'o', 'orange', fs='none', mode = 1, zo=4)
	plotref(MM15, 3, 'v', 'purple', zo=5)
	plotref(MM15_, 3, 'v', 'purple', zo=5)
	plotref(SA16, 3, 'D', 'r', fs='none', zo=6)
	plotref(SA16_, 3, 'D', 'r', fs='none', zo=6)
	plotref(HS17, 3, 'H', 'pink', zo=7)
	plotref(HS17_, 3, 'H', 'pink', mf=0, zo=7)
	#plotref(SD20, 3, 'p', 'cyan', fs='none', mode = 1, zo=8)
	#plotref(SD20_, 3, 'p', 'cyan', fs='none', mode = 1, zo=8)
	plotref(HT20, 3, '*', oi[ci], fs='none', zo=9)
	plotref(HT20_, 3, '*', oi[ci], fs='none', zo=9)
	plt.ylabel(r'$R_{c}\ [\mathrm{AU}]$', size=14)
	#plt.xlabel(r'$t,\ \tau(4\pi G\rho_{\rm ad})^{-1/2}\ [\mathrm{yr}]$', size=14)
	plt.xlabel(r'$t\ [\rm yr]$')
	plt.ylim(0.1, 1e6)
	plt.subplot(224)
	plt.loglog(lt, Mbar(lt, lt), 'k', label=r'$t_{\rm frag}=t_{\rm acc}$', zorder=0)
	plotref(SA12, 4, 'x', 'k', zo=1, mf=1)
	plotref(SA12_, 4, 'x', 'k', zo=1, mf=1)
	plotref(GT12, 4, 's', 'b', zo=2)
	plotref(SA13, 4, '^', 'g', fs='none', zo=3)
	plotref(SA13_, 4, '^', 'g', fs='none', zo=3)
	plotref(SH14, 4, 'o', 'orange', fs='none', mode = 1, zo=4)
	plotref(SH14_, 4, 'o', 'orange', fs='none', mode = 1, zo=4)
	plotref(MM15, 4, 'v', 'purple', zo=5)
	plotref(MM15_, 4, 'v', 'purple', zo=5)
	plotref(SA16, 4, 'D', 'r', fs='none', zo=6)
	plotref(SA16_, 4, 'D', 'r', fs='none', zo=6)
	plotref(HS17, 4, 'H', 'pink', mode=0, zo=7)
	plotref(HS17_, 4, 'H', 'pink', mf=0, zo=7)
	#plotref(SD20, 4, 'p', 'cyan', fs='none', mode = 1, zo=8)
	#plotref(SD20_, 4, 'p', 'cyan', fs='none', mode = 1, zo=8)
	plotref(HT20, 4, '*', oi[ci], fs='none', zo=9)
	plotref(HT20_, 4, '*', oi[ci], fs='none', zo=9)
	ltf = [1, 3e1, 1e3]
	i = 0
	for tf in ltf:
		sel = lt>tf
		plt.loglog(lt[sel], Mbar(tf, lt[sel]), 'k', ls = lls[i+1], 
			label=r'$t_{\rm frag}='+str(tf)+'\ \mathrm{yr}$', zorder=0)
		i += 1
	plt.ylabel(r'$\bar{M}_{\star}\ [\mathrm{M_{\odot}}]$', size=14)
	#plt.xlabel(r'$t,\ \tau(4\pi G\rho_{\rm ad})^{-1/2}\ [\mathrm{yr}]$', size=14)
	plt.xlabel(r'$t\ [\rm yr]$')
	plt.legend(loc=2, fontsize=12)
	plt.ylim(1e-2, 1e3)
	plt.tight_layout()
	plt.savefig('popIII_t.pdf')
	plt.close()
	
	ltd = tdecay(M_t(lt), R_t(lt))
	plt.figure()
	plt.loglog(lt, ltd)
	plt.xlabel(r'$t,\ \tau(4\pi G\rho_{\rm ad})^{-1/2}\ [\mathrm{yr}]$', size=14)
	plt.ylabel(r'$t_{\rm decay}\ [\rm yr]$')
	plt.tight_layout()
	plt.savefig('tdecay_t.pdf')
	plt.close()
	
	
	
	tf = 100
	#ta0 = 1e5
	#ta1 = 1e6
	#M0 = M_t(ta0)
	#M1 = M_t(ta1)
	lta = [1e5, 3.5e5, 1e6]
	lM = [M_t(ta) for ta in lta]
	Ns = N_t(tf)
	#lnum0, lnum1 = [], []
	dnum = [[] for ta in lta]
	drat = [[] for ta in lta]
	m1 = 1
	nb = 200
	x1, x2 = 0.0, 2.35#1.7
	lalp = np.linspace(-x1, -x2, nb)
	i = 0
	mcut0 = [0, 8, 40, 140, 260]
	for ta in lta:
		for alp in lalp:
			imf_form = lambda m: m**alp
			m2, A = Mmax(tf, ta, alp, m1)
			#m2 = min(m2, 1e3)
			d = SP_info(m1, m2, imf_form, mcut=mcut0, mode=1)
			dnum[i].append(d['num'][-1]*d['N/M'])
			drat[i].append(d['num'][-1]/d['num'][1])
		i += 1
		#m21, A = Mmax(tf, ta1, alp, m1)
		#mcut1 = [0, 8, 25, 140, 260]#m21]
		#d = SP_info(m1, m21, imf_form, mcut=mcut1, mode=0)
		#lnum1.append(d['num'][-1])
	m1, m2 = 0.1, 200
	ln0 = []
	lr0 = []
	for alp in lalp:
		d = SP_info(m1, m2, imf_form, mcut=mcut0, mode=1)
		ln0.append(d['num'][-1]*d['N/M'])
		lr0.append(d['num'][-1]/d['num'][1])
	drat = np.array(drat)
	dnum = np.array(dnum)
	lr0 = np.array(lr0)
	ln0 = np.array(ln0)
	
	y1, y2 = 1e-2, 10
	plt.figure()
	for i in range(len(lta)):
		lrat = drat[i]
		M0 = lM[i]
		plt.plot(-lalp, lrat, ls=lls[i], label=r'$M_{\star,\rm PopIII}='+'{:.0f}'.format(M0)+r'\ \rm M_{\odot}$')
	plt.plot(-lalp, lr0, ls=lls[3], label=r'$M_{\min}=0.1\ \rm M_{\odot}$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.yscale('log')
	plt.xlabel(r'$\alpha$')
	plt.ylabel(r'$\langle N_{\rm PISN}\rangle/\langle N_{\rm CCSN}\rangle$')
	plt.legend()
	plt.tight_layout()
	plt.savefig('rat_PISN_CCSN_alp.pdf')
	plt.close()
		
	y1, y2 = 1e-2, 10
	plt.figure()
	for i in range(len(lta)):
		lrat = drat[i]
		lnum0 = dnum[i]
		M0 = lM[i]
		plt.plot(-lalp, lnum0/lrat, ls=lls[i], label=r'$M_{\star,\rm PopIII}='+'{:.0f}'.format(M0)+r'\ \rm M_{\odot}$')
	plt.plot(-lalp, ln0/lr0, ls=lls[3], label=r'$M_{\min}=0.1\ \rm M_{\odot}$')
	plt.xlim(x1, x2)
	#plt.ylim(y1, y2)
	plt.yscale('log')
	plt.xlabel(r'$\alpha$')
	plt.ylabel(r'$\langle \epsilon_{\rm CCSN}\rangle\ [\rm M_{\odot}^{-1}]$')
	plt.legend()
	plt.tight_layout()
	plt.savefig('eCCSN_alp.pdf')
	plt.close()
		
	y1, y2 = 5e-5, 5e-3
	plt.figure()
	for i in range(len(lta)):
		lnum0 = dnum[i]
		M0 = lM[i]
		plt.plot(-lalp, np.array(lnum0), ls = lls[i], label=r'$M_{\star,\rm PopIII}='+'{:.0f}'.format(M0)+r'\ \rm M_{\odot}$')
	plt.plot(-lalp, ln0, ls=lls[3], label=r'$M_{\min}=0.1\ \rm M_{\odot}$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.yscale('log')
	plt.xlabel(r'$\alpha$')
	plt.ylabel(r'$\langle \epsilon_{\rm PISN}\rangle\ [\rm M_{\odot}^{-1}]$')
	plt.legend()
	plt.tight_layout()
	plt.savefig('ePISN_alp.pdf')
	plt.close()
	
	if more==0:
		exit()
	
	#"""
	tf0, ta0, alp0, M0 = 1e2, 3e5, -1.17, 1
	lparam = []
	lparam.append([tf0, ta0, alp0, M0])
	lparam.append([1e1, ta0, alp0, M0])
	lparam.append([1e3, ta0, alp0, M0])
	lparam.append([tf0, 1e5, alp0, M0])
	lparam.append([tf0, 1e6, alp0, M0])
	lparam.append([tf0, ta0, -0.17, M0])
	lparam.append([tf0, ta0, -1, M0])
	lparam.append([tf0, ta0, -1.7, M0])
	lparam.append([tf0, ta0, alp0, 3])
	lparam.append([tf0, ta0, alp0, 10])
	lparam.append([ta0, ta0, -2.35, 0.1])
	
	plt.figure(figsize=(9,7))
	x1, x2 = 1, 2e3
	y1, y2 = 0.3, 30
	plt.subplot(221)
	for i in [0, 1, 2]:
		plotIMF(lparam[i], i, 0)
	#plt.xlabel(r'$M_{\star}\ [\mathrm{M_{\odot}}]$', size=14)
	plt.ylabel(r'$\Phi(M_{\star})\equiv dN_{\star}/d\ln M_{\star}\propto M_{\star}^{1-\alpha}$', size=14)#\ [\mathrm{M_{\odot}^{-1}}]$', size=14)
	plt.legend(loc=1, fontsize=12)
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	x1, x2 = 1, 3e3
	y1, y2 = 0.7, 10
	plt.subplot(222)
	for i in [0, 3, 4]:
		plotIMF(lparam[i], i, 1)
	#plt.xlabel(r'$M_{\star}\ [\mathrm{M_{\odot}}]$', size=14)
	#plt.ylabel(r'$\Phi(M_{\star})\equiv dN_{\star}/dM_{\star}\ [\mathrm{M_{\odot}^{-1}}]$', size=14)
	plt.legend(loc=1, fontsize=12)
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	x1, x2 = 1, 1e3
	y1, y2 = 1e-2, 50
	plt.subplot(223)
	for i in [0, 5, 6, 7]:
		plotIMF(lparam[i], i, 2)
	plt.xlabel(r'$M_{\star}\ [\mathrm{M_{\odot}}]$', size=14)
	plt.ylabel(r'$\Phi(M_{\star})\equiv dN_{\star}/d\ln M_{\star}\propto M_{\star}^{1-\alpha}$', size=14)#\ [\mathrm{M_{\odot}^{-1}}]$', size=14)
	plt.legend(fontsize=12)
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	x1, x2 = 1, 1e3
	y1, y2 = 1e-2, 1e2
	plt.subplot(224)
	for i in [0, 8, 9]:
		plotIMF(lparam[i], i, 3)
	plotIMF(lparam[10], 10, -1)
	plt.xlabel(r'$M_{\star}\ [\mathrm{M_{\odot}}]$', size=14)
	#plt.ylabel(r'$\Phi(M_{\star})\equiv dN_{\star}/dM_{\star}\ [\mathrm{M_{\odot}^{-1}}]$', size=14)
	plt.legend(fontsize=12)
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig('popIII_IMF.pdf')
	plt.close()
		
	Mtot, Ns = M_t(ta0), N_t(tf0)
	labtf = r'$t_{\rm frag}='+str(tf0/1e3)+'\ \mathrm{kyr}$'
	#labtf += r'$N_{\star}='+'{:.0f}'.format(Ns)+'$'
	labta = r'$t_{\rm acc}='+str(ta0/1e3)+'\ \mathrm{kyr}$'
	#labta += r'$M={:.0f}'.format(Mtot)+'\ \mathrm{M_{\odot}}$'
	labalp = r'$\alpha='+str(-alp0)+'$'
	
	plt.figure(figsize=(9, 3))
	ltf = np.geomspace(1e1, 1e3, 100)
	lmm = [Mmax(tf, ta0, alp0, M0)[0] for tf in ltf]
	plt.subplot(131)
	plt.plot(ltf,lmm, label=labta+', \n'+labalp)
	plt.ylabel(r'$M_{\max}\ [\mathrm{M_{\odot}}]$', size=14)
	plt.xlabel(r'$t_{\rm frag}\ [\mathrm{yr}]$', size=14)
	plt.xscale('log')
	#plt.yscale('log')
	plt.legend(fontsize=12)
	lta = np.geomspace(1e4, 1e6, 100)
	lmm = [Mmax(tf0, ta, alp0, M0)[0] for ta in lta]
	plt.subplot(132)
	plt.plot(lta,lmm, label=labtf+', \n'+labalp)
	#plt.ylabel(r'$M_{\max}\ [\mathrm{M_{\odot}}]$')
	plt.xlabel(r'$t_{\rm acc}\ [\mathrm{yr}]$', size=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.legend(fontsize=12)
	lalp = np.linspace(-1.7, 0.2, 100)
	lmm = [Mmax(tf0, ta0, alp, M0)[0] for alp in lalp]
	plt.subplot(133)
	plt.plot(-lalp,lmm, label=labtf+', \n'+labta)
	#plt.ylabel(r'$M_{\max}\ [\mathrm{M_{\odot}}]$')
	plt.xlabel(r'$\alpha$', size=14)
	plt.yscale('log')
	plt.legend(fontsize=12)
	plt.tight_layout()
	plt.savefig('Mmax_tf_ta_alp.pdf')
	plt.close()
	#"""
		
		
		
		
	
