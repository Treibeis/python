from process import *
from scipy.optimize import curve_fit
import matplotlib
import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import sys
import os
import multiprocessing as mp

Zsun = 0.02
Ob = 0.048
Om = 0.315

#Mup = lambda z: 2.5e7*((1+z)/10)**-1.5
#Mdown = lambda z: 1e6*((1+z)/10)**-2

#Mdown = lambda z: 1.54e5*((1+z)/31.)**-2.074
Mdown = lambda z: 3e6*((1+z)/10)**-1.5
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5

Mbh_Ms1 = lambda x: 10**(1.4*np.log10(x)-6.45)
Mbh_Ms2 = lambda x: 10**(1.05*np.log10(x)-4.1)

def linear(x, b0, b1):
	return b0 + x*b1

def get_range(l, fac = 0.7):
	return np.min(l)*fac, np.max(l)/fac

def Jeansm(T, rho, mu = 0.63, gamma=5./3):
	cs = (gamma*T*BOL/(mu*PROTON))**0.5
	MJ = np.pi/6 * cs**3/(GRA**3*rho)**0.5
	return MJ/Msun
	
def Mreion(z, delta=125, T=2e4):#, Ob=0.048, Om=0.315):
	rho = rhom(1/(1+z)) * delta
	MJ = Jeansm(T, rho)
	return MJ

lK0 = [9.04, 9.04, 8.99, 8.79, 8.90]
llogM0 = [11.18, 11.57, 12.38, 12.76, 12.87]
lz_mmr = [0.07, 0.7, 2.2, 3.5, 3.5]

def Z_ms_obs(ms, key):
	logM0 = llogM0[key]
	K0 = lK0[key]
	logms = np.log10(ms)
	logoh = -0.0864*(logms-logM0)**2 + K0 - 12
	logZ = logoh + 3
	return 10**logZ

def Z_ms_fire_gas(ms, z):
	logms = np.log10(ms)
	logZ = 0.35*(logms-10)+0.93*np.exp(-0.43*z)-1.05
	return 10**logZ

def Z_ms_fire_star(ms, z):
	logms = np.log10(ms)
	logZ = 0.4*(logms-10)+0.67*np.exp(-0.5*z)-1.04
	return 10**logZ

# UFDs: Segue I, Com Ber, UMa II, CVn II, Leo IV, UMa I, Bootes I, Hercules
#lmUFD = np.array([1e3, 3.7e3, 4.1e3, 7.9e3, 1e4, 1.4e4, 2.9e4, 3.7e4])
#FeHl = np.array([-3.8, -3, -3.3])
#FeHu = np.array([-2.4, -2.2, -1.1])

LGDFs = np.array(retxt('mmr_LGDFs.txt', 2))

Ms_MV = lambda mv: 1e3 * 10**((-mv-1.3)/2.5)
MVsun = 4.83
dex1 = 0.25
dex2 = 0.16
def Z_ms_local(ms, sct=0):
	FeH = (-1.68-0.03*sct*dex1/dex2)+(0.29+0.02*sct*dex1/dex2)*np.log10(ms/Ms_MV(MVsun)/1e6)
	return 10**(FeH+0.2)

def plotmmr(zind1=0, zind2=4, zref21=0, zref22=6, DFflag = 1):
	lm1 = np.geomspace(1e8, 2e11, 100)
	lm2 = np.geomspace(1e4, 1e11, 100)
	lm3 = np.geomspace(Ms_MV(0), Ms_MV(-14), 100)
	lZref31 = Z_ms_local(lm3, 1)/10**dex1
	lZref32 = Z_ms_local(lm3, -1)*10**dex1
	#print(lZref31)
	#zind1, zind2 = 0, 4
	zref11, zref12 = lz_mmr[zind1], lz_mmr[zind2] #3.5
	#zref21, zref22 = 0.0, 6.0
	lZref11 = Z_ms_obs(lm1, zind1)
	lZref12 = Z_ms_obs(lm1, zind2)
	lZref21 = Z_ms_fire_star(lm2, zref21)
	lZref22 = Z_ms_fire_star(lm2, zref22)
	#lZref3 = Z_ms_local(lm3)
	#lZref31 = lZref3/10**dex1
	#lZref32 = lZref3*10**dex1
	plt.fill_between(lm1, lZref11, lZref12, fc='r', label=r'AMAZE, $z\sim {:.1f}-{:.1f}$'.format(zref11, zref12), alpha=0.3, zorder=0)
	plt.fill_between(lm2, lZref21, lZref22, fc='b', label=r'FIRE, $z\sim {:.0f}-{:.0f}$'.format(zref21, zref22), alpha=0.3, zorder=0)
	plt.fill_between(lm3, lZref31, lZref32, fc='orange', label=r'LGDFs, $z\sim 0$', alpha=0.3, zorder=0)
	if DFflag>0:
		plt.plot(Ms_MV(LGDFs[0]), 10**(LGDFs[1]+0.2), '*', color='orange')

def halo_star(d, rep, sn, s=128, Ob = 0.048, Om=0.315, Mmin = 32*585, alp=0.7):
	z = d['z']
	mbh = d['data'][0]
	ms = d['data'][4]
	mh = d['data'][3]
	mg = d['data'][5]
	sel = ms>Mmin
	ngal = np.sum(sel)
	nh = len(mh)
	occ = ngal/nh
	lab = r'$z={:.2f}$'.format(z)+r', $f_{\mathrm{occ}}'+r'={:.3f}'.format(occ)+r'$'

	ms = ms[sel]
	mh = mh[sel]
	mbh = mbh[sel]
	mg = mg[sel]
	mb = ms + mg
	#print('Ms/Mh: ', sumy(ms/mh))
	print('Mb/Mh: ', sumy(mb/mh))
	print('Ms/Mb: ', sumy(ms/mb))
	print('Ms/Mh: ', sumy(ms/mh), np.sum(ms)/np.sum(mh))
	x1, x2 = get_range(mh)
	y1, y2 = get_range(ms)
	xr = np.array([x1, x2])
	plt.figure()
	#plt.scatter(mh, ms, s=s, c = np.log10(mb/mh*Om/Ob), cmap=plt.cm.cool, label=lab, alpha=alp)
	plt.scatter(mh, ms, s=s, c = mb/mh*Om/Ob, cmap=plt.cm.cool, label=lab, alpha=alp)
	cb = plt.colorbar()
	#cb.ax.set_title(r'$\log\left(\frac{M_{\rm{baryon}}}{f_{\rm{baryon}}M_{\rm{halo}}}\right)$')#, y=-0.11)
	cb.ax.set_title(r'$\frac{M_{\rm{baryon}}}{f_{\rm{baryon}}M_{\rm{halo}}}$')
	plt.plot(xr, xr*Ob/Om, 'k-', label=r'$M_{\mathrm{\star}}/M_{\mathrm{halo}}=f_{\mathrm{baryon}}$')
	plt.plot(xr, xr*Ob/Om*0.1, 'k--', label=r'$M_{\mathrm{\star}}/M_{\mathrm{halo}}=0.1f_{\mathrm{baryon}}$')
	plt.plot(xr, xr*Ob/Om*0.01, 'k-.', label=r'$M_{\mathrm{\star}}/M_{\mathrm{halo}}=0.01f_{\mathrm{baryon}}$')
	plt.plot(xr, xr*Ob/Om*0.001, 'k:', label=r'$M_{\mathrm{\star}}/M_{\mathrm{halo}}=10^{-3}f_{\mathrm{baryon}}$')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$M_{\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')
	plt.ylabel(r'$M_{\star}\ [\mathrm{M}_{\odot}]$')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'Ms_Mh_'+str(sn)+'.pdf')
	plt.close()	
	
def plotrela(d, rep, mode, base, sn, s = 128, Mmin = 32*585, y1=40, y2=1e5, f1=1e5, f2=1e8, alp=0.7):
	z = d['z']
	mbh = d['data'][0]
	ms = d['data'][1]
	lr = d['data'][2]
	sel = (mbh>0) * (ms>Mmin)
	nbh = np.sum(sel)
	ngal = np.sum(ms>Mmin)
	occ = nbh/ngal
	lab = r'$z={:.2f}$'.format(z)+r', $f_{\mathrm{occ}}'+r'={:.3f}'.format(occ)+r'$'
	
	ms = ms[sel]
	mbh = mbh[sel]
	lr = lr[sel]
	x1, x2 = get_range(ms)
	y1, y2 = get_range(mbh)
	xr = np.array([x1, x2])
	#sel = ms>0
	#sel = (ms>f1) * (ms<f2)
	fit = curve_fit(linear, np.log10(ms), np.log10(mbh))
	b0, b1 = fit[0]
	if b0>=0:
		flab = r'$\log M_{\mathrm{BH}}='+'{:.2f}'.format(b1)+r'\log M_{\star}'+r'+{:.2f}$'.format(b0)
	else:
		flab = r'$\log M_{\mathrm{BH}}='+'{:.2f}'.format(b1)+r'\log M_{\star}'+r'-{:.2f}$'.format(-b0)
	perr = np.sqrt(np.diag(fit[1]))
	#print(perr)
	Mbh_Ms0 = lambda x: 10**(np.log10(x)*b1 + b0)
	Mbh_Msu = lambda x: 10**(np.log10(x)*(b1+perr[1]) + b0+perr[0])
	Mbh_Msd = lambda x: 10**(np.log10(x)*(b1-perr[1]) + b0-perr[0])
	plt.figure()
	#plt.scatter(ms, mbh, s=s, c = np.log10(lr), cmap=plt.cm.cool, label=lab, alpha=alp)
	plt.scatter(ms, mbh, s=s, c = lr, cmap=plt.cm.cool, label=lab, alpha=alp)
	cb = plt.colorbar()
	#cb.ax.set_title('$\log(R_{1/2}\ [\mathrm{kpc}])$')#, y=-0.11)
	cb.ax.set_title('$R_{1/2}\ [\mathrm{kpc}]$')
	plt.plot(xr, Mbh_Ms0(xr), 'k-', label=flab)
	plt.fill_between(xr, Mbh_Msd(xr), Mbh_Msu(xr), facecolor='gray', alpha=0.5, zorder=0)
	plt.plot(xr, Mbh_Ms1(xr), 'k--', label=r'$\log M_{\mathrm{BH}}=1.4\log M_{\star}-6.45$')
	plt.plot(xr, Mbh_Ms2(xr), 'k:', label=r'$\log M_{\mathrm{BH}}=1.05\log M_{\star}-4.1$')
	plt.xlim(xr)
	plt.ylim(y1, 1e5)#y2)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$M_{\star}\ [\mathrm{M}_{\odot}]$')
	plt.ylabel(r'$M_{\mathrm{BH}}\ [\mathrm{M}_{\odot}]$')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'Mbh_Ms_'+str(sn)+'_'+base+str(mode)+'.pdf')
	plt.close()

def mbaryon(ad):
	return np.sum(ad[('PartType4', 'Masses')].to('Msun')) + np.sum(ad[('PartType0', 'Masses')].to('Msun'))

def halfr(cen0, ds, r0, rat = 10, eps = 0.1, nmax = 7, adp = 1, bs = 4000, rmin = 0.1):
	#ad = ds.sphere(cen, (r0, 'kpccm/h'))
	#mtot = mbaryon(ad)
	cen = cen0
	ad = ds.sphere(cen, (r0, 'kpccm/h'))
	m = mbaryon(ad)
	mdm = np.sum(ad[('PartType1', 'Masses')].to('Msun'))
	if m/mdm>rat:
		return r0
	r1 = 0
	r2 = r0
	old = 0
	count = 0
	while count<nmax:
		r = (r1+r2)/2
		if r<rmin:
			r = rmin
			break
		ad = ds.sphere(cen, (r, 'kpccm/h'))
		m = mbaryon(ad)
		mdm = np.sum(ad[('PartType1', 'Masses')].to('Msun'))
		if old==m/mdm:
			break
		old = m/mdm
		if abs(rat-old)/rat<=eps or mdm==0:
			break
		if old<rat:
			r2 = (r1+r2)/2
		else:
			r1 = (r1+r2)/2
		if adp>0:
			posb = np.array(ad[('PartType4','Coordinates')].to('kpccm/h'))
			cen = center(posb, cen, r, bs)
		count += 1
	return r

def rockstar(hd):
	return np.array([hd[2], hd[4], *hd[8:11]*1e3])

def caesar_obj(obj, gal = 0, Ob = 0.048, Om=0.315):
	lm = []
	lr = []
	lpos = []
	if gal==0:
		lh = obj.halos
		norm = 1
	else:
		lh = obj.galaxies
		norm = Ob/Om
	for halo in lh:
		lm.append(halo.masses['total'].to('Msun/h')/norm)
		lr.append(halo.radii['virial'].to('kpccm/h')/norm**(1/3))
		lpos.append(halo.pos.to('kpccm/h'))
	else:
		lg = obj.galaxies
	return np.array([lm, lr, *np.array(lpos).T])

def process_halo(hd0, ds, fac1 = 1, fac2 = 0, bs = 4000, norm = 1, mode=0, rfac=0.5, nc = 3, fsk = 0.5, rat = 10, rmin = 1, nsep = 5, ncore = 4, gal = 1, adp = 1, Mmin = 32*585, mlab=0): #Om/(Om-Ob)):
	z = ds['Redshift']
	h = ds['HubbleParam']
	if mlab==0:
		Mat = Mdown(z)
	else:
		Mat = Mup(z)
	lmh = hd0[0]*norm/h
	sel = lmh>Mat
	hd = hd0.T[sel]
	hd = hd.T
	lr = hd[1]
	lpos = hd[2:5].T
	nh = len(lr)
	print('Nhalo_tot: {}'.format(nh))
	np_core = int(nh/ncore)
	base = np.linspace(0, np_core, np_core+1, dtype='int')*ncore #np.array(range(np_core+1))*ncore
	lpr = [base+i for i in range(ncore)]
	lpr = [pr[pr<nh] for pr in lpr]
	print(lpr)
	manager = mp.Manager()
	def sess(pr, k):
		nh = len(pr)
		print('Nhalo({}): {}'.format(k, nh))
		lmh = np.zeros(nh)#hd[0]*norm/h
		lrhf = np.zeros(nh)
		lms = np.zeros(nh)
		lmg = np.zeros(nh)
		lmbh = np.zeros(nh)
		ct = 0
		for i in pr:
			pos0 = lpos[i]
			R = lr[i]*fac1
			ad0 = ds.sphere(pos0, (R, 'kpccm/h'))
			lms[ct] = np.sum(np.array(ad0[('PartType4', 'Masses')].to('Msun')))
			lmg[ct] = np.sum(np.array(ad0[('PartType0', 'Masses')].to('Msun')))
			mdm = np.sum(np.array(ad0[('PartType1', 'Masses')].to('Msun')))
			lmh[ct] = mdm + lmg[ct] + lms[ct]
			if gal==0:
				posb = np.vstack([ad0[('PartType0','Coordinates')].to('kpccm/h'), ad0[('PartType4','Coordinates')].to('kpccm/h')])
			else:
				posb =  ad0[('PartType4','Coordinates')].to('kpccm/h')
			posb = np.array(posb)
			r = R
			if len(posb)==0 or lms[ct]<Mmin:
				lms[ct]=0
				cen = pos0
			else:
				ad = ad0
				cen = pos0
				for j in range(nc):
					cen = center(posb, cen, r, bs)
					r = r*fsk
					if r<rmin:
						r = r/fsk
						break
					ad = ds.sphere(cen, (r, 'kpccm/h'))
					if gal==0:
						posb = np.vstack([ad[('PartType0','Coordinates')].to('kpccm/h'), ad[('PartType4','Coordinates')].to('kpccm/h')])
					else:
						posb =  ad[('PartType4','Coordinates')].to('kpccm/h')
					posb = np.array(posb)
					if len(posb)==0:
						r = r/fsk
						break
			if fac2>0:
				rhf = max(lr[i]/fac2, rmin)
			else:
				rhf = halfr(cen, ds, r, rat, adp=adp, bs=bs, rmin=rmin)*rfac
			lrhf[ct] = rhf
			ad_ = ds.sphere(cen, (rhf, 'kpccm/h'))
			stbh = np.array(ad_[('PartType3', 'Star Type')])
			if mode==0:
				lmbh[ct] = np.max(ad_[('PartType3', 'BH_Mass')][stbh>=30000])
			else:
				lmbh[ct] = np.sum(ad_[('PartType3', 'BH_Mass')][stbh>=30000])
			ct += 1
			if ct%nsep==0 and k==0:
				print('Processor {}: {}%'.format(k, ct/nh*100))
		output.put(np.array([lmbh, lms, lrhf, lmh, lms, lmg]))
	output = manager.Queue()
	pros = [mp.Process(target=sess, args=(pr, k)) for pr, k in zip(lpr,range(ncore))]
	for p in pros:
		p.start()
	for p in pros:
		p.join()
	out = [output.get() for p in pros]
	datah = np.hstack(out)
	d = {}
	d['z'] = z
	d['data'] = datah
	return d
		
def recaesar(sn, rep = './', base = 'snapshot', ext = '.hdf5', post = 'caesar'):
	#ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	obj = caesar.load(rep+post+'_'+str(sn).zfill(3)+ext)
	#obj.yt_dataset = ds
	return obj

def SFhistory(sn, ind, mh, ds, ad, rep='./', dt = 1, edge=0.5):
	z = ds['Redshift']
	redlab = r'$z={:.1f}$'.format(z)
	mhlab = r'$M_{\rm halo}='+'{:.2e}'.format(mh)+r'\ \rm M_{\odot}$'
	t0 = TZ(z)/YR/1e6
	m0 = np.array(ad[('PartType4', 'Masses')].to('Msun'))[0]
	sa = np.hstack([ad[('PartType4', 'StellarFormationTime')], ad[('PartType3', 'StellarFormationTime')]])
	stype = np.hstack([ad[('PartType4', 'Star Type')], ad[('PartType3', 'Star Type')]])
	sel3 = stype >= 30000
	sel2 = np.logical_not(sel3)
	t3 = np.array([TZ(x) for x in 1/sa[sel3]-1])/YR/1e6
	t2 = np.array([TZ(x) for x in 1/sa[sel2]-1])/YR/1e6
	t = np.hstack([t3, t2])
	ti = np.min(t)
	nt = int(abs(t0-ti)/dt)+1
	print('Num of timebins: ', nt)
	print('Delay time: {:.1f} Myr'.format(np.min(t2)-np.min(t3)))
	ti_ = t0-nt*dt
	ed = np.linspace(ti_, t0, nt+1)
	h2, ed = np.histogram(t2, ed)
	h3, ed = np.histogram(t3, ed)
	his, ed = np.histogram(t, ed)
	base = midbin(ed)
	norm = m0/(dt*1e6)
	y1, y2 = norm*edge, np.max(his*norm)/edge
	plt.figure()
	plt.plot(base, his*norm, 'k', label='Total\n'+redlab+', '+mhlab, zorder=0)
	plt.plot(base, h3*norm, '--', label='Pop III', zorder=2)
	plt.plot(base, h2*norm, '-.', label='Pop II', zorder=1)
	plt.xlabel(r'$t\ [\rm Myr]$')
	plt.ylabel(r'$\dot{M}_{\star}\ [\rm M_{\odot}\ yr^{-1}]$')
	plt.yscale('log')
	#plt.xscale('log')
	plt.xlim(ti_, t0)
	plt.ylim(y1, y2)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'SFR_t_'+str(sn)+'_'+str(ind)+'.pdf')
	plt.close()

def popIII_pos(ds, ad, dt = 10, Zsun = 0.02):
	z = ds['Redshift']
	t = TZ(z)/YR/1e6
	t1 = (t-dt)/1e3
	z1 = ZT(np.log10(t1))
	a1 = 1/(1+z1)
	#ad = ds.all_data()
	lpos = [[], []]
	lZ = [[], []]
	poss = np.array(ad[('PartType4','Coordinates')].to('kpccm/h'))
	Zs = np.array(ad[('PartType4', 'Metallicity_00')]/Zsun)
	st = ad[('PartType4', 'Star Type')]
	sels = st>=3e4
	if np.sum(sels)>0:
		lpos[0] = np.array(poss[sels])
		lZ[0] = Zs[sels]
		print('Z (<3 Myr) [Zsun]: ', Zs[sels])
	stbh = np.array(ad[('PartType3', 'Star Type')])
	poss = np.array(ad[('PartType3','Coordinates')].to('kpccm/h'))[stbh>=3e4]
	Zbh = np.array(ad[('PartType3', 'Metallicity_00')]/Zsun)[stbh>=3e4]
	st = np.array(ad[('PartType3', 'StellarFormationTime')])[stbh>=3e4]
	selbh = st>a1
	if np.sum(selbh)>0:
		print('Z (<10 Myr) [Zsun]: ', Zbh[selbh])
		if len(lpos[0])>0:
			lpos[1] = np.vstack([lpos[0], poss[selbh]])
		else:
			lpos[1] = np.array(poss[selbh])
		if len(lZ[0])>0:
			lZ[1] = np.hstack([lZ[0], Zbh[selbh]])
		else:
			lZ[0] = Zbh[selbh]
	else:
		lpos[1] = lpos[0]
		lZ[1] = lZ[0]
	print('Number of active Pop III particles at z = {:.2f}: {} (<3 Myr), {} (<10 Myr)'.format(z, len(lZ[0]), len(lZ[1])))
	return np.array(lpos), lZ 
	
def starcount(ds, pos, R, dt, m0=586, metal=0, gas=0, mode=0):
	z = ds['Redshift']
	t = TZ(z)/YR/1e6
	t1 = (t-dt)/1e3
	z1 = ZT(np.log10(t1))
	a1 = 1/(1+z1)
	#h = ds['HubbleParam']
	ad = ds.sphere(pos, (R, 'kpccm/h'))
	lms = np.array(ad[('PartType4', 'Masses')].to('Msun'))
	ms = 0
	ms_ = 0
	if len(lms)>0:
		ms = np.sum(lms)
		if mode>0:
			las = np.array(ad[('PartType4', 'StellarFormationTime')])
			sel = las>a1
			if np.sum(sel)>0:
				ms_ = np.sum(lms[sel])
	st = ad[('PartType4', 'Star Type')]
	sel = st>=30000
	ms1 = 0
	if np.sum(sel)>0 and ms>0:
		ms1 = np.sum(np.array(ad[('PartType4', 'Masses')][sel].to('Msun')))
	#lmbh = np.array(ad[('PartType3', 'Masses')].to('Msun'))
	stbh = np.array(ad[('PartType3', 'Star Type')])
	selbh = stbh>=30000
	labh = np.array(ad[('PartType3', 'StellarFormationTime')])[selbh]
	lmbh = np.array(ad[('PartType3', 'BH_NProgs')])[selbh]*m0
	sel = labh>a1
	ms2 = np.sum(lmbh[sel])
	lZ = []
	if metal>0:
		mbh = np.sum(lmbh)
		lZs = np.array(ad[('PartType4', 'Metallicity_00')]/Zsun)
		lZbh = np.array(ad[('PartType3', 'Metallicity_00')]/Zsun)[selbh]
		if gas>0:
			lmg = np.array(ad[('PartType0', 'Masses')].to('Msun'))
			lZg = np.array(ad[('PartType0', 'Metallicity_00')]/Zsun)
			mg = np.sum(lmg)
			Zg = 0
			if mg>0:
				Zg = np.sum(lZg*lmg)/mg
		Zs = 0
		if ms>0:
			Zs = np.sum(lZs*lms)/ms
		Zbh = 0
		if mbh>0:
			Zbh = np.sum(lZbh*lmbh)/mbh
		if gas>0:
			lZ = [Zs, Zbh, Zg, ms, mbh, mg]
		else:
			lZ = [Zs, Zbh, ms, mbh]
	if mode==0:
		return [ms+ms2, ms1, ms2+ms1] + lZ
	else:
		return [ms_+ms2, ms1, ms2+ms1] + lZ

llb = [r'$x\ [h^{-1}\mathrm{kpc}]$', r'$y\ [h^{-1}\mathrm{kpc}]$', r'$z\ [h^{-1}\mathrm{kpc}]$']

def hostdis(rep, sn, hd0, ms10, ms20, ds, dt, norm=1, nb=100):
	h = ds['HubbleParam']
	z = ds['Redshift']
	lmh = hd0[0]*norm/h
	ms1 = np.array(ms10)
	ms2 = np.array(ms20)
	lmh1, sel1 = np.unique(ms1[0], return_index=True)
	lmh2, sel2 = np.unique(ms2[0], return_index=True)
	lms1 = ms1[6][sel1] #np.unique(ms1[6])
	lms2 = ms2[6][sel2] #np.unique(ms2[6])
	m31 = ms1[7][sel1] #np.unique(ms1[7])
	m32 = ms2[8][sel2] #np.unique(ms2[8])
	print('z = {:.1f}'.format(z))
	print(len(lmh1), len(ms1[0]))
	print(len(lmh2), len(ms2[0]))
	m1, m2 = min(Mdown(z), 1e6), np.max(lmh)*1.1
	ed = np.geomspace(m1, m2, nb+1)
	redlab = r'$z={:.1f}$'.format(z)
	plt.figure()
	his, edge, pat = plt.hist(lmh, ed, alpha=0.5, label='Haloes, '+redlab, color='g')
	plt.hist(lmh1, ed, label=r'$\tau<3\ \rm Myr$', color='orange', alpha=0.9)
	plt.hist(lmh2, ed, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
	y1, y2 = 0.7, np.max(his)*2
	plt.plot([Mup(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm atom}$')
	plt.plot([Mreion(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm J,ion}$')
	plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'Counts')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(m1, m2)
	plt.ylim(y1, y2)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'mh_dis_'+str(sn)+'.pdf')
	plt.close()
	mion = Mreion(z)
	matm = Mup(z)
	nh = np.sum(lmh>matm)
	nh_ = np.sum(lmh>mion)
	n1 = np.sum(lmh1>matm)
	n2 = np.sum(lmh2>matm)
	n1_ = np.sum(lmh1>mion)
	n2_ = np.sum(lmh2>mion)
	lfmol = [np.sum(ms1[0]<matm)/len(ms1[0]), np.sum(ms2[0]<matm)/len(ms2[0])]
	lfiso = [np.sum(ms1[7]==ms1[6])/len(ms1[0]), np.sum(ms2[8]==ms2[6])/len(ms2[0])]
	print('Occupation fraction (>Mth_atom): {:.3e} (< 3 Myr), {:.3e} (< 10 Myr)'.format(n1/nh, n2/nh))
	print('Occupation fraction (>MJ_ion): {:.3e} (< 3 Myr), {:.3e} (< 10 Myr)'.format(n1_/nh_, n2_/nh_))
	sel1m = ms1[0]<matm
	fiso1m = np.sum(ms1[7][sel1m]==ms1[6][sel1m])/np.sum(sel1m)
	sel2m = ms2[0]<matm
	fiso2m = np.sum(ms2[7][sel2m]==ms2[6][sel2m])/np.sum(sel2m)
	print('Fration of new SF in minihaloes: {:.2f}, {:.2f}'.format(fiso1m, fiso2m))
	return [n1/nh, n2/nh, n1_/nh_, n2_/nh_, nh, nh_], lfiso + lfmol, [np.sum(ms1[7][sel1m]==ms1[6][sel1m]), np.sum(sel1m), np.sum(ms2[7][sel2m]==ms2[6][sel2m]), np.sum(sel2m)]

def cumdis(a):
	cum = np.cumsum(a)
	return cum/cum[-1]

def hostscatter(rep, ms1, ms2, pos1, pos2, zr, dt, nb = 30, Zth=1e-4, gas=0, Zfloor=1e-8):
	#zm = np.mean(zr)
	zm = zr[0]
	redlab0 = r'($z\sim {:.0f})$'.format(zm)
	redlab = r'$z\sim {:.1f}-{:.1f}$'.format(*zr)
	lmh1, sel1 = np.unique(ms1[0], return_index=True)
	lmh2, sel2 = np.unique(ms2[0], return_index=True)
	lms1 = ms1[6][sel1] #np.unique(ms1[6])
	lms2 = ms2[6][sel2] #np.unique(ms2[6])
	m31 = ms1[7][sel1] #np.unique(ms1[7])
	m32 = ms2[8][sel2] #np.unique(ms2[8])
	print('Nhalo = {} (< 3 Myr), {} (< 10 Myr)'.format(len(m31), len(m32)))
	print(sumy(m31))
	print(sumy(m32))
	mion = Mreion(zm)
	print('MJ_ion/Mth_atom = {:.2f}'.format(mion/Mup(zm)))
	
	m1, m2 = min(np.min(lmh1), np.min(lmh2))*0.99, max(np.max(lmh1), np.max(lmh2))*1.01
	ed = np.geomspace(m1, m2, nb+1)
	base = midbin(ed)
	plt.figure()
	his1, edge, pat = plt.hist(ms1[0], ed, label=r'$\tau<3\ \rm Myr$', color='orange', alpha=0.9)
	his2, edge, pat = plt.hist(ms2[0], ed, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
	sel = base>mion
	tf = his2[sel]
	mtt0 = np.sum(his2[base<=Mup(zm)])
	mtt1 = np.sum(tf)
	fit = curve_fit(linear, np.log10(base[sel][tf>0]), np.log10(tf[tf>0]))
	b0, b1 = fit[0]
	fms0 = lambda x: 10**(np.log10(x)*b1 + b0)
	flab = r'$dM_{\rm PopIII}/d\log M_{\rm halo} \propto M_{\rm halo}^{'+'{:.2f}'.format(b1)+'}$'
	mt1 = fms0(mion)
	plt.plot(base[sel], fms0(base[sel]), 'k-', label=flab)
	sel = (base<=mion) * (base>Mup(zm))
	tf = his2[sel]
	mtt2 = np.sum(tf)
	fit = curve_fit(linear, np.log10(base[sel][tf>0]), np.log10(tf[tf>0]))
	b0, b1 = fit[0]
	fms0 = lambda x: 10**(np.log10(x)*b1 + b0)
	flab = r'$dM_{\rm PopIII}/d\log M_{\rm halo} \propto M_{\rm halo}^{'+'{:.2f}'.format(b1)+'}$'
	mt2 = fms0(mion)
	print('Jump ratio: {:.2f}'.format(mt2/mt1))
	print('Mass ratio: {:.2f}, {:.2f}'.format(mtt2/mtt1, mtt0/mtt1))
	plt.plot(base[sel], fms0(base[sel]), 'k--', label=flab)
	y1, y2 = 0.7, np.max(np.hstack([his1, his2]))*10
	plt.plot([Mup(zm)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm atom}$'+redlab0)
	plt.plot([mion]*2, [y1, y2], 'k-.', label=r'$M_{\rm J,ion}$'+redlab0)
	#plt.plot([x1, x2], [1, (x2/x1)**(1/3)], 'k-', 
	#	label=r'$\propto (R_{\rm PopIII}/R_{\rm vir})^{1/3}$')
	plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'Counts')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(m1, m2)
	plt.ylim(y1, y2)
	plt.legend(loc=1)
	plt.text(m2/10, y2*0.04, redlab)
	plt.tight_layout()
	plt.savefig(rep+'mh_dis.pdf')
	plt.close()

	sel1 = lmh1>mion
	sel2 = lmh2>mion
	
	#"""
	y1, y2 = np.min(m32)*0.7, np.max(m32)*7
	xr = [np.min(lmh2)*0.7, np.max(lmh2)/0.7]
	fit = curve_fit(linear, np.log10(lmh2[sel2]), np.log10(m32[sel2]))
	b0, b1 = fit[0]
	b1 = 0.22
	b0 -= b1*np.log10(mion)
	perr = np.sqrt(np.diag(fit[1]))
	flab = r'$M_{\rm PopIII}\propto M_{\rm halo}^{'+'{:.2f}'.format(b1)+'}$'
	fms0 = lambda x: 10**(np.log10(x)*b1 + b0)
	#fmsu = lambda x: 10**(np.log10(x)*(b1+1*perr[1]) + b0+1*perr[0])
	#fmsd = lambda x: 10**(np.log10(x)*(b1-1*perr[1]) + b0-1*perr[0])
	#plt.plot(lmh2[lmh2>mion], fms0(lmh2[lmh2>mion]), 'k-', label=flab)
	#plt.fill_between(xr, fmsd(xr), fmsu(xr), facecolor='gray', alpha=0.35, zorder=0)
	plt.scatter(lmh2, m32, c=np.log10(lms2), cmap=plt.cm.cool, 
		marker='o', label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=0.7)
	cb = plt.colorbar()
	cb.ax.set_title(r'$\log(M_{\rm \star}\ [\rm M_{\odot}])$')
	plt.scatter(lmh1, m31, c='k', cmap=plt.cm.cool, 
		marker='^', label=r'$\tau<3\ \rm Myr$', alpha=1)
	plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		#plt.plot([Mdown(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm mol}$')
	plt.plot([Mup(zm)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm atom}$'+redlab0)
	plt.plot([mion]*2, [y1, y2], 'k-.', label=r'$M_{\rm J,ion}$'+redlab0)
	plt.ylabel(r'$M_{\mathrm{PopIII}}\ [\rm M_{\odot}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	plt.text(xr[1]/10, y2/2, redlab)
	plt.legend(loc=2)
	plt.tight_layout()
	plt.savefig(rep+'MpopIII_Mh_uniq.pdf')
	plt.close()
	#"""
	
	prflag = 0
	x1, x2 = 293, 9669
	ed = np.linspace(x1, x2, 17)
	base = midbin(ed)
	plt.figure()
	if prflag>0:
		mm = np.mean(m32[lmh2>mion])
		his1, edge, pat = plt.hist(m31[lmh1>mion], ed, label=r'$\tau<3\ \rm Myr$', color='orange', alpha=0.9)
		his2, edge, pat = plt.hist(m32[lmh2>mion], ed, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
	else:
		mm = np.mean(m32)
		his1, edge, pat = plt.hist(m31, ed, label=r'$\tau<3\ \rm Myr$', color='orange', alpha=0.9)
		his2, edge, pat = plt.hist(m32, ed, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
	y1, y2 = 0.7, np.max(np.hstack([his1, his2]))*2
	plt.plot([mm]*2, [y1, y2], 'k:', label=r'$\langle M_{\rm PopIII}\rangle\simeq 10^{3}\ \rm M_{\odot}$')
	plt.xlabel(r'$M_{\rm PopIII}\ [\rm M_{\odot}]$')
	plt.ylabel(r'Counts')
	#plt.xscale('log')
	plt.yscale('log')
	plt.xlim(0, x2)
	plt.ylim(y1, y2)
	plt.legend(loc=1)
	plt.text(x2/1.3, y1*2, redlab)
	plt.tight_layout()
	if prflag>0:
		plt.savefig(rep+'MpopIII_dis_pr.pdf')
	else:
		plt.savefig(rep+'MpopIII_dis.pdf')
	plt.close()
	
	sel1_ = lmh1>Mup(zm)
	sel2_ = lmh2>Mup(zm)
	r1 = m31/lms1
	r2 = m32/lms2
	x1, x2 = 1e-6, 1
	ed = np.geomspace(x1, x2, 1201)
	base = midbin(ed)
	his1_, edge = np.histogram(r1[sel1_], ed)
	his2_, edge = np.histogram(r2[sel2_], ed)
	his1, edge = np.histogram(r1[sel1], ed)
	his2, edge = np.histogram(r2[sel2], ed)
	cum1_ = cumdis(his1_)
	cum2_ = cumdis(his2_)
	cum1 = cumdis(his1)
	cum2 = cumdis(his2)
	print(1-np.max(cum1_[cum1_<1]), 1-np.max(cum2_[cum2_<1]))
	print(1-np.max(cum1[cum1<1]), 1-np.max(cum2[cum2<1]))
	
	plt.figure()
	plt.plot(base, 1-cum1_, label=r'$\tau<3\ \rm Myr$ ($M_{\rm halo}>M_{\rm th}^{\rm atom}$)', color='orange')
	plt.plot(base, 1-cum2_, label=r'$\tau<{}\ \rm Myr$'.format(dt)+r' ($M_{\rm halo}>M_{\rm th}^{\rm atom}$)', color='r', ls='--')
	plt.plot(base, 1-cum1, label=r'$\tau<3\ \rm Myr$ ($M_{\rm halo}>M_{\rm J,ion}$)', color='orange', ls='-.')
	plt.plot(base, 1-cum2, label=r'$\tau<{}\ \rm Myr$'.format(dt)+r' ($M_{\rm halo}>M_{\rm J,ion}$)', color='r', ls=':')
	y1, y2 = 0, 1
	plt.fill_between([1e-4, 1e-2], [y1]*2, [y2]*2, fc='gray', alpha=0.5, label=r'$F_{\rm PopIII}\sim F_{\rm PopII/I}$')
	plt.legend()
	plt.xlabel(r'$M_{\rm PopIII}/M_{\star}$')
	plt.ylabel(r'$F(>M_{\rm PopIII}/M_{\star})$')
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.text(x1*2, y2*0.07, redlab)
	plt.tight_layout()
	plt.savefig(rep+'massrat_dis.pdf')
	plt.close()
	
	hflag = 0
	if hflag>0:
		mcut = Mup(zm)
	else:
		mcut = mion
	sel1 = ms1[0]>mcut
	sel2 = ms2[0]>mcut
	dr1 = ms1[5][sel1]
	dr2 = ms2[5][sel2]
	x1, x2 = min(np.min(dr2), np.min(dr1))*0.99, 1.0
	x1 = 1e-3
	ed = np.geomspace(x1, x2, nb+1)#int(nb/2)+1)
	base = midbin(ed)
	plt.figure()
	his1, edge, pat = plt.hist(dr1, ed, label=r'$\tau<3\ \rm Myr$', color='orange', alpha=0.9)
	his2, edge, pat = plt.hist(dr2, ed, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
	print('Fraction in the center: {:.2e}'.format(np.sum(his2[base<1e-2])/np.sum(his2)))
	print('Fraction on the edge: {:.2e}'.format(np.sum(his2[base>1e-1])/np.sum(his2)))
	#fit1 = curve_fit(linear, np.log10(base[his1>0]), np.log10(his1[his1>0]))
	fit2 = curve_fit(linear, np.log10(base[his2>0]), np.log10(his2[his2>0]))
	#print(fit1[0], fit2[0])
	b0, b1 = fit2[0]
	nm = 10**fit2[0][0]
	plt.plot([x1, x2], [nm*(x1/x2)**b1, nm], 'k-', 
		label=r'$dM_{\rm PopIII}/d\log r_{\rm PopIII}\propto r_{\rm PopIII}^{'+'{:.2f}'.format(b1)+'}$')
	y1, y2 = 0.7, np.max(np.hstack([his1, his2]))*1.1
	plt.xlabel(r'$r_{\rm PopIII}$')
	#plt.xlabel(r'$R_{\rm PopIII}/R_{\rm vir}$')
	plt.ylabel(r'Counts')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.legend(loc=2)
	plt.text(x2/5, y2*0.7, redlab)
	plt.tight_layout()
	if hflag>0:
		plt.savefig(rep+'rrat_dis_atom.pdf')
	else:
		plt.savefig(rep+'rrat_dis.pdf')
	plt.close()

	sel1_ = ms1[0]>Mup(zm)
	sel2_ = ms2[0]>Mup(zm)
	sel1m = ms1[0]<Mup(np.mean(zr))
	sel2m = ms2[0]<Mup(np.mean(zr))
	if len(ms1)>13:
		Z10 = (ms1[9]*ms1[12]+ms1[10]*ms1[13])/(ms1[12]+ms1[13])
		Z20 = (ms2[9]*ms2[12]+ms2[10]*ms2[13])/(ms2[12]+ms2[13])
	else:
		Z10 = (ms1[9]*ms1[11]+ms1[10]*ms1[12])/(ms1[11]+ms1[12])
		Z20 = (ms2[9]*ms2[11]+ms2[10]*ms2[12])/(ms2[11]+ms2[12])
	if gas==0:
		Z1 = Z10[sel1]
		Z1_ = Z10[sel1_]
		Z1m = Z10[sel1m]
		Z2m = Z20[sel2m]
		#Z2 = Z20[sel2]
		#Z2_ = Z20[sel2_]
	else:
		Z1 = ms1[11][sel1]
		Z1_ = ms1[11][sel1_]
		Z1m = ms1[11][sel1m]
		Z2m = ms2[11][sel2m]
		#Z2 = ms2[11][sel2]
		#Z2_ = ms2[11][sel2_]
	Z2 = Z20[sel2]
	Z2_ = Z20[sel2_]
	Z1_[Z1_<=Zfloor] = Zfloor
	Z2_[Z2_<=Zfloor] = Zfloor
	
	Z1m[Z1m<Zfloor] = Zfloor
	Z2m[Z2m<Zfloor] = Zfloor
	print(sumy(Z1m), sumy(Z2m))
	
	x1, x2 = Zfloor, 1
	ed0 = np.geomspace(x1, x2, 40+1)
	plt.figure()
	his1_, edge, pat = plt.hist(Z1m, ed0, label=r'$\tau<3\ \rm Myr$', color='orange', alpha=0.9)
	his2_, edge, pat = plt.hist(Z2m, ed0, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
	y1, y2 = 0.7, np.max(np.hstack([his1_, his2_]))*1.1
	plt.plot([Zth]*2, [y1, y2], 'k-.', label=r'$Z_{\rm th}$')
	plt.xlabel(r'$Z\ [\mathrm{Z_{\odot}}]$')
	plt.ylabel(r'Counts')
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.legend(loc=4)
	plt.text(x2/50, y2*0.7, redlab)
	plt.tight_layout()
	if gas==0:
		plt.savefig(rep+'Zs_dis.pdf')
	else:
		plt.savefig(rep+'Zg_dis.pdf')
	plt.close()
	
	ed = np.geomspace(x1, x2, 1601)
	base = midbin(ed)
	his1_, edge = np.histogram(Z1_, ed)
	his2_, edge = np.histogram(Z2_, ed)
	
	Z1s = pos1[-1][sel1_]
	Z2s = pos2[-1][sel2_]
	Z1s[Z1s<=Zfloor] = Zfloor
	Z2s[Z2s<=Zfloor] = Zfloor
	Z1[Z1<=Zfloor] = Zfloor
	Z2[Z2<=Zfloor] = Zfloor
	his1, edge = np.histogram(Z1, ed)
	his2, edge = np.histogram(Z2, ed)
	cum1 = cumdis(his1)
	cum2 = cumdis(his2)
	cum1_ = cumdis(his1_)
	cum2_ = cumdis(his2_)
	his1s, edge = np.histogram(Z1s, ed)
	his2s, edge = np.histogram(Z2s, ed)
	cum1s = cumdis(his1s)
	cum2s = cumdis(his2s)
	plt.figure()
	plt.plot(base, cum1s, color='orange', label=r'$\tau<3\ \rm Myr$ (self)', lw=3)
	plt.plot(base, cum2s, 'r--', label=r'$\tau<{}\ \rm Myr$ (self)'.format(dt), lw=3)
	plt.plot(base, cum1_, color='orange', label=r'$\tau<3\ \rm Myr$ ($M_{\rm halo}>M_{\rm th}^{\rm atom}$)', ls='-')
	plt.plot(base, cum2_, 'r--', label=r'$\tau<{}\ \rm Myr$'.format(dt)+r' ($M_{\rm halo}>M_{\rm th}^{\rm atom}$)')
	plt.plot(base, cum1, color='orange', label=r'$\tau<3\ \rm Myr$ ($M_{\rm halo}>M_{\rm J,ion}$)', ls='-.')
	plt.plot(base, cum2, 'r:', label=r'$\tau<{}\ \rm Myr$'.format(dt)+r' ($M_{\rm halo}>M_{\rm J,ion}$)')
	#y1, y2 = 1e-2, 50
	y1, y2 = 0, 1.7
	plt.plot([Zth]*2, [y1, y2], 'k-.', label=r'$Z_{\rm th}$')
	plt.xlabel(r'$Z\ [\mathrm{Z_{\odot}}]$')
	plt.ylabel(r'$F(<Z)$')
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.legend(loc=2)
	plt.text(x2/1e2, y2*0.9, redlab)
	plt.tight_layout()
	plt.savefig(rep+'ZpopIII_dis.pdf')
	plt.close()
	fmp1 = np.array([cum1s[0], cum1_[0], cum1[0]])
	fmp2 = np.array([cum2s[0], cum2_[0], cum2[0]])
	print(fmp1, fmp2)
	print(0.5*(fmp1+fmp2))
	print(np.min(base[cum1_>0.2]))
	print(np.min(base[cum2_>0.3]))

def promatch(rep, sn, ms1, ms2, pos1, pos2, ds, dt, ext0 = 0.3, Bs=4e3, norm0=0.7, nump=1e6, nb=1, fac=1.1, mode=0, Zth=1e-4, nf=1):
	z = ds['Redshift']
	redlab = r'$z={:.1f}$'.format(z)
	if mode==0:
		cen = np.array([Bs]*3)/2
		dim = np.array([ext]*3)*Bs/2
	else:
		cen = np.average(pos2,axis=0)
		print(cen)
		rmax = np.max([np.linalg.norm(pos-cen) for pos in pos2])*fac
		dim = np.array([min(ext*Bs/2, rmax)]*3)
	norm = norm0*(8e2/dim[0])**2
	le, re = cen-dim, cen+dim
	ad = ds.box(le, re)
	posg = np.array(ad[('PartType1','Coordinates')].to('kpccm/h'))
	#pos0 = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
	if nump>0:
		sel = np.random.choice(posg.shape[0], min(int(nump),posg.shape[0]),replace=False)
		posg = posg[sel].T
		#sel0 = np.random.choice(pos0.shape[0], min(int(nump),posg.shape[0]),replace=False)
		#pos0 = pos0[sel0]
	else:
		posg = posg.T
	#lZg = np.array(ad[('PartType0', 'Metallicity_00')]/Zsun)[sel0]
	#sell = lZg<Zth
	#pos0 = pos0[sell]
	lr1 = np.array(ms1[4])
	lr2 = np.array(ms2[4])
	posh1 = ms1[1:4]
	posh2 = ms2[1:4]
	if nf>1:
		plt.figure(figsize=(11,5.5))
	else:
		plt.figure(figsize=(5.5,5.5))
	ax = [0, 1]
	ax0 = ax
	if nf>1:
		axis = plt.subplot(121)
	else:
		axis = plt.subplot(111)
	axis.set_aspect(aspect=1)
	plt.hist2d(posg[ax[0]],posg[ax[1]],bins=int(dim[0]/nb),norm=LogNorm(),cmap=plt.cm.Blues)
	plt.clim(1, 5e3)
	#cb = plt.colorbar()
	plt.scatter(posh1[ax[0]],posh1[ax[1]],s=lr1**2*norm,ec='orange',fc='none',lw=1,zorder=3)
	plt.scatter(posh2[ax[0]],posh2[ax[1]],s=lr2**2*norm,ec='r',fc='none',lw=1,zorder=2)
	if len(pos1)>0:
		plt.scatter(*pos1.T[ax], marker='^', s=16, alpha=1, color='orange', zorder=4, label=r'$\tau<3\ \rm Myr$, '+redlab, lw=0.5, ec='k')
	if len(pos2)>len(pos1):
		plt.scatter(*pos2[len(pos1):].T[ax], marker='o', s=16, alpha=0.9, color='r', zorder=3, label=r'$\tau\sim 3-{}\ \rm Myr$'.format(dt), lw=0.5, ec='k')
	plt.xlim(le[ax[0]], re[ax[0]])
	plt.ylim(le[ax[1]], re[ax[1]])
	plt.xlabel(llb[ax[0]])
	plt.ylabel(llb[ax[1]])
	if nf<=1:
		plt.legend(loc=2)
		plt.tight_layout()
		plt.savefig(rep+'popIIIpos_'+str(sn)+'_'+str(np.sum(ax))+'.png', dpi=300)
		plt.close()
	ax = [0, 2]
	if nf>1:
		axis = plt.subplot(122)
	else:
		plt.figure(figsize=(5.5,5.5))
		axis = plt.subplot(111)
	axis.set_aspect(aspect=1)
	#plt.hist2d(*pos0.T[ax],bins=int(dim[0]/nb),norm=LogNorm(),cmap=plt.cm.Purples)
	plt.hist2d(posg[ax[0]],posg[ax[1]],bins=int(dim[0]/nb),norm=LogNorm(),cmap=plt.cm.Blues)
	plt.clim(1, 5e3)
	plt.scatter(posh1[ax[0]],posh1[ax[1]],s=lr1**2*norm,ec='orange',fc='none',lw=1,zorder=3)
	plt.scatter(posh2[ax[0]],posh2[ax[1]],s=lr2**2*norm,ec='r',fc='none',lw=1,zorder=2)
	if len(pos1)>0:
		plt.scatter(*pos1.T[ax], marker='^', s=16, alpha=1, color='orange', zorder=4, label=r'$\tau<3\ \rm Myr$, '+redlab, lw=0.5, ec='k')
	if len(pos2)>len(pos1):
		plt.scatter(*pos2[len(pos1):].T[ax], marker='o', s=16, alpha=0.9, color='r', zorder=3, label=r'$\tau<{}\ \rm Myr$'.format(dt), lw=0.5, ec='k')
	plt.xlim(le[ax[0]], re[ax[0]])
	plt.ylim(le[ax[1]], re[ax[1]])
	plt.xlabel(llb[ax[0]])
	if nf<=1 or ax[1]!=ax0[1]:
		plt.ylabel(llb[ax[1]])
	plt.legend(loc=2)
	plt.tight_layout()
	if nf<=1:
		plt.savefig(rep+'popIIIpos_'+str(sn)+'_'+str(np.sum(ax))+'.png', dpi=300)
	else:
		plt.savefig(rep+'popIIIpos_'+str(sn)+'.png', dpi=300)
	plt.close()
	#print(lr2)
	#print(np.log10(ms2[0]))

def halomatch(hd0, ds, dt=10, fac=1, mlab=1, norm=1, m0=586, mmin=1e6, metal=0, gas=0, mode=0):
	z = ds['Redshift']
	h = ds['HubbleParam']
	ad = ds.all_data()
	dpos, dZ = popIII_pos(ds, ad, dt)
	pos1, pos2 = dpos
	lZ1, lZ2 = dZ
	nact = len(pos1)
	nall = len(pos2)
	lms1 = np.zeros((nact,9+metal*(4+2*gas)))
	lms2 = np.zeros((nall,9+metal*(4+2*gas)))
	if mlab==0:
		Mat = mmin #Mdown(z)
	else:
		Mat = Mup(z)
	lmh = hd0[0]*norm/h
	sel = lmh>Mat
	hd = hd0.T[sel]
	lmh = lmh[sel]
	hd = hd.T
	lr = hd[1]
	lpos = hd[2:5].T
	nh = len(lr)
	i = 0
	for pos in pos2:
		dis = np.array([np.linalg.norm(posh-pos) for posh in lpos])
		#dism = np.min(dis)
		#sel = dis==dism
		sel = dis<=lr
		sel_ = lmh[sel]==np.max(lmh[sel])
		posh = lpos[sel][sel_][0]
		dism = dis[sel][sel_][0]/lr[sel][sel_][0]
		ip = [lmh[sel][sel_][0], *lpos[sel][sel_][0], lr[sel][sel_][0], dism]
		sel0 = ip[0] == lms2[:,0]
		if np.sum(sel0)==0:
			ms = starcount(ds, lpos[sel][0], fac*lr[sel][0], dt, m0, metal, gas, mode)
			lms2[i] = np.hstack([ip, ms])
		else:
			print('repeat at {} logmh = {:.2f}'.format(i, np.log10(ip[0])))
			lms2[i] = lms2[sel0][0]
		if i<nact:
			lms1[i] += lms2[i]
		i += 1
	lms1, lms2 = lms1.T, lms2.T
	return lms1, lms2, np.vstack([pos1.T, lZ1]), np.vstack([pos2.T, lZ2])

def halo_popIII(hd0, ds, dt = 10, fac=1, mlab = 1, norm = 1, m0=586, mmin=1e6, metal=0, gas=0, mode=0):
	z = ds['Redshift']
	h = ds['HubbleParam']
	if mlab==0:
		Mat = mmin #Mdown(z)
	else:
		Mat = Mup(z)
	lmh = hd0[0]*norm/h
	sel = lmh>Mat
	hd = hd0.T[sel]
	hd = hd.T
	lr = hd[1]
	lpos = hd[2:5].T
	nh = len(lr)
	print('Nhalo_tot: {}'.format(nh))
	lms = np.zeros((nh,3+metal*(4+2*gas)))
	for i in range(nh):
		lms[i] = starcount(ds, lpos[i], fac*lr[i], dt, m0, metal, gas, mode)
		#print(lms[i])
		if i%100==0:
			print('{:.2f}% completed, z = {:.2f}'.format(i/nh*100, z))
	return np.vstack([lmh[sel], lms.T])

def focc(lms, ed, key, fac=0.5):
	lms = sorted(lms, key = lambda x: x[key])
	nb = len(ed)-1
	base = (ed[:-1]+ed[1:])*0.5
	count = 0
	nc = np.zeros(nb)
	mc = np.zeros(nb)
	nsc = np.zeros(nb)
	msc = np.zeros(nb)
	n3c1 = np.zeros(nb)
	n3c2 = np.zeros(nb)
	ms3c1 = np.zeros(nb)
	ms3c2 = np.zeros(nb)
	nf = len(lms[0])
	mtot = 0
	ntot = 0
	if nf>4:
		Zc = [[] for i in range(nb)]
	for x in lms:
		y = x[key]
		if count>=nb:
			break
		while y>ed[count+1]:
			count += 1
			#continue
		mtot += x[0]
		ntot += 1
		nc[count] += 1
		mc[count] += x[0] #* (x[0]>0)
		nsc[count] += x[1]>0
		msc[count] += x[1] #* (x[1]>0)
		n3c1[count] += x[2]>0
		ms3c1[count] += x[2] #* (x[2]>0)
		n3c2[count] += x[3]>0
		ms3c2[count] += x[3] #* (x[3]>0)
		if nf>4 and x[1]>0:
		#if nf>4:
			Zc[count].append(x[6]*x[4] + x[5]*x[7])
	sel0 = nc>0
	sel = nsc>0
	fsf = nsc[sel0]/nc[sel0]
	fmsf = msc[sel0]/mc[sel0]
	f310 = n3c1[sel0]/nc[sel0]
	f31 = n3c1[sel]/nsc[sel]
	f320 = n3c2[sel0]/nc[sel0]
	f32 = n3c2[sel]/nsc[sel]
	fm31 = ms3c1[sel]/msc[sel]
	fm32 = ms3c2[sel]/msc[sel]
	out = [fsf, f310, f31, fm31, f320, f32, fm32]
	if nf>4:
		#Zc = np.array(Zc)
		Zmean = np.zeros(nb)
		Zstd = np.zeros(nb)
		for i in range(nb):
			if len(Zc[i])<=0:
				continue
			Zmean[i] = np.sum(Zc[i])/msc[i]
			if len(Zc[i])>1:
				Zstd[i] = np.std(Zc[i])/msc[i] * nsc[i]
			else:
				Zstd[i] = Zmean[i]*fac
	d = {}
	d['b'] = [base, base[sel0], base[sel]]
	d['fsf'] = [fsf, fmsf]
	d['t1'] = [f310, f31, fm31]
	d['t2'] = [f320, f32, fm32]
	d['nc'] = [nc[sel0], nsc[sel]]
	if nf>4:
		d['Z'] = [Zmean, Zstd]
	lms = np.array(lms)
	lmh = lms.T[0]
	lmss = lms.T[1]
	sel = lmh >= 1.63e6
	e1 = np.sum(lmss[sel])/np.sum(lmh[sel])
	e2 = np.sum(lmss)/np.sum(lmh)
	sel_ = lmss>0
	e3 = np.sum(lmss[sel_])/np.sum(lmh[sel_])
	d['eta'] = [e1, e2, e3]#, np.sum(msc)/np.sum(mc)] #[np.sum(msc)/np.sum(mc), np.sum(msc[sel])/np.sum(mc[sel])]
	#print(len(lmh), ntot, mtot/np.sum(lmh), np.sum(mc)/np.sum(lmh), np.sum(msc)/np.sum(lmss))
	d['mass'] = [np.sum(lmss[sel]), np.sum(lmh[sel]), np.sum(lmss[sel_]), np.sum(lmh[sel_])]
	raw = [nc, mc, nsc, msc, n3c1, ms3c1, n3c2, ms3c2]
	d['raw'] = raw
	return d	

def popIIIocc(rep, lms0, z, sn, dt = 10, nb=30, fac = 0.5, edge=0.5, xflag=0):
	redlab = r'$z={:.1f}$'.format(z)
	mhd, mhu = np.min(lms0[0]), np.max(lms0[0])
	sel = lms0[1]>0
	msd, msu = np.min(lms0[1][sel]), np.max(lms0[1][sel])
	mhu *= 1.01
	msu *= 1.01
	mhed = np.geomspace(mhd, mhu, nb+1)
	#mhb = (mhed[:-1]+mhed[1:])*0.5
	msed = np.geomspace(msd, msu, nb+1)
	#msb = (msed[:-1]+msed[1:])*0.5
	lms = lms0.T
	print('Halo mass range: {:.3e} - {:.3e} Msun'.format(mhd, mhu))
	
	if xflag==0:
		dh = focc(lms, mhed, 0, fac)
		x1, x2 = mhd, mhu/edge
		dex = np.log10(mhu/mhd)/nb
		print('{:.3e} {:.3e} {:.3e} {:.3e}'.format(z, *dh['eta']))
		print(z, *dh['mass'])
	else:
		dh = focc(lms, msed, 1, fac)
		x1, x2 = msd*edge, msu/edge
		dex = np.log10(msu/msd)/nb
	y1, y2 = sumy(dh['fsf'][0])[:2]
	y1, y2 = y1*edge, y2/edge
	if xflag==0:
		plt.figure()
		plt.plot(dh['b'][1], dh['fsf'][0], label=redlab)
		plt.plot([Mup(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm atom}$')
		plt.ylabel(r'$f_{\rm SF}$')
		plt.legend()
		plt.tight_layout()
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim(y1, y2)
		plt.xlim(x1, x2)
		plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		plt.savefig(rep+'fSF_Mh_'+str(sn)+'.pdf')
		plt.close()
	
	if xflag>0:
		print('fSF = {:.2e}, eta = {:.2e} for M_star < {:.2e} Msun, M_halo > {:.2e} Msun'.format(dh['fsf'][0][0], dh['fsf'][1][0], msed[1], Mdown(z)))
	
	plt.figure()
	if xflag==0:
		y1, y2 = sumy(dh['fsf'][1])[:2]
		y1, y2 = y1*edge, y2/edge
		plt.plot(dh['b'][1], dh['fsf'][1], label=redlab)
		#plt.plot([Mdown(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm mol}$')
		plt.plot([Mup(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm atom}$')
	else:
		y1, y2 = sumy(dh['fsf'][1][1:])[:2]
		y1, y2 = y1*edge, y2/edge
		plt.plot(dh['b'][1][1:], dh['fsf'][1][1:], label=redlab)
	plt.ylabel(r'$\eta$')
	plt.legend()
	plt.tight_layout()
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(y1, y2)
	plt.xlim(x1, x2)
	if xflag==0:
		plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		plt.savefig(rep+'eta_Mh_'+str(sn)+'.pdf')
	else:
		plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$')
		plt.savefig(rep+'eta_Ms_'+str(sn)+'.pdf')
	plt.close()
	
	ylab = [r'$f_{\rm occ,all}$', r'$f_{\rm occ,SF}$', r'$f_{\rm mass}$']
	plt.figure(figsize=(12,4))
	for i in range(3):
		if i==0:
			ib = 1
		else:
			ib = 2
		plt.subplot(130+1*(i+1))
		f1 = dh['t1'][i]
		f2 = dh['t2'][i]
		n1 = dh['t1'][i>0]*dh['nc'][ib-1]
		n2 = dh['t2'][i>0]*dh['nc'][ib-1]
		if np.sum(f1>0)>0:
			y1 = np.min(f1[f1>0])*edge
		else:
			y1 = 1e-6
		if np.sum(f2>0)>0:
			y1 = min(y1, np.min(f2[f2>0])*edge)
			y2 = np.max(f2)/edge**2
		else:
			y2 = 2
		if i==0:
			y2 = y2/edge**2
		plt.errorbar(dh['b'][ib], f1, yerr=f1/n1**0.5, fmt='^', label=r'$\tau<3\ \rm Myr$')
		plt.errorbar(dh['b'][ib]*10**(dex/2), f2, yerr=f2/n2**0.5, fmt='o', label=r'$\tau<{}\ \rm Myr$'.format(dt))
		if xflag==0:
			#plt.plot([Mdown(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm mol}$')
			plt.plot([Mup(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm atom}$, '+redlab)
			plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		else:
			plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$')
		plt.ylabel(ylab[i])
		plt.xscale('log')
		plt.yscale('log')
		plt.tight_layout()
		if i==0:
			plt.legend(loc=2)
		plt.ylim(y1, y2)
		plt.xlim(x1, x2)
	if xflag==0:
		plt.savefig(rep+'focc_Mh_'+str(sn)+'.pdf')
	else:
		plt.savefig(rep+'focc_Ms_'+str(sn)+'.pdf')
	plt.close()
	
	#print(lms[0])
	if len(lms[0])>4:
		lZ = dh['Z'][0]
		y1, y2 = sumy(lZ[lZ>0])[:2]
		y1, y2 = y1*edge, y2/edge
		plt.figure()
		plt.errorbar(dh['b'][0], dh['Z'][0], yerr=dh['Z'][1], label=redlab)
		if xflag==0:
			#plt.plot([Mdown(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm mol}$')
			plt.plot([Mup(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm atom}$')
			plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		else:
			plt.xlabel(r'$M_{\star}\ [\rm M_{\odot}]$')
		plt.ylabel(r'$Z\ [\rm Z_{\odot}]$')
		plt.legend()
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(x1, x2)
		plt.ylim(y1, y2)
		plt.tight_layout()
		if xflag==0:
			plt.savefig(rep+'Z_Mh_'+str(sn)+'.pdf')
		else:
			plt.savefig(rep+'Z_Ms_'+str(sn)+'.pdf')
	#ds = focc(lms0, msed, 1)
	
def midbin(l):
	return (l[1:]+l[:-1])*0.5
	
def plotpopIII(rep, lms, z, sn, dt = 10, xflag = 0, edge=0.5, m0=586, n0=1, alp=0.7, gas=0, log=0):
	sel = lms[1]>0
	lms = lms.T[sel]
	redlab = r'$z={:.1f}$'.format(z)
	print('z = {:.3f}'.format(z))
	if len(lms[0])>4:
		Zbase = np.linspace(-6, 1, 36)
		if gas>0:
			#mh, ms, ms1, ms2, Zs, Zbh, Zg, ms0, mbh, mg = lms.T
			mh, ms, ms1, ms2, Zg, Zs, Zbh, mg, ms0, mbh = lms.T
			Zg[Zg<1e-6] = 1e-6
			plt.figure()
			plt.hist(np.log10(Zg), Zbase, alpha=alp, color='g', label='SF haloes, '+redlab)
			plt.hist(np.log10(Zg[ms1>0]), Zbase, label=r'$\tau<3\ \rm Myr$', histtype='step', lw=1.5, color='k')
			plt.hist(np.log10(Zg[ms2>0]), Zbase, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
			plt.xlabel(r'$\log(Z_{\rm gas}\ [\mathrm{Z_{\odot}}])$')
			plt.ylabel(r'Counts')
			if log>0:
				plt.yscale('log')
			plt.legend()
			plt.tight_layout()
			plt.savefig(rep+'Zg_dis_'+str(sn)+'.pdf')
			plt.close()
		else:
			mh, ms, ms1, ms2, Zs, Zbh, ms0, mbh = lms.T
		Zm = (Zs*ms0+Zbh*mbh)/(ms0+mbh)
		print('BH metallicity: ', sumy(Zbh))
		Zs[Zs<1e-6] = 1e-6
		plt.figure()
		plt.hist(np.log10(Zs), Zbase, alpha=alp, label='SF haloes, '+redlab, color='g')
		plt.hist(np.log10(Zs[ms1>0]), Zbase, label=r'$\tau<3\ \rm Myr$', histtype='step', lw=1.5, color='k')
		plt.hist(np.log10(Zs[ms2>0]), Zbase, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
		plt.xlabel(r'$\log(Z_{\star}\ [\mathrm{Z_{\odot}}])$')
		plt.ylabel(r'Counts')
		if log>0:
			plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'Zs_dis_'+str(sn)+'.pdf')
		plt.close()
		
		Zm[Zm<1e-6] = 1e-6
		plt.figure()
		if xflag==0:
			x1, x2 = np.min(mh)*edge, np.max(mh)/edge
			plt.loglog(mh, Zm, '.', label='SF haloes, '+redlab, color='g', alpha=0.5, zorder=1, markersize=3)
			#plt.scatter(mh[ms2>0], Zm[ms2>0], marker='o', c=np.log10(ms[ms2>0]), cmap=plt.cm.cool, label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=alp, zorder=2)
			#cb = plt.colorbar()
			#cb.ax.set_title(r'$\log(M_{\star}\ [\rm M_{\odot}])$')
			plt.scatter(mh[ms1>0], Zm[ms1>0], marker='^', c='k', label=r'$\tau<3\ \rm Myr$', alpha=1, zorder=3)
			plt.xlabel(r'$M_{\rm halo}\ \rm [M_{\odot}]$')
		else:
			#x1, x2 = np.min(ms)*edge, np.max(ms)/edge
			x1 = np.min(ms)*edge
			x2 = 3e11
			plt.loglog(ms, Zm, '.', color='g', alpha=0.5, zorder=1, markersize=3, label='SF haloes, '+redlab)
			#plt.scatter(ms[ms2>0], Zm[ms2>0], marker='o', c=np.log10(mh[ms2>0]), cmap=plt.cm.cool, alpha=alp, zorder=2, label=r'$\tau<{}\ \rm Myr$'.format(dt))#+', '+redlab, )
			#cb = plt.colorbar()
			#cb.ax.set_title(r'$\log(M_{\rm halo}\ [\rm M_{\odot}])$')
			plt.scatter(ms[ms1>0], Zm[ms1>0], marker='^', c='k', label=r'$\tau<3\ \rm Myr$', alpha=1, zorder=3)
			plotmmr()
			plt.xlabel(r'$M_{\star}\ \rm [M_{\odot}]$')
		plt.ylabel(r'$Z\ \rm [Z_{\odot}]$')
		lgnd = plt.legend(loc=4)
		#for lh in lgnd.legendHandles:
		#	lh._sizes = [64]
		plt.xlim(x1, x2)
		plt.tight_layout()
		if xflag==0:
			plt.savefig(rep+'Zsbh_Mh_'+str(sn)+'.pdf')
		else:
			plt.savefig(rep+'Zsbh_Ms_'+str(sn)+'.pdf')
		plt.close()
		
		plt.figure()
		plt.hist(np.log10(Zm), Zbase, alpha=alp, label='SF haloes, '+redlab, color='g')
		plt.hist(np.log10(Zm[ms1>0]), Zbase, label=r'$\tau<3\ \rm Myr$', histtype='step', lw=1.5, color='k')
		plt.hist(np.log10(Zm[ms2>0]), Zbase, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
		plt.xlabel(r'$\log(Z_{\star}\ [\mathrm{Z_{\odot}}])$')
		plt.ylabel(r'Counts')
		if log>0:
			plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'Zsbh_dis_'+str(sn)+'.pdf')
		plt.close()
		
		Zbh[Zbh<1e-8] = 1e-8
		
		plt.figure()
		if xflag==0:
			x1, x2 = np.min(mh)*edge, np.max(mh)/edge
			plt.loglog(mh, Zbh, '.', label='SF haloes, '+redlab, color='g', alpha=0.5, zorder=1, markersize=1)
			plt.scatter(mh[ms2>0], Zbh[ms2>0], marker='o', c=np.log10(ms[ms2>0]), cmap=plt.cm.cool, label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=alp, zorder=2)
			cb = plt.colorbar()
			cb.ax.set_title(r'$\log(M_{\star}\ [\rm M_{\odot}])$')
			plt.scatter(mh[ms1>0], Zbh[ms1>0], marker='^', c='k', label=r'$\tau<3\ \rm Myr$', alpha=1, zorder=3)
			plt.xlabel(r'$M_{\rm halo}\ \rm [M_{\odot}]$')
		else:
			#x1, x2 = np.min(ms)*edge, np.max(ms)/edge
			x1 = np.min(ms)*edge
			x2 = 3e11
			plt.loglog(ms, Zbh, '.', color='g', alpha=0.5, zorder=1, markersize=1)#, label='SF haloes, '+redlab)
			plt.scatter(ms[ms2>0], Zbh[ms2>0], marker='o', c=np.log10(mh[ms2>0]), cmap=plt.cm.cool, label=r'$\tau<{}\ \rm Myr$'.format(dt)+', '+redlab, alpha=alp, zorder=2)
			cb = plt.colorbar()
			cb.ax.set_title(r'$\log(M_{\rm halo}\ [\rm M_{\odot}])$')
			plt.scatter(ms[ms1>0], Zbh[ms1>0], marker='^', c='k', label=r'$\tau<3\ \rm Myr$', alpha=1, zorder=3)
			#plotmmr()
			plt.xlabel(r'$M_{\star}\ \rm [M_{\odot}]$')
		plt.ylabel(r'$Z_{\rm PopIII}\ \rm [Z_{\odot}]$')
		lgnd = plt.legend(loc=4)
		#for lh in lgnd.legendHandles:
		#	lh._sizes = [64]
		plt.xlim(x1, x2)
		plt.tight_layout()
		if xflag==0:
			plt.savefig(rep+'Zbh_Mh_'+str(sn)+'.pdf')
		else:
			plt.savefig(rep+'Zbh_Ms_'+str(sn)+'.pdf')
		plt.close()
		
		Zbase = np.linspace(-8, 1, 46)
		plt.figure()
		plt.hist(np.log10(Zbh), Zbase, alpha=alp, label='SF haloes, '+redlab, color='g')
		plt.hist(np.log10(Zbh[ms1>0]), Zbase, label=r'$\tau<3\ \rm Myr$', histtype='step', lw=1.5, color='k')
		plt.hist(np.log10(Zbh[ms2>0]), Zbase, label=r'$\tau<{}\ \rm Myr$'.format(dt), histtype='step', lw=1.5, color='r', ls='--')
		plt.xlabel(r'$\log(Z_{\star}\ [\mathrm{Z_{\odot}}])$')
		plt.ylabel(r'Counts')
		if log>0:
			plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'Zbh_dis_'+str(sn)+'.pdf')
		plt.close()
	else:
		mh, ms, ms1, ms2 = lms.T
	#print(ms2/ms)
	
	if xflag==0:
		xr = sumy(ms)[:2]
		xr[0] *= edge
		xr[1] /= edge
		bins = np.geomspace(*xr, 1001)
		his1, ed = np.histogram(ms[ms1>0], bins)
		his2, ed = np.histogram(ms[ms2>0], bins)
	else:
		xr = sumy(mh)[:2]
		xr[0] *= edge
		xr[1] /= edge
		bins = np.geomspace(*xr, 1001)
		his1, ed = np.histogram(mh[ms1>0], bins)
		his2, ed = np.histogram(mh[ms2>0], bins)		
	base = midbin(ed)
	cum1 = np.cumsum(his1)
	cum2 = np.cumsum(his2)
	y1, y2 = 0, 1
	plt.figure()
	plt.plot(base, cum1/cum1[-1], label=r'$\tau<3\ \rm Myr$')
	plt.plot(base, cum2/cum2[-1], '--', label=r'$\tau<{}\ \rm Myr$'.format(dt))
	plt.text(xr[1]/10, 0.1, redlab)
	plt.xscale('log')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	if xflag==0:
		plt.xlabel(r'$M_{\rm \star}\ [\rm M_{\odot}]$')
		plt.ylabel(r'$F(<M_{\rm \star})$')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'cumms_'+str(sn)+'.pdf')
		sel = base>1e4
		c1 = np.min(cum1[sel])/cum1[-1]
		c2 = np.min(cum2[sel])/cum2[-1]
	else:
		fit = curve_fit(linear, np.log10(base), cum1/cum1[-1])
		b0, b1 = fit[0]
		perr = np.sqrt(np.diag(fit[1]))
		f0 = lambda x: linear(np.log10(x), b0, b1)
		fup = lambda x: np.log10(x)*(b1+2*perr[1])+b0+2*perr[0]
		fdown = lambda x: np.log10(x)*(b1-2*perr[1])+b0-2*perr[0]
		#plt.plot(xr, f0(xr), 'k-.', label=r'Log-flat')
		plt.fill_between(xr, f0(xr)+0.2, f0(xr)-0.2, fc='gray', alpha=0.5, label=r'Log-flat')
		plt.plot([Mup(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm atom}$')
		plt.plot([Mreion(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm J,ion}$')
		plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		plt.ylabel(r'$F(<M_{\rm halo})$')
		plt.legend(loc=2)
		plt.tight_layout()
		plt.savefig(rep+'cummh_'+str(sn)+'.pdf')
		sel = base>Mup(z)
		c1 = np.min(cum1[sel])/cum1[-1]
		c2 = np.min(cum2[sel])/cum2[-1]
	plt.close()
	
	r1, r2 = ms1/ms, ms2/ms
	if np.sum(r1>0)>0:
		print('f_occ (<3 Myr): ', sumy(r1[r1>0]))
		y1 = np.min(r1[r1>0])*edge
	if np.sum(r2>0)>0:
		print('f_occ (<{} Myr): '.format(dt), sumy(r2[r2>0]))
		y1 = min(y1, np.min(r2[r2>0])*edge)
	else:
		y1 = 1e-5
	y2 = 1/edge
	nh = len(mh)
	mtot = np.sum(ms)
	print('Number of popIII hosts: {} (<3 Myr), {} (<10 Myr)'.format(np.sum(r1>0), np.sum(r2>0)))
	print('Occupation (mass) fraction (<3 Myr): {} ({})'.format(np.sum(ms1>0)/nh, np.sum(ms1)/mtot))
	#print(sumy(r1[r1>0]))
	print('Occupation (mass) fraction (<{} Myr): {} ({})'.format(dt, np.sum(ms2>0)/nh, np.sum(ms2)/mtot))
	#print(sumy(r2[r2>0]))
	def linear_(x, b1):
		b0 = -np.log10(m0*n0)*b1
		return x*b1 + b0
	plt.figure()
	if xflag==0:
		#if np.sum(r2>0)>0:
		#	xr = sumy(ms[r2>0])[:2]
		#else:
		xr = sumy(ms)[:2]
		xr[0] *= edge
		xr[1] /= edge
		fit0 = curve_fit(linear_, np.log10(ms[r1>0]), np.log10(r1[r1>0]))
		fit = curve_fit(linear, np.log10(ms[r2>0]), np.log10(r2[r2>0]))
		b10 = fit0[0][0]
		b0, b1 = fit[0]
		flab = r'$f_{\rm PopIII}\propto M_{\star}^{'+'{:.2f}'.format(b1)+'}$'
		flab0 = r'$f_{\rm PopIII}\propto M_{\star}^{'+'{:.2f}'.format(b10)+'}$'
		perr = np.sqrt(np.diag(fit[1]))
		#print(perr)
		fms00 = lambda x: 10**linear_(np.log10(x), b10) #10**(np.log10(x)*b10 + b00)
		#fms0 = lambda x: 10**linear_(np.log10(x), b1)
		fms0 = lambda x: 10**(np.log10(x)*b1 + b0)
		#fmsu = lambda x: 10**linear_(np.log10(x), b1+perr[0]*3)
		fmsu = lambda x: 10**(np.log10(x)*(b1+2*perr[1]) + b0+2*perr[0])
		#fmsd = lambda x: 10**linear_(np.log10(x), b1-perr[0]*3)
		fmsd = lambda x: 10**(np.log10(x)*(b1-2*perr[1]) + b0-2*perr[0])
		plt.plot(xr, fms0(xr), 'k-', label=flab)
		plt.plot(xr, fms00(xr), 'k--', label=flab0)
		plt.fill_between(xr, fmsd(xr), fmsu(xr), facecolor='gray', alpha=alp/2, zorder=0)
		plt.scatter(ms[r2>0], r2[r2>0], c=np.log10(mh[r2>0]), cmap=plt.cm.cool, 
		marker='o', label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=alp)
		cb = plt.colorbar()
		cb.ax.set_title(r'$\log(M_{\rm halo}\ [\rm M_{\odot}])$')
		plt.scatter(ms[r1>0], r1[r1>0], c='k', cmap=plt.cm.cool, 
		marker='^', label=r'$\tau<3\ \rm Myr$', alpha=1)
		plt.xlabel(r'$M_{\rm \star}\ [\rm M_{\odot}]$')
	else:
		#if np.sum(r2>0)>0:
		#	xr = sumy(mh[r2>0])[:2]
		#else:
		xr = sumy(mh)[:2]
		xr[0] *= edge
		xr[1] /= edge
		xr[0] = min(xr[0], Mdown(z)*edge)
		plt.scatter(mh[r2>0], r2[r2>0], c=np.log10(ms[r2>0]), cmap=plt.cm.cool, 
		marker='o', label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=alp)
		cb = plt.colorbar()
		cb.ax.set_title(r'$\log(M_{\rm \star}\ [\rm M_{\odot}])$')
		plt.scatter(mh[r1>0], r1[r1>0], c='k', cmap=plt.cm.cool, 
		marker='^', label=r'$\tau<3\ \rm Myr$', alpha=1)
		plt.plot([Mdown(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm mol}$')
		plt.plot([Mup(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm atom}$')
		plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$f_{\mathrm{PopIII}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	plt.text(xr[1]/10, y2/3, redlab)
	plt.legend(loc=3)
	plt.tight_layout()
	if xflag==0:
		plt.savefig(rep+'fpopIII_Ms_'+str(sn)+'.pdf')
	else:
		plt.savefig(rep+'fpopIII_Mh_'+str(sn)+'.pdf')
	plt.close()
	
	y1, y2 = np.min(ms2[ms2>0])*edge, np.max(ms2)/edge**3
	plt.figure()
	if xflag==0:
		print('Average Pop III mass for tau < 3 (10) Myr: {:.2f} ({:.2f})'.format(np.mean(ms1[ms1>0]), np.mean(ms2[ms2>0])))
		flab = r'$M_{\rm PopIII}\propto M_{\star}^{'+'{:.2f}'.format(b1+1)+'}$'
		flab0 = r'$M_{\rm PopIII}\propto M_{\star}^{'+'{:.2f}'.format(b10+1)+'}$'
		plt.plot(xr, xr*fms0(xr), 'k-', label=flab)
		#plt.plot(xr, xr*fms00(xr), 'k--', label=flab0)
		plt.fill_between(xr, xr*fmsd(xr), xr*fmsu(xr), facecolor='gray', alpha=alp/2, zorder=0)
		plt.scatter(ms[r2>0], ms2[r2>0], c=np.log10(mh[r2>0]), cmap=plt.cm.cool, 
		marker='o', label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=alp)
		cb = plt.colorbar()
		cb.ax.set_title(r'$\log(M_{\rm halo}\ [\rm M_{\odot}])$')
		plt.scatter(ms[r1>0], ms1[r1>0], c='k', cmap=plt.cm.cool, 
		marker='^', label=r'$\tau<3\ \rm Myr$', alpha=1)
		plt.xlabel(r'$M_{\rm \star}\ [\rm M_{\odot}]$')
	else:
		fit = curve_fit(linear, np.log10(mh[r2>0]), np.log10(ms2[ms2>0]))
		b0, b1 = fit[0]
		perr = np.sqrt(np.diag(fit[1]))
		flab = r'$M_{\rm PopIII}\propto M_{\rm halo}^{'+'{:.2f}'.format(b1)+'}$'
		fms0 = lambda x: 10**(np.log10(x)*b1 + b0)
		fmsu = lambda x: 10**(np.log10(x)*(b1+1*perr[1]) + b0+1*perr[0])
		fmsd = lambda x: 10**(np.log10(x)*(b1-1*perr[1]) + b0-1*perr[0])
		plt.plot(xr, fms0(xr), 'k-', label=flab)
		plt.fill_between(xr, fmsd(xr), fmsu(xr), facecolor='gray', alpha=alp/2, zorder=0)
		plt.scatter(mh[r2>0], ms2[r2>0], c=np.log10(ms[r2>0]), cmap=plt.cm.cool, 
		marker='o', label=r'$\tau<{}\ \rm Myr$'.format(dt), alpha=alp)
		cb = plt.colorbar()
		cb.ax.set_title(r'$\log(M_{\rm \star}\ [\rm M_{\odot}])$')
		plt.scatter(mh[r1>0], ms1[r1>0], c='k', cmap=plt.cm.cool, 
		marker='^', label=r'$\tau<3\ \rm Myr$', alpha=1)
		plt.xlabel(r'$M_{\rm halo}\ [\rm M_{\odot}]$')
		#plt.plot([Mdown(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm th}^{\rm mol}$')
		plt.plot([Mup(z)]*2, [y1, y2], 'k:', label=r'$M_{\rm th}^{\rm atom}$')
		plt.plot([Mreion(z)]*2, [y1, y2], 'k-.', label=r'$M_{\rm J,ion}$')
	plt.ylabel(r'$M_{\mathrm{PopIII}}\ [\rm M_{\odot}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(xr)
	plt.ylim(y1, y2)
	plt.text(xr[1]/10, y2/2, redlab)
	plt.legend(loc=2)
	plt.tight_layout()
	if xflag==0:
		plt.savefig(rep+'MpopIII_Ms_'+str(sn)+'.pdf')
	else:
		plt.savefig(rep+'MpopIII_Mh_'+str(sn)+'.pdf')
	plt.close()
	return c1, c2
	
def loadhcat(sn, rep, metal = 1, gas = 0):
	fdata0 = rep+'popIII_'+str(sn)+'_0.txt'
	fdata1 = rep+'popIII_'+str(sn)+'_1.txt'
	fdata2 = rep+'popIII_'+str(sn)+'_1_full.txt'
	if gas>0:
		if os.path.exists(fdata2):
			lms = np.array(retxt(fdata2, 4+(4+2*gas), 0,0))
			return lms
	if os.path.exists(fdata1) and metal>0:
		lms = np.array(retxt(fdata1, 4+4, 0,0))
	elif os.path.exists(fdata0):
		lms = np.array(retxt(fdata0, 4, 0,0))
	elif metal==0 and os.path.exists(fdata1):
		lms = np.array(retxt(fdata1, 4+4, 0,0))
	else:
		print('Missing halo catalog for snapshot {}'.format(sn))
		lms = []
	return lms
	
	
#raw = [nc, mc, nsc, msc, n3c1, ms3c1, n3c2, ms3c2]
def main(lsn, lz, mr = [5e5, 5e10], key = 0, rep = './', nb = 30, log=0, pflag=0, bd = 0, metal=0):
	if len(lsn)!=len(lz):
		print('Miss match of snapshots and redshift bins!')
		return []
	med = np.geomspace(*mr, nb+1)
	mb = (med[1:]+med[:-1])*0.5
	if log==0:
		X, Y = np.meshgrid(mb, lz, indexing = 'ij')
	else:
		X, Y = np.meshgrid(mb, np.log10(lz+1), indexing = 'ij')
	eta = np.zeros(X.shape)
	SFocc = np.zeros(X.shape)
	f310, f31, fm31 = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
	f320, f32, fm32 = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
	met = []
	zmet = []
	indmet = []
	frac1 = np.zeros(len(lz))
	frac2 = np.zeros(len(lz))
	occ1 = np.zeros((5, len(lz)))
	occ2 = np.zeros((5, len(lz)))
	lfmol = np.zeros((4, len(lz)))
	#print(X.shape)
	i = 0
	for sn in lsn:
		lms = loadhcat(sn, rep, metal)
		nf = lms.shape[0]
		z = lz[i]
		if pflag>0:
			popIIIocc(rep, lms, z, sn, dt = 10, nb = 20, xflag=0)
			popIIIocc(rep, lms, z, sn, dt = 10, nb = 20, xflag=1)
			cs1, cs2 = plotpopIII(rep, lms, z, sn, dt = 10, xflag=0, gas=0, log=log)
			ch1, ch2 = plotpopIII(rep, lms, z, sn, dt = 10, xflag=1, gas=0, log=log)
			lfmol[0][i] = ch1
			lfmol[1][i] = ch2
			lfmol[2][i] = cs1
			lfmol[3][i] = cs2
		
		if bd>0:
			mh = lms[0]
			sel = mh>Mdown(z)
			lms = lms.T[sel]
			lms = lms.T
		
		mh, ms, m31, m32 = lms[:4]
		frac1[i] = np.sum(m31[m31>0]/ms[m31>0]==1)/np.sum(m31>0)
		frac2[i] = np.sum(m32[m32>0]/ms[m32>0]==1)/np.sum(m32>0)
		
		selatm = mh > Mup(z)
		selion = mh > Mreion(z)
		
		occ1[0][i] = np.sum(m31>0)/len(mh)
		occ1[1][i] = np.sum(m31>0)/np.sum(ms>0)
		occ1[2][i] = np.sum(m31)/np.sum(ms)
		occ1[3][i] = np.sum(m31[selatm]>0)/np.sum(ms[selatm]>0)
		occ1[4][i] = np.sum(m31[selion]>0)/np.sum(ms[selion]>0)
		
		occ2[0][i] = np.sum(m32>0)/len(mh)
		occ2[1][i] = np.sum(m32>0)/np.sum(ms>0)
		occ2[2][i] = np.sum(m32)/np.sum(ms)
		occ2[3][i] = np.sum(m32[selatm]>0)/np.sum(ms[selatm]>0)
		occ2[4][i] = np.sum(m32[selion]>0)/np.sum(ms[selion]>0)
		
		docc = focc(lms.T, med, key)
		nc, mc, nsc, msc, n3c1, ms3c1, n3c2, ms3c2 = docc['raw']
		sels = nsc>0
		sel = nc>0
		eta[sel,i] = msc[sel]/mc[sel]
		SFocc[sels,i] = nsc[sels]/nc[sels]
		f310[sel,i] = n3c1[sel]/nc[sel]
		f31[sels,i] = n3c1[sels]/nsc[sels]
		fm31[sels,i] = ms3c1[sels]/msc[sels]
		f320[sel,i] = n3c2[sel]/nc[sel]
		f32[sels,i] = n3c2[sels]/nsc[sels]
		fm32[sels,i] = ms3c2[sels]/msc[sels]
		if nf>4:
			zmet.append(lz[i])
			met.append(docc['Z'])
			indmet.append(i)
		i += 1
	d = {}
	d['map'] = X, Y, eta, SFocc
	d['p31'] = f310, f31, fm31
	d['p32'] = f320, f32, fm32
	d['met'] = np.array([zmet, met, indmet])
	d['mr'] = mr
	d['mb'] = mb
	d['lz'] = lz
	d['frac'] = [frac1, frac2]
	d['occ'] = [occ1, occ2]
	d['fmol'] = lfmol
	return d
		
def plotmap(X, Y, var, rep, fn, lab, nbin, key, log, cmap, lxlab, lylab, lzf):
	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(var), nbin, cmap=cmap)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(lab,size=12)
	if log==0:
		ztp = lzf
	else:
		ztp = np.log10(lzf+1)
	if key==0:
		plt.plot(Mup(lzf), ztp, 'k-.', label=r'$M_{\rm th}^{\rm atom}$')
	plt.xscale('log')
	plt.xlabel(lxlab[key])
	plt.ylabel(lylab[log])
	plt.tight_layout()
	if key==0:
		plt.legend()
		plt.savefig(rep+fn+'_z_Mh.pdf')
	else:
		plt.savefig(rep+fn+'_z_Ms.pdf')
	plt.close()

def flow_z(z, plat1=0.05, plat2=0.2, z0=6, z1=7.5, z2=12.5, z3 = 19.0):
	y = 1.0*(z>z3)
	y += (plat1 + (z-z2)*(1-plat1)/(z3-z2)) * (z<=z3)*(z>z2)
	y += plat1 * (z<=z2)*(z>z1)
	y += (plat2 + (z-z0)*(plat1-plat2)/(z1-z0))* (z<=z1)*(z>z0)
	y += plat2 * (z<=z0)#*(z>z0)
	return y

def plotfracz(rep, lz, y1, y2, ylab, fn, log, dt=10):
	plt.figure()
	plt.plot(lz, y1, label=r'$\tau<3\ \rm Myr$')#, color='orange')
	plt.plot(lz, y2, '--', label=r'$\tau<{}\ \rm Myr$'.format(dt))#, color='r')
	if fn is 'fiso':
		lzz = np.linspace(0, 21, 1000)
		flow = flow_z(lzz)
		plt.plot(lz, llow1, ls='-.', label=r'$f_{\rm mol}^{\rm sim}(\tau<3\ \rm Myr)$', color='orange')
		plt.plot(lz, llow2, 'r:', label=r'$f_{\rm mol}^{\rm sim}'+r'(\tau<{}\ \rm Myr)$'.format(dt))
		plt.plot(lzz, flow, 'k-', lw=4.5, alpha = 0.5, label=r'$f_{\rm mol}^{\rm sim}$ (smoothed)')
		plt.xlim(4, 20)
		plt.ylim(0, 1.0)
	plt.xlabel(r'$z$')
	plt.ylabel(ylab)
	plt.legend()
	if log>0:
		plt.yscale('log')
	#else:
	#	plt.ylim(0, min(np.max(y2)*1.1, 1.0))
	plt.tight_layout()
	plt.savefig(rep+fn+'_z.pdf')
	plt.close()

def plotiso(rep, lz, d, log, dt=10):
	y1, y2, y3, y4 = d
	plt.figure()
	plt.plot(lz, y3, ls='-', label=r'$f_{\rm mol}(\tau<3\ \rm Myr)$', color='orange')
	plt.plot(lz, y4, 'r--', label=r'$f_{\rm mol}'+r'(\tau<{}\ \rm Myr)$'.format(dt))
	plt.plot(lz, y1, '-.', label=r'$f_{\rm new}(\tau<3\ \rm Myr)$', color='orange')
	plt.plot(lz, y2, 'r:', label=r'$f_{\rm new}'+r'(\tau<{}\ \rm Myr)$'.format(dt))#, color='r')
	lzz = np.linspace(0, 21, 1000)
	flow = flow_z(lzz)
	plt.plot(lzz, flow, 'k-', lw=4.5, alpha = 0.5, label=r'$f_{\rm mol}$ (smoothed)')
	plt.xlim(4, 20)
	plt.ylim(0, 1.0)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm mol}$ ($f_{\rm new}$)')
	plt.legend(loc=4)
	if log>0:
		plt.yscale('log')
	#else:
	#	plt.ylim(0, min(np.max(y2)*1.1, 1.0))
	plt.tight_layout()
	plt.savefig(rep+'fiso_z.pdf')
	plt.close()

def plotocc(rep, lz, d, log, dt=10, mode=0):
	y1, y2, y3, y4, nh, nh_ = d
	sel1 = lz>9.5
	sel2 = lz<6.5
	if mode==0:
		fpre1 = np.mean(y1[sel1])
		fpre2 = np.mean(y2[sel1])
		fpos1 = np.mean(y3[sel2])
		fpos2 = np.mean(y4[sel2])
	else:
		fpre1 = np.sum(y1[sel1]*nh[sel1])/np.sum(nh[sel1])
		fpre2 = np.sum(y2[sel1]*nh[sel1])/np.sum(nh[sel1])
		fpos1 = np.sum(y3[sel2]*nh_[sel2])/np.sum(nh_[sel2])
		fpos2 = np.sum(y4[sel2]*nh_[sel2])/np.sum(nh_[sel2])
	print('Pop III host fractions: {:.2e}, {:.2e} (pre-reion), {:.2e}, {:.2e} (post-reion)'.format(fpre1, fpre2, fpos1, fpos2))
	plt.figure()
	plt.plot(lz, y1, label=r'$\tau<3\ \rm Myr$ ($M_{\rm halo}>M_{\rm th}^{\rm atom}$)', color='orange')
	plt.plot(lz, y2, '--', label=r'$\tau<{}\ \rm Myr$'.format(dt)+r' ($M_{\rm halo}>M_{\rm th}^{\rm atom}$)', color='r')
	if mode==0:
		f1, f2 = '^', 'o'
	else:
		f1, f2 = '-.', ':'
	plt.plot(lz, y3, f1, label=r'$\tau<3\ \rm Myr$ ($M_{\rm halo}>M_{\rm J,ion}$)', color='orange')
	plt.plot(lz, y4, f2, label=r'$\tau<{}\ \rm Myr$'.format(dt)+r' ($M_{\rm halo}>M_{\rm J,ion}$)', color='r')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm occ,PopIII}$')
	plt.legend()
	if log>0:
		plt.yscale('log')
	#else:
	#	plt.ylim(0, min(np.max(y2)*1.1, 1.0))
	plt.tight_layout()
	plt.savefig(rep+'focc_z.pdf')
	plt.close()

lls = ['-', '--', '-.', ':']
llw = [1.5]*4 + [3]*4
lls = lls*2
		
mhr = [4e5, 4e10]
msr = [5e2, 3e8]
lsn = np.array([1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 22, 23], dtype='int')
lz = np.array(retxt('zbase_box.txt',1,0,0)[0])
dlow = np.array(retxt('fmol.txt',4)) #[0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.2, 0.2]
llow1 = dlow[0] #(dlow[0] + dlow[1])*0.5
llow2 = dlow[1]
#llow[12] = 0.325
#llow[6] = 0.507
#llow[8] = 0.487
#llow[10] = 0.4
#lzsn = lz
#lzsn = [x for x in lzsn]# + [3.5, 3., 2, 1, 0]
#flow_z_sim = interp1d(lzsn, llow)#np.hstack([llow, [0.1, 0, 0, 0, 0]]))
#print('redshift list: ', lz)
#print(llow)
		
if __name__=='__main__':
	mode = 1
	sfr = 0
	mflag = 1
	logflag = 1
	ncore = 1
	sn = 23
	#snind = [9, 10, 11, 12, 13]
	snind = range(14) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	rest = 5
	ctflag = 0
	proflag = 1
	rep = 'rock_box/'
	#rep = 'rock_zoom/'
	#rep = 'rock_zoom_HR/'
	#rep = 'caesar/'
	hc = 'halos_0.0_box_'+str(sn)+'.ascii'
	#hc = 'halos_0.0_zoom.ascii'
	#hc = 'halos_0.0_zoom_HR.ascii'
	nf = 11
	#if mode==0 or mflag==0:
	halocata = rockstar(np.array(retxt(hc,nf,19,0)))
		#halocata = caesar_obj(recaesar(sn), 0)
	
	datarep = '/media/friede/Seagate Backup Plus Drive/HighzBH/FDbox_Lseed/'
	
	#fname = '~/Documents/SimulationCodes/FDzoom_Hseed/snapshot_'+str(sn).zfill(3)+'.hdf5'
	#fname = '~/Documents/SimulationCodes/FDNzoom_Hseed_UVB/snapshot_'+str(sn).zfill(3)+'.hdf5'
	#fname = '~/Documents/SimulationCodes/FDbox_Lseed/snapshot_'+str(sn).zfill(3)+'.hdf5'
	fname = datarep+'snapshot_'+str(sn).zfill(3)+'.hdf5'
	#fname = '~/Documents/SimulatioCodes/FDzoom_Lseed_HR/snapshot_'+str(sn).zfill(3)+'.hdf5'
	
	if not os.path.exists(rep):
		os.makedirs(rep)
	
	#"""
	ds = yt.load(fname)
	z = ds['Redshift']
	metal = 1
	# metal info available for snapshot 23, 18, 7, 5-1
	gas = 1
	fdata = rep+'popIII_'+str(sn)+'_'+str(metal)+'.txt'
	fdata1 = rep+'popIII1_'+str(sn)+'_'+str(metal)+'.txt'
	fdata2 = rep+'popIII2_'+str(sn)+'_'+str(metal)+'.txt'
	pdata1 = rep+'popIIIpos1_'+str(sn)+'_'+str(metal)+'.txt'
	pdata2 = rep+'popIIIpos2_'+str(sn)+'_'+str(metal)+'.txt'
	ext = 1
	if mflag==0:
		mmin = min(Mdown(z), 1e6)
		lms1, lms2, pos1, pos2 = halomatch(halocata, ds, dt = 10, fac = 1, mlab = 0, norm=1, 
			m0=586, mmin=mmin, metal=metal, gas=gas, mode=sfr)
		totxt(fdata1, lms1)
		totxt(fdata2, lms2)
		totxt(pdata1, pos1)
		totxt(pdata2, pos2)
		hostdis(rep, sn, halocata, lms1, lms2, ds, dt=10, nb=50)
		#if proflag>0:
		promatch(rep, sn, lms1, lms2, pos1[:3].T, pos2[:3].T, ds, dt=10, ext0=ext, fac=1.1, mode=0, nump=7e7)
	else:
		lms1 = np.array(retxt(fdata1, 9+metal*(4+2*gas)))
		lms2 = np.array(retxt(fdata2, 9+metal*(4+2*gas)))
		pos1 = np.array(retxt(pdata1, 4))
		pos2 = np.array(retxt(pdata2, 4))
		hostdis(rep, sn, halocata, lms1, lms2, ds, dt=10, nb=50)
		if proflag>0:
			promatch(rep, sn, lms1, lms2, pos1[:3].T, pos2[:3].T, ds, dt=10, ext0=ext, fac=1.1, mode=0, nump=7e7)
	gas_ = gas
	if mflag>0:
		lsn_ = lsn[snind]
		lfd1 = [rep+'popIII1_'+str(sn)+'_'+str(metal)+'.txt' for sn in lsn_]
		lfd2 = [rep+'popIII2_'+str(sn)+'_'+str(metal)+'.txt' for sn in lsn_]
		dms1 = [retxt(fd1, 9+metal*(4+2*gas)) for fd1 in lfd1]
		dms2 = [retxt(fd2, 9+metal*(4+2*gas)) for fd2 in lfd2]
		if logflag>0:
			lds = [yt.load(datarep+'snapshot_'+str(sn).zfill(3)+'.hdf5') for sn in lsn_]
			lhd = [rockstar(np.array(retxt('halos_0.0_box_'+str(sn)+'.ascii',nf,19,0))) for sn in lsn_]
			dhost = [hostdis(rep, sn, hd, ms1, ms2, ds, dt=10, nb=50) for sn, hd, ms1, ms2, ds in zip(lsn_, lhd, dms1, dms2, lds)]
			docc = np.array([x[0] for x in dhost]).T
			dfiso = np.array([x[1] for x in dhost]).T
			dfisom = np.array([x[2] for x in dhost]).T
			print('Fraction of new SF haloes in minihaloes: {:.2f}, {:.2f}'.format(np.sum(dfisom[0])/np.sum(dfisom[1]),np.sum(dfisom[2])/np.sum(dfisom[3])))
			plotocc(rep, lz[snind], docc, 1, mode=0)
			plotiso(rep, lz[snind], dfiso, 0)
		lpd1 = [rep+'popIIIpos1_'+str(sn)+'_'+str(metal)+'.txt' for sn in lsn_]
		lpd2 = [rep+'popIIIpos2_'+str(sn)+'_'+str(metal)+'.txt' for sn in lsn_]
		dpos1 = [retxt(pd1, 4) for pd1 in lpd1]
		dpos2 = [retxt(pd2, 4) for pd2 in lpd2]
		#if proflag>0:
		#	lds = [yt.load(datarep+'snapshot_'+str(sn).zfill(3)+'.hdf5') for sn in lsn_]
		#	[promatch(rep, sn, ms1, ms2, pos1[:3].T, pos2[:3].T, ds, dt=10, ext0=ext, fac=1.1, mode=0, nump=7e7) for sn, ms1, ms2, pos1, pos2, ds in zip(lsn_, dms1, dms2, dpos1, dpos2, lds)]
		if rest>0:
			zr = sumy(lz[snind][-rest:])[:2]
			lms1 = np.hstack(dms1[-rest:])
			lms2 = np.hstack(dms2[-rest:])
			lpos1 = np.hstack(dpos1[-rest:])
			lpos2 = np.hstack(dpos2[-rest:])
			repo = rep
		else:
			zr = sumy(lz[snind])[:2]
			lms1 = np.hstack(dms1)
			lms2 = np.hstack(dms2)
			lpos1 = np.hstack(dpos1)
			lpos2 = np.hstack(dpos2)
			repo = rep+'all/'
		if not os.path.exists(repo):
			os.makedirs(repo)
		hostscatter(repo, lms1, lms2, lpos1, lpos2, zr, dt=10, nb = 30, gas=gas_)
	gas = 0
	if metal>0:
		if gas>0:
			fdata = rep+'popIII_'+str(sn)+'_'+str(metal)+'_full.txt'
	if mode==0:
		mmin = min(Mdown(z), 1e6) #Mup(z)
		lms = halo_popIII(halocata, ds, dt = 10, fac = 1, mlab = 0, norm=1, 
			m0=586, mmin=mmin, metal=metal, gas=gas, mode=sfr)
		totxt(fdata, lms, 0,0,0)
	if os.path.exists(rep+'popIII_'+str(sn)+'_'+str(metal)+'.txt'):
		lms = np.array(retxt(fdata, 4+metal*(4+2*gas), 0,0))
	elif os.path.exists(rep+'popIII_'+str(sn)+'_0.txt'):
		fdata = rep+'popIII_'+str(sn)+'_0.txt'
		lms = np.array(retxt(fdata, 4, 0,0))
	else:
		print('No data file')
		exit()
	log = 1
	#popIIIocc(rep, lms, z, sn, dt = 10, nb = 20, xflag=0)
	#popIIIocc(rep, lms, z, sn, dt = 10, nb = 20, xflag=1)
	plotpopIII(rep, lms, z, sn, dt = 10, xflag=0, gas=gas, log=log)
	plotpopIII(rep, lms, z, sn, dt = 10, xflag=1, gas=gas, log=log)
	#"""
	if ctflag==0:
		exit()
	
	#"""
	lxlab = [r'$M_{\rm halo}\ [\rm M_{\odot}]$', r'$M_{\star}\ [\rm M_{\odot}]$']
	lylab = ['$z$', '$\log(z+1)$']
	
	nb = 28
	pflag = 0
	key = 0
	log = 0
	bd = 0
	metflag = 1
	if key==0:
		mr = mhr
	else:
		mr = msr
	d = main(lsn, lz, mr, key, rep, nb, log, pflag, bd, metflag)
	X, Y, eta, SFocc = d['map']
	f310, f31, fm31 = d['p31']
	f320, f32, fm32 = d['p32']
	lzm0, lmet0, snmet = d['met']
	if pflag>0:
		fmol = d['fmol']
		totxt('fmol.txt', fmol)
	#print(snmet)
	#lzsel = np.array([1, 3, 5, 7, 8, 9, 11, 13])
	#lzsel = np.array([1, 3, 5, 7, 9, 11, 13])
	lzsel = np.array([0, 1, 2, 3, 5, 7, 9, 13])
	sel = [np.sum(lzsel==snm)>0 for snm in snmet]
	#print(sel)
	lzm = lzm0[sel]
	lmet = lmet0[sel]
	#print(lzm)
	frac1, frac2 = d['frac']
	occ1, occ2 = d['occ']
	mr = d['mr']
	mb = d['mb']
	
	dex = np.log10(mr[1]/mr[0])/nb
	plt.figure(figsize=(9,4))
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	zlm = 0
	y1, y2 = 1e-5, 1
	count = 0
	for i in range(len(lzm)):
		redlab = r'$z={:.1f}$'.format(lzm[i])
		zmean, zstd = lmet[i]
		#plt.errorbar(mb*10**(i*dex/8), zmean, yerr=zstd, label=redlab, ls=lls[count])
		if lzm[i]<zlm:
			ax1.loglog(mb[zmean>0], zmean[zmean>0], label=redlab, ls=lls[count], lw=llw[count])
			ax2.loglog(mb[zstd>0], zstd[zstd>0], ls=lls[count], lw=llw[count])
		else:
			ax1.loglog(mb[zmean>0], zmean[zmean>0], ls=lls[count], lw=llw[count])
			ax2.loglog(mb[zstd>0], zstd[zstd>0], label=redlab, ls=lls[count], lw=llw[count])
		count += 1
	if key==0:
		zu, zd = np.max(lzm), np.min(lzm)
		ax1.fill_between([Mup(zu), Mup(zd)], [y1, y1], [y2, y2], fc='gray', label=r'$M_{\rm th}^{\rm atom}$', alpha=0.5)
		ax2.fill_between([Mup(zu), Mup(zd)], [y1, y1], [y2*10, y2*10], fc='gray', alpha=0.5)
	else:
		lm1 = np.geomspace(1e8, 2e11, 100)
		lm2 = np.geomspace(1e4, 1e11, 100)
		lm3 = np.geomspace(Ms_MV(0), Ms_MV(-14), 100)
		lZref31 = Z_ms_local(lm3, 1)/10**dex1
		lZref32 = Z_ms_local(lm3, -1)*10**dex1
		zind1, zind2 = 0, 4
		zref11, zref12 = lz_mmr[zind1], lz_mmr[zind2] #3.5
		zref21, zref22 = 0.0, 6.0
		lZref11 = Z_ms_obs(lm1, zind1)
		lZref12 = Z_ms_obs(lm1, zind2)
		lZref21 = Z_ms_fire_star(lm2, zref21)
		lZref22 = Z_ms_fire_star(lm2, zref22)
		#plt.plot(lm1, lZref11, color='k', label=r'AMAZE (MR08), $z={:.1f}$'.format(zref1), ls=':', lw=3, zorder=0)#ls=(0, (5, 10)))
		ax1.fill_between(lm1, lZref11, lZref12, fc='r', label=r'AMAZE, $z\sim {:.1f}-{:.1f}$'.format(zref11, zref12), alpha=0.3, zorder=0)
		ax1.fill_between(lm2, lZref21, lZref22, fc='b', label=r'FIRE, $z\sim {:.0f}-{:.0f}$'.format(zref21, zref22), alpha=0.3, zorder=0)
		ax1.fill_between(lm3, lZref31, lZref32, fc='orange', label=r'LGDFs, $z\sim 0$', alpha=0.3, zorder=0)
	ax1.set_xlabel(lxlab[key])
	ax2.set_xlabel(lxlab[key])
	ax1.set_ylabel(r'$\langle Z\rangle\ [\rm Z_{\odot}]$')
	ax2.set_ylabel(r'$\delta Z\ [\rm Z_{\odot}]$')
	if key==0:
		ax1.set_xlim(mr)
	else:
		x1 = np.min(mr)
		x2 = 3e11
		ax1.set_xlim(x1, x2)
	ax2.set_xlim(mr)
	ax1.set_ylim(y1, y2)
	ax2.set_ylim(y1, y2)
	ax1.legend(ncol=1, loc=4)
	ax2.legend(ncol=2)
	plt.tight_layout()
	if key==0:
		plt.savefig(rep+'Z_Mh.pdf')
	else:
		plt.savefig(rep+'Z_Ms.pdf')
	plt.close()
	
	
	z1, z2 = np.min(lz), np.max(lz)
	lzf = np.linspace(z1, z2, 100)
	if log==0:
		ztp = lzf
		rep += 'linear/'
	else:
		ztp = np.log10(lzf+1)
		rep += 'log/'
	if not os.path.exists(rep):
		os.makedirs(rep)

	#plotfracz(rep, lz, frac1, frac2, r'$f_{\rm PopIII,new}$', 'fiso', log)
	plotfracz(rep, lz, occ1[0], occ2[0], r'$\langle f_{\rm occ,all}\rangle$', 'f3A', log)
	plotfracz(rep, lz, occ1[1], occ2[1], r'$\langle f_{\rm occ, SF}\rangle$', 'f3SF', log)
	plotfracz(rep, lz, occ1[2], occ2[2], r'$\langle f_{\rm mass} \rangle$', 'fmass', log)
	plotocc(rep, lz, [occ1[3], occ2[3], occ1[4], occ2[4]], 1)
	
	cmap1, cmap2, cmap3 = plt.cm.cool, plt.cm.winter, plt.cm.gnuplot2
	nbin1 = np.linspace(-4.5, 0, 91)
	nbin2 = np.linspace(-3, 0, 61)
	nbin3 = np.linspace(-6.5, 0, 66)
	plotmap(X, Y, f310, rep, 'f31A', r'$\log(f_{\rm occ,all})$', nbin1, key, log, cmap1, lxlab, lylab, lzf)
	plotmap(X, Y, f31, rep, 'f31SF', r'$\log(f_{\rm occ,SF})$', nbin2, key, log, cmap2, lxlab, lylab, lzf)
	plotmap(X, Y, fm31, rep, 'fmass1', r'$\log(f_{\rm mass})$', nbin3, key, log, cmap3, lxlab, lylab, lzf)
	plotmap(X, Y, f320, rep, 'f32A', r'$\log(f_{\rm occ,all})$', nbin1, key, log, cmap1, lxlab, lylab, lzf)
	plotmap(X, Y, f32, rep, 'f32SF', r'$\log(f_{\rm occ,SF})$', nbin2, key, log, cmap2, lxlab, lylab, lzf)
	plotmap(X, Y, fm32, rep, 'fmass2', r'$\log(f_{\rm mass})$', nbin3, key, log, cmap3, lxlab, lylab, lzf)

	nbin = np.linspace(-6, -1.5, 91)
	#nbin = 100
	#etac = np.log10(Ob/Om*np.geomspace(1e-3, 1e-2, 2))
	etac = [-4, -3, -2.5]
	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(eta), nbin, cmap=plt.cm.viridis)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(\eta)$',size=12)
	plt.contour(X, Y, np.log10(eta), etac, colors='k', linestyles=['-', '--', ':'])
	if key==0:
		plt.plot(Mup(lzf), ztp, 'k-.', label=r'$M_{\rm th}^{\rm atom}$')
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlabel(lxlab[key])
	plt.ylabel(lylab[log])
	plt.tight_layout()
	if key==0:
		plt.legend()
		plt.savefig(rep+'eta_z_Mh.pdf')
	else:
		plt.savefig(rep+'eta_z_Ms.pdf')
	plt.close()
	
	if key>0:
		exit()
	nbin = np.linspace(-3.5, 0, 71)
	#nbin = 100
	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(SFocc), nbin, cmap=plt.cm.plasma)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(f_{\rm SF})$',size=12)
	plt.contour(X, Y, np.log10(SFocc), [-2, -1], colors='k', linestyles=['-', '--'])
	plt.plot(Mup(lzf), ztp, 'k-.', label=r'$M_{\rm th}^{\rm atom}$')
	plt.plot(Mdown(lzf), ztp, 'k:', label=r'$M_{\rm th}^{\rm mol}$')
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlabel(lxlab[key])
	plt.ylabel(lylab[log])
	plt.xlim(mr)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'fSF_z_Mh.pdf')
	plt.close()
	#"""
	
	
	
	"""
	mlab = 1
	bhmode = 1
	fac2 = 30
	rmin = 0.1
	rat = 1
	rfac = 0.5
	fsk = 0.7
	nc = 7
	gal, adp = 1, 0
	Mmin = 32*586
	if mode==0:
		rela = process_halo(halocata, ds, mode=bhmode, fac2=fac2, rfac=rfac, fsk=fsk, nc=nc, rat=rat, ncore=ncore, adp=adp, gal=gal, mlab=mlab, rmin = rmin)
		totxt(rep+'haloprop.txt', rela['data'],0,0,0)
	else:
		data = np.array(retxt(rep+'haloprop.txt',6,0,0))
		rela = {}
		rela['z'] = ds['Redshift']
		rela['data'] = data
	halo_star(rela, rep, sn, Mmin=Mmin)
	plotrela(rela, rep, bhmode, 'halo', sn, Mmin=Mmin)
	"""
	
	
	
