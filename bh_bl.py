from process import *
import random
from scipy.optimize import curve_fit
import mpl_toolkits.mplot3d
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from miniquasar import tGW, coaltime, sigma_inf, rho_inf, Macc_edd, binary_gen, ms_mbh_default, astar_GW
import sys
import os
from metalicity import *
from gwform import *

fac0 = 0.3783187*0.3497985*0.399293243
fac = fac0
#fac = 1
bs = 4
#bs = 2
V = (bs/0.6774)**3 * fac
Zsun = 0.02

#Mup = lambda z: 2.5e7*((1+z)/10)**-1.5
#Mdown = lambda z: 1e6*((1+z)/10)**-2

lMH2 = np.array([[0.48, 1.21, 6.08, 10.0], [1.63, 3.93, 11.27, 37.9]])*1e6
Mdown_ = lambda z: 1.54e5*((1+z)/31.)**-2.074
def Mdown(z, mmin=1e6):
	m0 = Mdown_(z)
	out = m0 * (m0>mmin)
	out += mmin * (m0<=mmin)
	return out
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5

def sumy(l):
	if len(l)>1:
		return np.array([np.min(l), np.max(l), np.average(l), np.median(l), np.std(l)])
	else:
		return np.array([np.min(l), np.max(l), np.average(l), np.median(l)])

def psfr(l, lb, ls = '-', mode = 0, V = V, fz=0):
	if mode==0:
		plt.plot(1./l[0]-1., l[2]/V, ls=ls, label=lb)
	else:
		if fz>0:
			plt.plot(1./l[0][l[3]>0]-1., l[3][l[3]>0], ls=ls, label=lb)
		else:
			plt.plot(1./l[0]-1., l[3], ls=ls, label=lb)

def psfr_tot(l1, l2, lb, ls = '-', mode = 0, V = V):
	if mode==0:
		plt.plot(1./l1[0]-1, (l1[2]+l2[2])/V, ls=ls, label=lb)
	else:
		plt.plot(1./l1[0]-1, (l1[3]+l2[3]), ls=ls, label=lb)

def sfrmd(t, t0, a=0.182, b=1.26, c=1.865, d=0.071):
	return a*(t**b*np.exp(-t/c)+d*np.exp(d*(t-t0)/c))

def sfrgb(z, alpha):
	return 0.2*((1+z)/4.8)**alpha

def sfrbf(z):
	return 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)

def sfrtf(z, a, b, c, d):
	#t = 1/(z+1)
	#return a*(t**b*np.exp(-t/c)+d*np.exp(d*(t-1)/c)) 
	return a*(1+z)**b/(1+((1+z)/c)**d)

def sfrd_gen(lc):
	a, b, c = lc
	def sfrd(z, d, e, f):
		return a*(1+z)**b/(1+((1+z)/c)**d) * np.exp((1-(1+z)**e)/f)
	return sfrd

def Bondi(M, nH, T, vg, mu=1.22, X=0.76):
	cs = (5./3*BOL*T/PROTON/mu)**0.5
	m = M*Msun
	rho = nH*PROTON/X
	macc = 4*np.pi*(GRA*m)**2*rho/(cs**2+(vg*1e5)**2)**1.5
	return macc * YR/Msun

def LWbg(z, sfrd3, sfrd2, eta = 2e4, t3=5, t2=5):
	j1 = (eta/1e4)*sfrd3 / 1e-2 * ((1+z)/10)**3 * t3/5.
	j2 = (eta/1e4)*sfrd2/5. / 1e-2 * ((1+z)/10)**3 * t2/5.
	return j1, j2, j1+j2 #(eta/1e4)*(sfrd3 + sfrd2/5.) / 1e-2 * ((1+z)/10)**3

def CH(z, c0=2.53, a0=-3.55, a1=-0.056):
	a = a0+a1*(z-8)
	c = 1+c0*((1+z)/9)**a
	return c

def SFR_reion(z, C = 6, fesc=0.1):
	return 0.05 * (C/6)*(0.1/fesc) * ((1+z)/7)**3

def countBH(ds, Zth = 100., abd=0):
	z = ds['Redshift']
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType3' for x in keys])
	if tag>0:
		ad = ds.all_data()
		stbh = np.array(ad[('PartType3', 'Star Type')])
		bht = np.array(ad['PartType3', 'BH Type'])
		if abd>0:
			sel = np.logical_or(stbh>=30000, bht==2)
		else:
			sel = stbh>0#np.ones(len(stbh), dtype='int')
		if np.sum(sel)>0:
			poss = np.array(ad[('PartType3','Coordinates')][sel].to('kpccm/h'))
			sid = ad[('PartType3','ParticleIDs')][sel]
			#star = np.array(ad['PartType4', 'Star Type'])
			bht = np.array(ad['PartType3', 'BH Type'][sel])
			Z = np.array(ad['PartType3', 'Metallicity_00'][sel])/Zsun
			jlw = np.array(ad[('PartType3','J21_SF')])[sel]
			h2sf = np.array(ad[('PartType3','H2_SF')])[sel]
			lz = 1/np.array(ad[('PartType3','StellarFormationTime')])[sel] - 1
			#print('JLW_min = {}, JLW_max = {}, JLW_bar = {}, JLW_med = {}'.format(*sumy(jlw)))
			select = (bht==2)#*(Z<Zth)
			if np.sum(select)>0:
				print('JLW_SF:', sumy(jlw[select]))
				print('H2_SF:', sumy(h2sf[select]))
				print('Z_SF:', sumy(Z[select]))
				print(poss[select])
				print('z: ', lz[select])
				#print('Star Type:', star[select])
			n1 = np.sum(bht==1) #len(bht[bht==1])
			n2 = np.sum(bht==2) #len(bht[bht==2])
			#n3 = len(bht[select])
			return [z, n1, n2, poss[select], sid[select]]#, n3]
		else:
			return [z, 0, 0]#, 0]
	else:
		return [z, 0, 0]

def smass(ds, rep='/', nb=100, alp=0.7, Z0=1e-6, abd=0):
	ad = ds.all_data()
	keys = ds.field_list	
	bflag = np.sum([x[0] == 'PartType3' for x in keys])
	lmass = np.array(ad[('PartType4','Masses')].to('Msun'))
	stype = ad[('PartType4', 'Star Type')]
	lm2 = lmass[stype == 2]
	print('Stellar Particle mass: {} Msun'.format(lmass[0]))
	fabd = 1
	if bflag>0:
		#lmbh = np.array(ad[('PartType3','Masses')].to('Msun'))
		stbh = np.array(ad[('PartType3', 'Star Type')])
		if abd>0:
			selbh = stbh>=30000
		else:
			selbh = stbh>0 #np.ones(len(stbh), dtype='int')
		fabd = np.sum(selbh)/len(stbh)
		bht = ad[('PartType3','BH Type')][selbh]
		sel = bht==1
		Mbh = np.sum(np.array(ad[('PartType3', 'BH_NProgs')][selbh][sel])) * lmass[0] #np.sum(lmbh[bht == 1])
	M2 = np.sum(lm2)
	M1 = np.sum(lmass) - M2 + Mbh
	print('Mass in PopIII BH particles: {:.0f} Msun Mpc^-3'.format(Mbh/V))
	
	if bflag>0:
		lZ = np.hstack([ad[('PartType4', 'Metallicity_00')], ad[('PartType3', 'Metallicity_00')][selbh][sel]])/Zsun
		lpc = np.percentile(lZ, [16, 50, 84])
		print('ZSF: ',sumy(lZ), lpc)
		lZ += Z0 * (lZ<Z0)
		#his, edge = np.histogram(np.lZ, nb+1)
		plt.figure()
		logZ = np.log10(lZ)
		n, edge, pat = plt.hist(logZ, nb+1, density=True, alpha = alp, cumulative=True)
		plt.plot([-4, -4], [0, 1], 'k--', label=r'$Z_{\mathrm{crit}}$')
		plt.plot(np.log10([lpc[1], lpc[1]]), [0, 1], 'k-.', label=r'$Z_{\mathrm{median}}='+r'{:.3f}'.format(lpc[1])+'\ \mathrm{Z_{\odot}}$')
		plt.fill_between(np.log10(lpc), [0,0,0], [1,1,1], facecolor='gray', label='16-84th')
		plt.xlabel(r'$\log(Z\ [\mathrm{Z_{\odot}}])$')
		plt.ylabel(r'$F(<Z)$')
		plt.yscale('log')
		plt.xlim(np.log10(Z0), np.max(logZ))
		plt.ylim(np.min(n), 1)
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'ZSF_dis.pdf')
		plt.close()
	
		#ldens = np.hstack([ad[('PartType4', 'SF_dens')], ad[('PartType3', 'SF_dens')][sel]])
		ldens = ad[('PartType4', 'SF_dens')]
		ldens_ = ad[('PartType3', 'SF_dens')][selbh]
		sel = ldens>0
		sel_ = ldens_>0
		print('nSF (star): ', sumy(ldens), np.sum(sel)/len(ldens))
		print('nSF (BH): ', sumy(ldens_), np.sum(sel_)/len(ldens_))
		plt.figure()
		y1, x1, p1 = plt.hist(np.log10(ldens[sel]), nb+1, density=True, alpha=alp, label='Star')
		y2, x2, p2 = plt.hist(np.log10(ldens_[sel_]), nb+1, density=True, alpha=alp, label='BH')
		plt.xlabel(r'$\log(n_{\mathrm{H,SF}}\ [\mathrm{cm^{-3}}])$')
		plt.ylabel(r'Probability density')
		plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'nSF_dis.pdf')
		plt.close()
	
		xlm = [2, 4] #np.max(x1))
		plt.figure()
		plt.plot(x1[1:], 1-np.cumsum(y1)/np.sum(y1), label='Star')
		plt.plot(x2[1:], 1-np.cumsum(y2)/np.sum(y2), '--', label='BH')
		plt.plot(xlm, [1e-3, 1e-3], 'k:')
		plt.plot(xlm, [1e-2, 1e-2], 'k-.')
		plt.xlabel(r'$\log(n_{\mathrm{H,SF}}\ [\mathrm{cm^{-3}}])$')
		plt.ylabel(r'$F(>n_{\mathrm{H,SF}})$')
		plt.yscale('log')
		plt.xlim(xlm)
		plt.ylim(1e-4, 1+1e-3)
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'nSF_cum_dis.pdf')
		plt.close()
	return [M1, M2, ldens_, fabd]

new_3Dax = lambda x: plt.subplot(x,projection='3d')

def plot3d(fn, ID, R, ds, nump=1e5, alp=0.5, age = 1e7):
	llb = [r'$x\ [h^{-1}\mathrm{kpc}]$', r'$y\ [h^{-1}\mathrm{kpc}]$', r'$z\ [h^{-1}\mathrm{kpc}]$']
	ad0 = ds.all_data()
	keys = ds.field_list
	
	bflag = np.sum([x[0] == 'PartType3' for x in keys])
	if bflag > 0:
		stbh = np.array(ad0[('PartType3', 'Star Type')])
		bid = ad0[('PartType3','ParticleIDs')][stbh>=30000]
		posb = np.array(ad0[('PartType3','Coordinates')][stbh>=30000].to('kpccm/h'))
		sel0 = bid==ID
	else:
		sel0 = [0]
	
	sflag = np.sum([x[0] == 'PartType4' for x in keys])
	if sflag > 0:
		sid = ad0[('PartType4','ParticleIDs')]
		poss = np.array(ad0[('PartType4','Coordinates')].to('kpccm/h'))
		sel1 = sid==ID
	else:
		sel1 = [0]
		
	gid = ad0[('PartType0','ParticleIDs')]
	posg = np.array(ad0[('PartType0','Coordinates')].to('kpccm/h'))
	sel2 = gid==ID
	if np.sum(sel0)>0:
		pos = posb[sel0][0]
	if np.sum(sel1)>0:
		pos = poss[sel1][0]
	if np.sum(sel2)>0:
		pos = posg[sel2][0]
	ad = ds.sphere(pos, (R, 'kpccm/h'))
	if sflag > 0:
		poss = np.array(ad[('PartType4','Coordinates')].to('kpccm/h'))
		lts = np.array(ad[('PartType4', 'Stellar Age')])
	else:
		poss = []
	if bflag > 0:
		stbh = np.array(ad[('PartType3', 'Star Type')])
		posb = np.array(ad[('PartType3','Coordinates')][stbh>=30000].to('kpccm/h'))
	else:
		posb = []
	posg = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
	posd = np.array(ad[('PartType1','Coordinates')].to('kpccm/h'))
	T = temp(ad[('PartType0','InternalEnergy')],ad[('PartType0','Primordial HII')])
	print('Ngas_ngb = {} for ID = {} with hBH = {} kpc'.format(len(T), ID, R))
	lpd = np.random.choice(posd.shape[0], min(int(nump),posd.shape[0]),replace=False)
	lpg = np.random.choice(posg.shape[0], min(int(nump),posg.shape[0]),replace=False)
	posd = posd[lpd].T
	posg = posg[lpg]
	lT = T[lpg]
	hot = lT>1e4
	cold = lT<=1e4
	posgh = posg[hot].T
	posgc = posg[cold].T
	ms = (2*nump/(posg.shape[1] + posd.shape[1]))**0.5 * 0.05
	plt.figure(figsize=(5,5))
	ax = new_3Dax(111)
	#ax.set_aspect('equal','box')
	ax.plot(*posd, '.', markersize=ms, label='DM', alpha = alp, zorder=0)
	ax.plot(*posgc, '.', markersize=ms, label='Cold gas', alpha = alp, zorder=2)
	if len(poss)>0:
		lps = np.random.choice(poss.shape[0], min(int(nump),poss.shape[0]),replace=False)
		poss = poss[lps]
		lts = lts[lps]
		old = lts > age
		young = lts <= age
		po = poss[old].T
		py = poss[young].T
		mss = min((2*nump/poss.shape[0])**0.5 * 0.1, 7)
		ax.plot(*po, '*', label='Old star', markersize=1.5*mss, alpha = alp, zorder=3)
		ax.plot(*py, 'h', label='Young star', markersize=1.5*mss, alpha = alp, zorder=4)
	if len(posb)>0:
		posb = posb.T
		ax.plot(*posb, '^', label='BH', markersize=ms, alpha = alp, zorder=4)
	ax.plot(*posgh, '.', markersize=ms, label='Hot gas', color='r', alpha = alp, zorder=1)
	#ax.plot(*[[x] for x in pos], 'o', color='g', label='Target particle', alpha=0.5, zorder=5)
	ax.set_xlabel(llb[0])
	ax.set_ylabel(llb[1])
	ax.set_zlabel(llb[2])
	ax.set_xlim(pos[0]-R, pos[0]+R)
	ax.set_ylim(pos[1]-R, pos[1]+R)
	ax.set_zlim(pos[2]-R, pos[2]+R)
	plt.legend()
	plt.tight_layout()
	plt.savefig(fn)
	#plt.show()
	plt.close()

def LW_bg(sn, ds, le = [1500, 1500, 1950], re = [2500, 2500, 2050], ax = [0, 1], rep = './', mode=0, nump=1e6):
	llb = [r'$x\ [h^{-1}\mathrm{kpc}]$', r'$y\ [h^{-1}\mathrm{kpc}]$', r'$z\ [h^{-1}\mathrm{kpc}]$']
	ad = ds.box(le, re)
	posg = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
	sel = np.random.choice(posg.shape[0], min(int(nump),posg.shape[0]),replace=False)
	posg = posg[sel].T
	jlw = np.array(ad[('PartType0','J21')][sel])
	stbh = np.array(ad[('PartType3', 'Star Type')])
	bht = np.array(ad['PartType3', 'BH Type'])[stbh>=30000]
	poss = np.array(ad[('PartType3','Coordinates')][stbh>=30000].to('kpccm/h'))
	posbh1 = poss[bht==1].T
	posbh2 = poss[bht==2].T
	plt.figure()
	axis = plt.subplot(111)
	axis.set_aspect(aspect=1)
	plt.tripcolor(posg[ax[0]], posg[ax[1]], np.log10(jlw),cmap=plt.cm.binary)
	cb = plt.colorbar()
	plt.clim(-.5,2.5)
	cb.set_label(r'$\log(J_{21})$')
	if mode!=0:
		plt.plot(*posbh1[ax], '.', color='g', label='Stellar BH', alpha=0.5)
		plt.plot(*posbh2[ax], '.', color='purple', label='DCBH', alpha=0.5)
		plt.legend()
	plt.xlim(le[ax[0]], re[ax[0]])
	plt.ylim(le[ax[1]], re[ax[1]])
	plt.xlabel(llb[ax[0]])
	plt.ylabel(llb[ax[1]])
	plt.tight_layout()
	plt.savefig(rep+'LW_bg_'+str(sn)+'_'+str(np.sum(ax))+'.png',dpi=300)
	plt.close()
	print(sumy(jlw))

def temp_bg(sn, ds, le = [1500, 1500, 1950], re = [2500, 2500, 2050], ax = [0, 1], rep = './', mode=0, nump=1e6, lflag=0, wh=6.5, lss=[1, 8, 64]):
	llb = [r'$x\ [h^{-1}\mathrm{kpc}]$', r'$y\ [h^{-1}\mathrm{kpc}]$', r'$z\ [h^{-1}\mathrm{kpc}]$']
	ad = ds.box(le, re)
	posg = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
	sel = np.random.choice(posg.shape[0], min(int(nump),posg.shape[0]),replace=False)
	posg = posg[sel].T
	lT = temp(ad[('PartType0','InternalEnergy')][sel],ad[('PartType0','Primordial HII')][sel])
	stbh = np.array(ad[('PartType3', 'Star Type')])
	bht = np.array(ad['PartType3', 'BH Type'])[stbh>=3e4]
	poss = np.array(ad[('PartType3','Coordinates')][stbh>=3e4].to('kpccm/h'))
	posst = np.array(ad[('PartType4','Coordinates')].to('kpccm/h')).T
	posbh1 = poss[bht==1].T
	posbh2 = poss[bht==2].T
	if lflag==0:
		asp = (re[0]-le[0])/(re[1]-le[1])
		fsz = (wh, wh/asp)
		plt.figure(figsize=fsz)
	else:
		plt.figure()
	axis = plt.subplot(111)
	axis.set_aspect(aspect=1)
	plt.tripcolor(posg[ax[0]], posg[ax[1]], np.log10(lT),cmap=plt.cm.seismic)
	plt.clim(2.5,5.5)
	if lflag>0:
		cb = plt.colorbar()
		cb.set_label(r'$\log(T\ [\mathrm{K}])$')
	if mode!=0:
		if lflag==0:
			nc = 1
			plt.scatter(*posst[ax], marker='.', c='orange', s=lss[0], alpha=1.0, label='Star')
		else:
			nc = 0
		plt.scatter(*posbh1[ax], marker='.', color='g', label='Stellar BH', s=lss[1], alpha=0.5)
		if len(posbh2[0])>0:
			plt.scatter(*posbh2[ax], marker='.', color='purple', label='DCBH', alpha=0.5, zorder=2, s=lss[2])
		if lflag>0:
			lgnd = plt.legend()#markerscale=3)
			#lgnd.legendHandles[0]._sizes = [32]
			lgnd.legendHandles[nc]._sizes = [64]
			if len(posbh2[0])>0:
				lgnd.legendHandles[nc+1]._sizes = [256]
		else:
			plt.tick_params(top=False, bottom=False, left=False, right=False, 
			labelleft=False, labelbottom=False)
		#for legend_handle in lgd.legendHandles:
		#	legend_handle._legmarker.set_markersize(9)
	plt.xlim(le[ax[0]], re[ax[0]])
	plt.ylim(le[ax[1]], re[ax[1]])
	if lflag>0:
		plt.xlabel(llb[ax[0]])
		plt.ylabel(llb[ax[1]])
	plt.tight_layout()
	if lflag==0:
		plt.savefig(rep+'temp_'+str(sn)+'.png',dpi=666)
	else:
		plt.savefig(rep+'temp_'+str(sn)+'_'+str(np.sum(ax))+'.png',dpi=300)
	plt.close()
	print(sumy(lT))

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

def denspro(sn, ds, hd0, le = [1500, 1500, 1950], re = [2500, 2500, 2050], ax = [0, 1], nb=100, rep = './', norm = 6.66e6, mode=0, bh = 0, lflag=0, nump=1e6, hs = 0):
	h = ds['HubbleParam']
	z = ds['Redshift']
	if hs>0:
		mth = Mup(z)
		ns = Mup(z)/Mdown(z)
	else:
		mth = Mdown(z)
		ns = 1
	#mth = Mdown(z)
	llb = [r'$x\ [h^{-1}\mathrm{kpc}]$', r'$y\ [h^{-1}\mathrm{kpc}]$', r'$z\ [h^{-1}\mathrm{kpc}]$']
	ad = ds.box(le, re)
	posg = np.array(ad[('PartType1','Coordinates')].to('kpccm/h'))
	print('log(Ngas) = {:.2f}'.format(np.log10(len(posg))))
	if nump>0:
		sel = np.random.choice(posg.shape[0], min(int(nump),posg.shape[0]),replace=False)
		posg = posg[sel].T
	else:
		posg = posg.T
	print('log(Ngas) = {:.2f}'.format(np.log10(len(posg[0]))))
	"""
	stbh = np.array(ad[('PartType3', 'Star Type')])
	bht = np.array(ad['PartType3', 'BH Type'])[stbh>=3e4]
	poss = np.array(ad[('PartType3','Coordinates')][stbh>=3e4].to('kpccm/h'))
	posbh1 = poss[bht==1].T
	posbh2 = poss[bht==2].T
	dcbh = 0
	dcbh = len(posbh2[0])>0
	"""
	dcbh = 0
	lmh = hd0[0]/h
	sel = lmh>mth
	hd = hd0.T[sel]
	hd = hd.T
	lr = hd[1]
	posh = hd[2:5]
	lselh = [(posh[i]>le[i]) * (posh[i]<re[i]) for i in range(3)]
	selh = lselh[0] * lselh[1] * lselh[2]
	posh = posh.T
	posh = posh[selh]
	lm = lmh[sel][selh]
	posh = posh.T
	#obj = recaesar(sn, rep)
	#posh = np.array([x.pos.to('kpccm/h') for x in obj.halos]).T
	#lselh = [(posh[i]>le[i]) * (posh[i]<re[i]) for i in range(3)]
	#selh = lselh[0] * lselh[1] * lselh[2]
	#posh = posh.T
	#posh = posh[selh]
	#lm = np.array([(1e10*x.virial_quantities['circular_velocity']**2*x.radii['virial'].to('kpc/h')/G) for x in obj.halos])[selh]
	#sel = lm>=mth
	#posh = posh[sel]
	#posh = posh.T
	
	plt.figure()
	axis = plt.subplot(111)
	axis.set_aspect(aspect=1)
	print('Generating 2D map...')
	plt.hist2d(posg[ax[0]],posg[ax[1]],bins=nb,norm=LogNorm(),cmap=plt.cm.Blues)
	#plt.clim(1, 1e3)
	#"""
	if lflag>0:
		cb = plt.colorbar()
		cb.set_label(r'$\log(N)$')
	"""
	if bh>0:
		plt.plot(*posbh1[ax], '.', color='g', label='Stellar BH', alpha=0.5,zorder=2)
		if dcbh:
			plt.plot(*posbh2[ax], '.', color='purple', label='DCBH', alpha=0.5)
	"""
	print('Plotting haloes...')
	if mode!=0:
		plt.scatter(posh[ax[0]],posh[ax[1]],s=ns*(lm/mth),edgecolors='k',facecolors='none',lw=1, zorder=1, linewidths=0.3, alpha=0.5, label='Halo')
		plt.legend()
		lgnd = plt.legend()
		lgnd.legendHandles[(bh>0)*(1+dcbh)]._sizes = [256]
	if lflag==0:
		plt.tick_params(top=False, bottom=False, left=False, right=False, 
			labelleft=False, labelbottom=False)
	else:
		plt.xlabel(llb[ax[0]])
		plt.ylabel(llb[ax[1]])
	plt.xlim(le[ax[0]], re[ax[0]])
	plt.ylim(le[ax[1]], re[ax[1]])
	plt.tight_layout()
	if lflag>0:
		plt.savefig(rep+'denspro_'+str(sn)+'_'+str(np.sum(ax))+'.png',dpi=300)
	else:
		plt.savefig(rep+'denspro_'+str(sn)+'.png',dpi=300)
	plt.close()

def pbd(x, Bxs):
	# make sure that the displacement between two particles
	# is calculated with their nearest images 
	return x - Bxs*(x>Bxs/2.) + Bxs*(x<-Bxs/2.)

def pbd2(x, Bxs):
	# convert all particles outside the box to their images inside the box
	return x + Bxs*(x<0) - Bxs*(x>Bxs)

def profile0(pos, cen, R, nbin = 20, fac=1e-2, mode = 1, Bxs = 4000):
	lpart = np.array(pos)
	rcen = np.array(cen)
	lr0 = np.array([np.linalg.norm(pbd(x-rcen, Bxs)) for x in lpart])
	within = lr0<R
	lr = lr0[within]
	lr = sorted(lr, key=lambda x: x)
	if mode==0:
		lrb1 = np.linspace(fac*R, R, nbin)
	else:
		lrb1 = np.geomspace(lr[0]*2, R, nbin)
	lrb0 = np.hstack([[0.0], lrb1])
	lrb = 0.5*(lrb0[1:] + lrb0[:-1])
	lvs = (np.pi*4/3)*(lrb0[1:]**3-lrb0[:-1]**3)
	lp = np.zeros(nbin)
	count = 0
	for r in lr:
		while r>=lrb0[count+1]:
			count += 1
		if count>=nbin:
			break
		lp[count] += 1
	sel = lp>0
	d = {}
	d['pro'] = [lrb[sel], (lp/lvs)[sel]]+ \
		[(lp**0.5/lvs)[sel], np.sum(lp), lvs[sel]]
		# radius bins, density, 
		# Poisson error, total particle number, volume of shells
	d['part'] = within # particles used in the profle calculation 
		# denoted by a boolean array
	#d['cen'] = rcen # CoM
	return d
	
def profile1(pos, cen, yin, R, nbin = 20, fac=1e-2, mode=1, Bxs = 4000):
	lpart = np.array(pos)
	rcen = np.array(cen)
	lr0 = np.array([np.linalg.norm(pbd(x-rcen, Bxs)) for x in lpart])
	within = lr0<R
	lr = lr0[within]
	ly0 = yin[within]
	data = sorted(np.array([lr, ly0]).T, key=lambda x: x[0])
	data = np.array(data).T
	lr, ly = data
	if mode==0:
		lrb1 = np.linspace(fac*R, R, nbin)
	else:
		lrb1 = np.geomspace(lr[0]*2, R, nbin)
	lrb0 = np.hstack([[0.0], lrb1])
	lrb = 0.5*(lrb0[1:] + lrb0[:-1])
	lvs = (np.pi*4/3)*(lrb0[1:]**3-lrb0[:-1]**3)
	lp = np.zeros(nbin)
	lcol = [[] for i in range(nbin)]
	count = 0
	for r, y in zip(lr, ly):
		while r>=lrb0[count+1]:
			count += 1
		if count>=nbin:
			break
		lp[count] += 1
		lcol[count].append(y)
	sel = lp>0
	lden = np.zeros(nbin)
	lstd = np.zeros(nbin)
	for i in range(nbin):
		col = lcol[i]
		if lp[i]==0:
			lden[i], lstd[i] = 0, 0
			continue
		lden[i] = np.median(col)
		#lden[i] = np.average(col)
		if lp[i]==1:
			lstd[i] = lden[i]/2**0.5
		else:
			lstd[i] = np.std(col)
	d = {}
	d['pro'] = [lrb[sel], lden[sel], lstd[sel], np.sum(lp), lvs[sel]]
	d['data'] = data
	return d
	
def profile2(pos, cen, yin, R, nbin = 20, fac=1e-2, mode=1, Bxs = 4000):
	lpart = np.array(pos)
	rcen = np.array(cen)
	dis0 = np.array([pbd(x-cen, Bxs) for x in lpart])
	lr0 = np.array([np.linalg.norm(x) for x in dis0])
	within = lr0<R
	lr = lr0[within]
	dis = dis0[within]
	ly_raw = yin[within]
	ymean = np.average(ly_raw, axis=0)
	ly0 = -np.array([np.dot(x, y-ymean) for x, y in zip(dis, ly_raw)])/lr
	data = sorted(np.array([lr, ly0]).T, key=lambda x: x[0])
	data = np.array(data).T
	lr, ly = data
	if mode==0:
		lrb1 = np.linspace(fac*R, R, nbin)
	else:
		lrb1 = np.geomspace(lr[0]*2, R, nbin)
	lrb0 = np.hstack([[0.0], lrb1])
	lrb = 0.5*(lrb0[1:] + lrb0[:-1])
	lvs = (np.pi*4/3)*(lrb0[1:]**3-lrb0[:-1]**3)
	lp = np.zeros(nbin)
	lcol = [[] for i in range(nbin)]
	count = 0
	for r, y in zip(lr, ly):
		while r>=lrb0[count+1]:
			count += 1
		if count>=nbin:
			break
		lp[count] += 1
		lcol[count].append(y)
	lden = np.zeros(nbin)
	lstd = np.zeros(nbin)
	for i in range(nbin):
		col = lcol[i]
		if lp[i]==0:
			lden[i], lstd[i] = 0, 0
			continue
		lden[i] = np.average(col)
		if lp[i]==1:
			lstd[i] = lden[i]/2**0.5
		else:
			lstd[i] = np.std(col)
	sel = lp>0
	d = {}
	d['pro'] = [lrb[sel], lden[sel], lstd[sel], np.sum(lp), lvs[sel]]
	d['data'] = data
	return d

oi = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
mst = ['o', '^', 's', 'D', 'h', 'p']

aplog = lambda x: np.sign(x)*(np.log10(np.abs(x)+1))
rga = lambda l: [np.min(l), np.max(l)]
nth_bh, nth_sf, nth_hii = 2e3, 1e2, 1e3
T1, T2 = 7e3, 1e4
Zth = 2e-4
h2th = 1e-6

def main(sn, nbh = 3, R = 2, dm = 1, rep = './', x0 = 1e-2, nbin = 30, fac=1e-2, mode=1, Bxs = 4000, alp = 0.1, ms = 0.5, cs = oi, mst = mst):
	ds = yt.load(rep+'snapshot_'+str(sn).zfill(3)+'.hdf5')
	z = ds['Redshift']
	h = ds['HubbleParam']
	a = 1./(1.+z)
	ad0 = ds.all_data()
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType3' for x in keys])
	if tag<=0:
		print('There are no BHs!')
		return -1
	poss = np.array(ad0[('PartType3','Coordinates')].to('kpccm/h'))
	bht = np.array(ad0['PartType3', 'BH Type'])
	sel = bht==2
	n0 = np.sum(sel)
	if n0==0:
		print('There is no DCBH!')
		return -2
	n = int(min(n0, nbh))
	idbh = np.array(ad0['PartType3', 'ParticleIDs'])[sel]#
	posbh = poss[sel]#[0:n]
	Zbh = np.array(ad0['PartType3', 'Metallicity_00'])[sel]/Zsun #[0:n]/Zsun
	h2sf = np.array(ad0[('PartType3','H2_SF')])[sel]#[0:n]
	data_bh = [[i, p, z, h] for i, p, z, h in zip(idbh, posbh, Zbh, h2sf)]
	data_bh = np.array(sorted(data_bh, key = lambda x: x[0]))
	idbh = data_bh[:,0][0:n]
	posbh = np.array(data_bh[:,1])[0:n]
	Zbh = np.array(data_bh[:,2])[0:n]
	h2sf = np.array(data_bh[:,3])[0:n]
	if dm==0:
		norm = np.array(ad0[('PartType0', 'Masses')][0].to('g')/(KPC*a/h)**3)*Hfrac/PROTON
	else:
		norm = Hfrac/PROTON
	#"""	
	plt.figure(figsize=(8, 6))
	ax0 = plt.subplot(221)
	ax1 = plt.subplot(222)
	ax2 = plt.subplot(223)
	ax3 = plt.subplot(224)
	for i in range(n):
		cen = posbh[i]
		ad = ds.sphere(cen, (R, 'kpccm/h'))
		pos = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
		T = temp(ad[('PartType0','InternalEnergy')],ad[('PartType0','Primordial HII')])
		Z = np.array(ad['PartType0', 'Metallicity_00'])/Zsun
		xh2 = np.array(ad['PartType0', 'Primordial H2'])
		if dm==0:
			prho = profile0(pos, cen, R, nbin, fac, mode, Bxs)
		else:
			rho = np.array(ad[('PartType0','Density')].to('g/cm**3'))
			prho = profile1(pos, cen, rho, R, nbin, fac, mode, Bxs)
		ptemp = profile1(pos, cen, T, R, nbin, fac, mode, Bxs)
		pZ = profile1(pos, cen, Z, R, nbin, fac, mode, Bxs)
		ph2 = profile1(pos, cen, xh2, R, nbin, fac, mode, Bxs)
		p0, p1 = prho['pro'], ptemp['pro']
		p2, p3 = pZ['pro'], ph2['pro']
		if dm!=0:
			d0, d1 = prho['data'], ptemp['data']
		d2, d3 = pZ['data'], ph2['data']
		ax0.errorbar(p0[0], p0[1]*norm, p0[2]*norm, ls=lls[i], zorder = 2, color = cs[i])
		ax1.errorbar(p1[0], p1[1], p1[2], ls=lls[i], label=r'DCBH_'+str(i), zorder = 2, color = cs[i])
		if dm!=0:
			ax0.plot(d0[0], d0[1]*norm, '.', alpha = alp, zorder = 1, color = cs[i])
			ax1.plot(d1[0], d1[1], '.', alpha = alp, zorder = 1, color = cs[i])
		
		ax2.plot([x0*2], [Zbh[i]], mst[i], color = cs[i], alpha = 0.5)
		ax3.plot([x0*2], [h2sf[i]], mst[i], color = cs[i], alpha = 0.5)
		ax2.plot(p2[0], p2[1], ls=lls[i], zorder = 2, color = cs[i])
		ax3.plot(p3[0], p3[1], ls=lls[i], zorder = 2, color = cs[i])
		ax2.plot(d2[0], d2[1], '.', alpha = alp, zorder = 1, color = cs[i])
		ax3.plot(d3[0], d3[1], '.', alpha = alp, zorder = 1, color = cs[i])

	ax0.plot([x0, R], [nth_bh, nth_bh], 'k--', label=r'$n_{\mathrm{th}}$(BH)')
	ax0.plot([x0, R], [nth_sf, nth_sf], 'k-.', label=r'$n_{\mathrm{th}}$(SF)')
	ax0.plot([x0, R], [nth_hii, nth_hii], 'k:', label=r'$n_{\mathrm{th}}$(HII)')
	ax1.fill_between([x0, R], [T1, T1], [T2, T2], facecolor='k', zorder = 0, label=r'$T$(BH)', alpha = alp*3)
	ax2.plot([x0, R], [Zth, Zth], 'k--', label=r'$Z_{\mathrm{th}}$')
	ax3.plot([x0, R], [h2th, h2th], 'k--', label=r'$x_{\mathrm{H_{2},th}}$')

	ax0.set_ylabel(r'$n_{\mathrm{H}}\ [\mathrm{cm^{-3}}]$')
	ax1.set_ylabel(r'$T\ [\mathrm{K}]$')
	ax2.set_ylabel(r'$Z\ [\mathrm{Z_{\odot}}]$')
	ax3.set_ylabel(r'$\mathrm{[H_{2}/H]}$')
	ax0.set_yscale('log')
	ax1.set_yscale('log')
	ax2.set_yscale('log')
	ax3.set_yscale('log')
	#ax0.set_ylim(5e-1, 5e3)
	#ax1.set_ylim(1e2, 3e4)
	#ax2.set_ylim(1e-5, 1)
	#ax3.set_ylim(9e-11, 1e-4)
	
	ax0.set_xlim(x0, R)
	ax1.set_xlim(x0, R)
	ax2.set_xlim(x0, R)
	ax3.set_xlim(x0, R)
	ax0.set_xlabel(r'$r\ [h^{-1}\mathrm{kpc}]$')
	ax1.set_xlabel(r'$r\ [h^{-1}\mathrm{kpc}]$')
	ax2.set_xlabel(r'$r\ [h^{-1}\mathrm{kpc}]$')
	ax3.set_xlabel(r'$r\ [h^{-1}\mathrm{kpc}]$')
	ax0.set_xscale('log')
	ax1.set_xscale('log')
	ax2.set_xscale('log')
	ax3.set_xscale('log')
	ax0.legend(loc=3)
	ax1.legend(loc=4)
	ax2.legend(loc=2)
	ax3.legend(loc=4)	

	plt.tight_layout()
	plt.savefig(rep+'DCBHpro_'+str(sn)+'.pdf')
	#plt.show()
	plt.close()	
	#"""	
	#umass = np.array(ad0[('PartType0', 'Masses')][0].to('Msun'))
	uvel = 1e5
	ul = a*KPC/h
	plt.figure()
	for i in range(n):
		cen = posbh[i]
		ad = ds.sphere(cen, (R, 'kpccm/h'))
		pos = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
		vel = np.array(ad[('PartType0','Velocities')].to('km/s'))
		pro = profile2(pos, cen, vel, R, nbin, fac, mode, Bxs)
		if dm==0:
			prho = profile0(pos, cen, R, nbin, fac, mode, Bxs)
		else:
			rho = np.array(ad[('PartType0','Density')].to('g/cm**3'))
			prho = profile1(pos, cen, rho, R, nbin, fac, mode, Bxs)
		p = pro['pro']
		pr = prho['pro']
		lr = p[0]
		lm = np.cumsum(p[4]*pr[1]) * ul**3 *1e10/UM
		lacc = 4*np.pi*lr**2*p[1]*pr[1] * uvel * ul**2 *1e10/UM * YR
		lacc_er = 4*np.pi*lr**2*p[1]*pr[2] * uvel * ul**2 *1e10/UM * YR
		plt.errorbar(lm, lacc, lacc_er, ls=lls[i], zorder = 2, color = cs[i], label=r'DCBH_'+str(i))
		#plt.plot(lm, lacc, ls=lls[i], zorder = 2, color = cs[i], label=r'DCBH_'+str(i))
	xr = [1e4, 1e8]
	plt.plot(xr, [0, 0], 'k--', lw=0.5)
	plt.plot(xr, [0.04, 0.04], 'k-.', label=r'$\mathrm{\dot{M}_{crit}}$')
	plt.xlim(xr)
	plt.ylim(-0.5, 1.)
	plt.xscale('log')
	#plt.yscale('symlog')
	plt.xlabel(r'Enclosed mass [$\mathrm{M_{\odot}}$]')
	plt.ylabel(r'Accretion rate [$\mathrm{M_{\odot}\ yr^{-1}}$]')
	plt.legend(loc=2)
	plt.tight_layout()
	plt.savefig(rep+'Mdot_M_'+str(sn)+'.pdf')
	plt.close()
	print('Redshift: {:.3f}'.format(z))
	print(Zbh)
	print(h2sf)
	print(posbh)
	print(idbh)

def findhost(pos, obj, fac = 2., Bxs = 4000, h = 0.6774, mode=0):
	lh = obj.halos
	out = lh[0]
	rmin = fac
	lr = []
	for halo in lh:
		r200 = np.array(halo.radii['r200'].to('kpccm/h'))
		hpos = np.array(halo.pos.to('kpccm/h'))
		dis = pbd(hpos - pos, Bxs)
		r = np.linalg.norm(dis)
		if mode==0:
			rat = r/r200
		else:
			rat = r
		lr.append(r)
		if rat <= rmin:
			rmin = rat
			out = halo
	return [out, rmin]
	
def track(halo, sns, snf, fac = 5., rep = './', base = 'snapshot', ext = '.hdf5', post = 'caesar', Bxs  =4000):
	ds0 = yt.load(rep+base+'_'+str(sns).zfill(3)+ext)
	hub = ds0['HubbleParam']
	z0 = ds0['Redshift']
	ad0 = ds0.all_data()
	IDlist = ad0[('PartType1', 'ParticleIDs')][halo.dmlist]
	vel = np.linalg.norm(np.array(halo.vel))
	pos = np.array(halo.pos.to('kpccm/h'))
	r200 = np.array(halo.radii['r200'].to('kpccm/h'))
	sn = sns - 1
	lfrac = [1.0]
	lz = [z0]
	lhalo = [halo]
	while sn >= snf:
		obj = caesar.load(rep+post+'_'+str(sn).zfill(3)+ext)
		ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
		ad = ds.all_data()
		IDs = ad[('PartType1', 'ParticleIDs')]
		z = ds['Redshift']
		dt = np.abs(TZ(z) - TZ(z0))
		R = fac * max(dt * vel * 1e5 * (1+z)/KPC * hub, r200)
		frac = 0.0
		flag = 0
		for h in obj.halos:
			pos_ = np.array(h.pos.to('kpccm/h'))
			r = np.linalg.norm(pbd(pos_ - pos, Bxs))
			if r > R:
				continue
			flag = 1
			IDlist_ = IDs[h.dmlist]
			overlap = np.intersect1d(IDlist, IDlist_)
			frac_ = len(overlap)/len(IDlist)
			if frac_ > frac:
				frac = frac_
				sel = h
		if flag==0:
			break;
		sn -= 1
		print(frac)
		lfrac.append(frac)
		lhalo.append(sel)
		lz.append(z)
		z0 = z
		IDlist = IDs[sel.dmlist]
		vel = np.linalg.norm(np.array(sel.vel))
		pos = np.array(sel.pos.to('kpccm/h'))
		r200 = np.array(sel.radii['r200'].to('kpccm/h'))
	return np.array([lhalo, lz, lfrac])
	
def haloget(his, eps = 0.1):
	d = {}
	lh, lz, lfrac = his
	d['z'] = lz
	d['pos'] = np.array([h.pos.to('kpccm/h') for h in lh])
	d['mtot'] = np.array([h.masses['total'] for h in lh])
	d['mdm'] = np.array([h.masses['dm'] for h in lh])
	d['mgas'] = np.array([h.masses['gas'] for h in lh])
	d['mstr'] = np.array([h.masses['stellar'] for h in lh])*0.1
	d['temp'] = np.array([h.temperatures['mass_weighted'] for h in lh])
	d['vtemp'] = np.array([h.temperatures['virial'] for h in lh])
	d['r200'] = np.array([h.radii['r200'].to('kpccm/h') for h in lh])
	d['frac'] = lfrac
	return d

def SFgas(sn, rep = './', base = 'snapshot', ext = '.hdf5', X = 0.76):
	ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	ad = ds.all_data()
	gas = np.array(ad[('PartType0', 'Masses')].to('Msun'))
	mgas = np.max(gas)
	star = np.array(ad[('PartType4', 'Masses')].to('Msun'))
	mstar = star[0]
	mstar_tot = np.sum(star)
	lm = gas[gas<mgas]
	lms = mgas - lm
	ms_tot = np.sum(lms)
	Ng = int(mgas/mstar+0.1)
	lfrac = np.zeros(Ng)
	lfrac_ = np.zeros(Ng)
	for i in range(1, Ng):
		lfrac[i-1] = np.sum(lms[lms/mstar+0.1>i])
		lfrac_[i-1] = np.sum(lms[lms/mstar+0.1>i] - i*mstar)
	lfrac += mstar_tot - ms_tot
	return lfrac/mstar_tot, lfrac_/mstar_tot
	
def BHmass_spec(sn, ds, rep = './', tp = 3, tbh = 1, base = 'snapshot', ext = '.hdf5', X = 0.76, mode = 0, nbm=100, abd=0):
	#ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	z = ds['Redshift']
	zf = '{:.2f}'.format(z)
	h = ds['HubbleParam']
	a = 1./(1.+z)
	keys = ds.field_list
	#print(keys)
	plab = 'PartType'+str(tp)
	tag = np.sum([x[0] == plab for x in keys])
	if tag==0:
		print('No BH particles!')
		return 0
	ad = ds.all_data()
	bht = ad[(plab, 'BH Type')]
	if abd>0:
		stbh = ad[(plab, 'Star Type')]
		selbh = np.logical_or(stbh>=3e4, bht==2)
	else:
		stbh = ad[(plab, 'Star Type')]
		selbh = stbh>0 #np.ones(len(bht), dtype='int')
	bht = bht[selbh]
	sel = bht == tbh
	mass = np.array(ad[(plab, 'BH Mass')][selbh][sel])
	massm = np.array(ad[(plab, 'BH Max mass')][selbh][sel])
	mdot = np.array(ad[plab, 'BH_Mdot'][selbh])
	mdotmax = 0
	for i in range(len(mdot)):
		if mdot[i]>mdotmax:
			bhind = i
			mdotmax = mdot[i]
	print('Mdot: ', sumy(mdot))
	print('Maximum Mdot: {} [Msun yr^-1]'.format(mdotmax))
	bhm = np.array(ad[(plab, 'BH_Mass')][selbh])
	bhmacc = np.array(ad[(plab, 'BH_Macc')][selbh])
	medd = Macc_edd(bhm, 0.125)
	print('Macc/MBH: ', sumy(bhmacc/bhm))
	rat = mdot/medd
	print('Mdot/Medd: ', sumy(rat))
	bhmacc += (bhm - bhmacc) * (bhm < bhmacc)
	#print('Mdot: ', mdot)
	bhdens = np.array(ad[(plab, 'Density')][selbh].to('g/cm**3'))[sel]
	#print('Density: ', bhdens * X / PROTON)
	nb = min(int(len(mass)/30),nbm)
	#print(mass.shape, nb)
	
	Mbh = 125
	nH = 1
	T = 10000
	vg = 30
	Mbd1 = Bondi(Mbh, nH, T, vg)
	rat1 = Mbd1/Macc_edd(Mbh, 0.125)
	Mbd2 = Bondi(Mbh, nH, T, 0)
	rat2 = Mbd2/Macc_edd(Mbh, 0.125)
	
	plt.figure()
	his, edge, pat = plt.hist(np.log10(rat[rat>0]), nb, alpha = 0.5, label=r'$z='+zf+'$', density = True)
	y1, y2 = 0, np.max(his)/0.7
	plt.plot(np.log10([rat1, rat1]), [y1, y2], 'k--', label='Typical Bondi rate'+'\n'
		+r'($M_{\mathrm{BH}}='+str(Mbh)+'\ \mathrm{M_{\odot}}$, '
		+r'$n_{\mathrm{H}}='+str(nH)+'\ \mathrm{cm^{-3}}$, '
		+r'$T={}$ K, '.format(T)+r'$v_{\mathrm{g}}='+str(vg)+'\ \mathrm{km\ s^{-1}}$)')
	#plt.plot(np.log10([rat2, rat2]), [y1, y2], 'k-.', label='Bondi rate (central)')
	plt.legend()
	plt.xlabel(r'$\log(\dot{M}_{\mathrm{acc}}/\dot{M}_{\mathrm{Edd}})$')
	plt.ylabel(r'Probability density')
	plt.ylim(y1, y2)
	#plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'BHrat_spec_'+str(sn)+'.pdf')
	plt.close()
	
	plt.figure()
	his, edge, pat = plt.hist(np.log10(mdot[mdot>0]), nb, alpha = 0.5, label=r'$z='+zf+'$', density = True)
	y1, y2 = 0, np.max(his)/0.7
	plt.plot(np.log10([Mbd1, Mbd1]), [y1, y2], 'k--', label='Typical Bondi rate\n'
		+r'($M_{\mathrm{BH}}='+str(Mbh)+'\ \mathrm{M_{\odot}}$, '
		+r'$n_{\mathrm{H}}='+str(nH)+'\ \mathrm{cm^{-3}}$, '
		+r'$T={}$ K, '.format(T)+r'$v_{\mathrm{g}}='+str(vg)+'\ \mathrm{km\ s^{-1}}$)')
	#plt.plot(np.log10([Mbd2, Mbd2]), [y1, y2], 'k-.', label='Bondi rate (central)')
	plt.legend()
	plt.xlabel(r'$\log(\dot{M}_{\mathrm{acc}}\ [\mathrm{M_{\odot}\ yr^{-1}}])$')
	plt.ylabel(r'Probability density')
	plt.ylim(y1, y2)
	#plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'BHmdot_spec_'+str(sn)+'.pdf')
	plt.close()

	plt.figure()
	plt.hist(np.log10(bhdens[bhdens>0]), nb, alpha = 0.5, label=r'$z='+zf+'$', density = True)
	plt.legend()
	plt.xlabel(r'$\log(\rho_{\mathrm{gas}}\ [\mathrm{g\ cm^{-3}}])$')
	plt.ylabel(r'Probability density')
	#plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'BHdens_spec_'+str(sn)+'.pdf')
	plt.close()

	plt.figure()
	his, edge, pat = plt.hist(np.log10(bhmacc[bhmacc>0]), nb, alpha = 0.5, label=r'$z='+zf+'$', density = True)
	y1, y2 = 0, np.max(his)/0.7
	plt.legend()
	plt.xlabel(r'$\log(M_{\mathrm{acc}}\ [\mathrm{M}_{\odot}])$')
	plt.ylabel(r'Probability density')
	plt.ylim(y1, y2)
	#plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'BHmass_spec_acc_'+str(sn)+'.pdf')
	plt.close()
	
	plt.figure()
	if mode==0:
		plt.hist(bhm, nb, alpha = 0.5, label=r'$z='+zf+'$', density = True)
		plt.hist(massm, int(nb/4), alpha = 1, label=r'Lseed', density = True, histtype='step', lw=1.5, color='k')
		plt.xlabel(r'$M_{\mathrm{BH}}\ [\mathrm{M}_{\odot}]$')
	else:
		plt.hist(np.log10(bhm), nb, alpha = 0.5, label=r'$z='+zf+'$', density = True)
		plt.hist(np.log10(massm), int(nb/4), alpha = 1, label=r'Lseed', density = True, histtype='step', lw=1.5, color='k')#, ec='k', linestyle='--')#, fc=None)
		plt.xlabel(r'$\log(M_{\mathrm{BH}}\ [\mathrm{M}_{\odot}])$')
	plt.legend()
	plt.ylabel(r'Probability density')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'BHmass_spec_tot_'+str(sn)+'.pdf')
	plt.close()
	
	m1, m2 = 40, 600
	bins = np.linspace(m1, m2) #np.geomspace(m1, m2, nb)
	
	plt.figure()
	#plt.hist(np.log10(mass), np.log10(bins), alpha = 0.5, label=r'Hseed', density = True)
	plt.hist(mass, bins, alpha = 0.5, label=r'Hseed', density = True)
	#plt.hist(np.log10(massm), np.log10(bins), alpha = 0.5, label=r'Lseed', density = True)
	plt.hist(massm, bins, alpha = 0.5, label=r'Lseed', density = True)
	plt.legend()
	#plt.xlim(m1, m2)
	#plt.xlabel(r'$\log(M_{\mathrm{BH, seed}}\ [\mathrm{M}_{\odot}])$')
	plt.xlabel(r'$M_{\mathrm{BH, seed}}\ [\mathrm{M}_{\odot}]$')
	plt.ylabel(r'Probability density')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'BHmass_spec_seed_'+str(sn)+'.pdf')
	plt.close()
	
	return bhind

def stellar_env(ds, pbh, vbh, rbh, Rcm):
	z = ds['Redshift']
	h = ds['HubbleParam']
	if Rcm==0:
		R = rbh/1e3 * (1+z) * h
	else:
		R = Rcm
	ad = ds.sphere(pbh, (R, 'kpccm/h'))
	ms = np.sum(np.array(ad[('PartType4','Masses')].to('Msun')))
	r = R*1e3/h/(1+z)
	rhos = ms/(4*np.pi/3 * r**3)
	if ms>0:
		vel = np.array(ad[('PartType4','Velocities')].to('km/s'))
		dv = vel - vbh
		dv = dv.T
		if len(dv[0])>1:
			sig = np.sum([np.var(v) for v in dv])**0.5
		else:
			sig = np.sum(dv**2)**0.5
		return [rhos, sig]
	else:
		return [0, 0]

def BH_mass(rep, sn, ds, X = 0.76, rbh = 10, nbin = 30, Rcm = 0.05, mode=0, rho0=2.5, beta=2, typef=0):
	z = ds['Redshift']
	h = ds['HubbleParam']
	ad = ds.all_data()
	gas = np.sum(ad[('PartType0', 'Masses')].to('Msun'))
	star = np.sum(ad[('PartType4', 'Masses')].to('Msun'))
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType3' for x in keys])
	if tag>0:
		stbh = np.array(ad[('PartType3', 'Star Type')])
		selbh = stbh>=3e4
		bh = np.sum(ad[('PartType3', 'Masses')][selbh].to('Msun'))
		print('Mtot: ', gas+star+bh)
		sid = ad[('PartType3','ParticleIDs')][selbh]
		mass = np.array(ad[('PartType3', 'Masses')][selbh].to('Msun'))
		bhmseed = np.array(ad[('PartType3', 'BH Mass')])[selbh]
		bhm = np.array(ad[('PartType3', 'BH_Mass')])[selbh]
		
		if typef==0:
			bhnseed = np.array(ad[('PartType3', 'BH_NProgs')])#[selbh]
			bhpos = np.array(ad[('PartType3','Coordinates')].to('kpccm/h'))#[selbh]
			bhvel = np.array(ad[('PartType3','Velocities')].to('km/s'))#[selbh]
		else:
			bhnseed = np.array(ad[('PartType3', 'BH_NProgs')])[selbh]
			bhpos = np.array(ad[('PartType3','Coordinates')].to('kpccm/h'))[selbh]
			bhvel = np.array(ad[('PartType3','Velocities')].to('km/s'))[selbh]
		#bhsound = np.array(ad[('PartType3', 'BH_sound')])[selbh]
		#print('cs: ', sumy(bhsound**0.5))
		#bhgasv = np.array(ad[('PartType3', 'BH_gasvel')])[selbh]
		#print('vgas: ', sumy(bhgasv))
		#bhvtheta = np.array(ad[('PartType3', 'BH_vtheta')])[selbh]
		#print('vtheta: ', sumy(bhvtheta))
		nbh = len(mass)
		
		plt.figure()
		sel = bhnseed > 1
		nbbh = np.sum(sel)
		bhn = bhnseed[sel]
		plt.hist(bhn, np.linspace(0.5, 7.5, 8), alpha=0.5)
		#plt.legend()
		#plt.xlim(m1, m2)
		plt.xlabel(r'$N_{\mathrm{Progs}}$')
		plt.ylabel(r'Counts')
		plt.yscale('log')
		plt.tight_layout()
		plt.savefig(rep+'BBH_NProgs_'+str(sn)+'.pdf')
		plt.close()
		
		if mode==0:
			lenv = []
			for pos, vel in zip(bhpos, bhvel):
				env = stellar_env(ds, pos, vel, rbh, Rcm)
				lenv.append(env)
			rhos0, sigs0 = np.array(lenv).T
			totxt(rep+'bbh_env0.txt', [rhos0, sigs0], 0,0,0)
			#bhpos = bhpos[sel]
			#bhvel = bhvel[sel]	
			rhos, sigs = rhos0[sel], sigs0[sel]
			totxt(rep+'bbh_env.txt', [rhos, sigs], 0,0,0)
		#else:
		rhos0, sigs0 = np.array(retxt(rep+'bbh_env0.txt', 2, 0,0))
		rhos, sigs = np.array(retxt(rep+'bbh_env.txt', 2, 0,0))
		if rho0>0:
			rhos0 *= (rhos0/rho0)**beta * (rhos0>rho0) + 1 * (rhos0<=rho0)
			rhos *= (rhos/rho0)**beta * (rhos>rho0) + 1 * (rhos<=rho0)
		
		lm = np.geomspace(1e2, 1e4, 100)
		lg = [1.5, 1.0, 0.5, 0.0]
		lsig = sigma_inf(lm)
		
		plt.figure()
		i = 0
		for g in lg:
			lrho = rho_inf(lm, g)
			plt.loglog(lm, lrho/Msun * PC**3, ls = lls[i], lw=llw[i], label=r'$\gamma='+str(g)+'$')
			i += 1
		plt.scatter(bhm[sel], rhos)
		plt.xlabel(r'$M\ [\mathrm{M}_{\odot}]$')
		plt.ylabel(r'$\rho_{\mathrm{inf}}\ [\mathrm{M_{\odot}\ pc^{-3}}]$')
		plt.ylim(1, 1e5)
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'rhoinf_MB.pdf')
		plt.close()
	
		plt.figure()
		plt.loglog(lm, lsig/1e5)
		plt.scatter(bhm[sel], sigs)
		plt.xlabel(r'$M\ [\mathrm{M}_{\odot}]$')
		plt.ylabel(r'$\sigma_{\mathrm{inf}}\ [\mathrm{km\ s^{-1}}]$')
		plt.tight_layout()
		plt.savefig(rep+'siginf_MB.pdf')
		plt.close()
	
		sel_ = rhos > 0
		sel0_ = rhos0 > 0
		print('Embedded fraction: {:.3f} ({:.3f})'.format(np.sum(sel_)/nbbh, np.sum(sel0_)/nbh))
		
		le = [0.99, 0.9, 0]
		ltcol = np.array([[coaltime(m/2, m/2, sig*1e5, rho*Msun/PC**3, e) 
			for m, rho, sig in zip(bhm[sel][sel_], rhos[sel_], sigs[sel_])] for e in le])
		
		plt.figure()
		i = 0
		for tcol in ltcol:
			plt.hist(np.log10(tcol/1e6), nbin, density=True, alpha=0.5, label='$e={}$'.format(le[i]))
			i += 1
		plt.legend()
		plt.xlabel(r'$\log(t_{\rm GW}\ [\rm Myr])$')
		plt.ylabel(r'Probability density')
		plt.tight_layout()
		plt.savefig(rep+'BBH_tGW_'+str(sn)+'.pdf')
		plt.close()
		
		rb = np.linspace(-3, 6, 37)
		print('Rho_star: ', sumy(rhos[sel_]), sumy(rhos0[sel0_]))
		plt.figure()
		plt.hist(np.log10(rhos[sel_]), rb, alpha=0.5, density=True, label='BBHs')
		plt.hist(np.log10(rhos0[sel0_]), rb, density=True, label='All BHs', histtype='step', lw=1.5, color='k')
		plt.xlabel(r'$\log(\rho_{\star}\ [\mathrm{M_{\odot}\ pc^{-3}}])$')
		plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'BBH_rhos_'+str(sn)+'.pdf')
		plt.close()
		
		sb = np.linspace(0, 200, 41)
		print('Sigma_star: ', sumy(sigs[sel_]), sumy(sigs0[sel0_]))
		plt.figure()
		plt.hist(sigs[sel_], sb, alpha=0.5, density=True, label='BBHs')
		plt.hist(sigs0[sel0_], sb, density=True, label='All BHs', histtype='step', lw=1.5, color='k')
		plt.xlabel(r'$\sigma_{\star}\ [\mathrm{km\ s^{-1}}]$')
		plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'BBH_sigs_'+str(sn)+'.pdf')
		plt.close()
		
		print('MBH_seed: ', sumy(bhmseed))
		print('Nseed: ', sumy(bhnseed))
		print('MBH/MBH_seed: ', sumy(bhm/bhmseed))
		print('MBH/Mdyn: ', sumy(bhm/mass))
		print('MBH_seed/Mdyn', sumy(bhmseed/mass))
		bhacrb = np.array(ad[('PartType3', 'BH_AccretionLength')])[selbh]
		print('Hsml', sumy(bhacrb))
		#bhmgas = np.array(ad[('PartType3', 'BH_Ngas')])[selbh]#/h * UM/Msun
		#print('Ngas', sumy(bhmgas))
		#bhmstar = np.array(ad[('PartType3', 'BH_Nstar')])[selbh]#/h * UM/Msun
		#print('Ngstar', sumy(bhmstar))
		bhsinf = np.array(ad[('PartType3', 'BH_sigma_inf')])[selbh]/1e5
		print('sigma_inf', sumy(bhsinf))
		bhrinf = np.array(ad[('PartType3', 'BH_rho_inf')])[selbh] * PC**3/Msun #X / PROTON
		print('rho_inf', sumy(bhrinf))
		print('MBH/MBH_seed (total): {}'.format(np.sum(bhm) / np.sum(bhmseed)))
		print('NBH_seed/NBH: {}'.format(np.sum(bhnseed) / bhm.shape[0]))
		return [z, np.sum(bhmseed), np.sum(bhm), sid, bhacrb]
	else:
		return [z, 0, 0]

def bbhpick(d, ds):
	ad = ds.all_data()
	st = ad[('PartType3', 'Star Type')]
	ids = ad[('PartType3','ParticleIDs')]
	lid = ids[st<3e4]
	id1 = d[1]
	id2 = d[6]
	sel1 = np.array([np.sum(lid==ID)<=0 for ID in id1])
	sel2 = np.array([np.sum(lid==ID)<=0 for ID in id2])
	return np.logical_and(sel1, sel2)

def details(sn, z, rep = './', rep_='./', ncol = 13, npro = 168, base = 'bhmergers', rep0 = 'blackhole_details/', ext = '.txt', abd=0, h=0.6774):
	out = []
	for i in range(npro):
		fname = rep+rep0+base+'_'+str(i)+ext
		l = retxt_nor(fname, ncol, 0, 0)
		out.append(l)
	out = np.array(np.hstack(out))
	sel = out[0] <= 1/(1+z)
	#print(np.sum(sel))
	out = out.T[sel]
	out = out.T
	if abd>0:
		ds = yt.load(rep+'snapshot_'+str(sn).zfill(3)+'.hdf5')
		sel = bbhpick(out, ds)
		out = out.T[sel]
		out = out.T
	totxt(rep_+base+'_'+str(sn)+ext, out,0,0,0)
	la, lm1, lm2, le = out[0], out[2]*UM/h, out[7]*UM/h, out[12]
	small = [la, lm1/Msun, lm2/Msun, le]
	if ncol>13:
		ld1, ld2, la1, la2 = out[ncol-6], out[ncol-5], out[ncol-2], out[ncol-1]
		small += [ld1, ld2, la1, la2]
	lab = ['a_BBH', 'M1 [Msun]', 'M2', 'e', 'nH_SF_1', 'nH_SF_2', 'a_SF_1', 'a_SF_2']
	totxt(rep_+'binary_cat_'+str(sn)+ext,small,lab,1,0)
	return out

nb = 20

def splothis(d, rep, s = 128, alp = 0.7, mode=0, selab=0, ne = nb, lsens=[], lins=[], fage=0, LIGO = [], lnsf=[]):
	gamma = d['gamma']
	if mode==0:
		rg = d['rg']
		le0 = d['le0']
		plt.figure()
		his0, edge, pat = plt.hist(le0, np.linspace(0, 1, ne+1), alpha=alp)#, density=True)
		plt.xlabel(r'$e$')
		#plt.ylabel(r'Probability density')
		plt.ylabel('Counts')
		plt.yscale('log')
		plt.tight_layout()
		plt.savefig(rep+'ecc_dis.pdf')
		plt.close()
		
		le = d['le_hdb']
		plt.figure()
		plt.hist(le0, np.linspace(0, 1, ne+1), alpha=alp, label='All', histtype='step', lw=1.5, color='k')
		his1, edge, pat = plt.hist(le, np.linspace(0, 1, ne+1), alpha=alp, label='HDB-only')#, density=True)
		plt.xlabel(r'$e$')
		#plt.ylabel(r'Probability density')
		plt.ylabel('Counts')
		plt.yscale('log')
		plt.legend(loc=2)
		plt.tight_layout()
		plt.savefig(rep+'ecc_dis_hdb.pdf')
		plt.close()
		
		base = 0.5*(edge[:-1]+edge[1:])
		plt.figure()
		plt.plot(base[his0>0], his1[his0>0]/his0[his0>0])
		plt.xlabel(r'$e$')
		plt.ylabel(r'Reduction rate')
		plt.tight_layout()
		plt.savefig(rep+'ecc_rat.pdf')
		plt.close()
		
		la0 = d['la0']
		plt.figure()
		plt.hist(np.log10(la0), ne, alpha=alp, density=True)
		plt.xlabel(r'$\log(a_{0}\ [\mathrm{pc}])$')
		plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.tight_layout()
		plt.savefig(rep+'a0_dis.pdf')
		plt.close()
	else:
		fsk = d['sk']
	dz = d['dz']
	if fage>0:
		ldage = d['Dage']
		sel = ldage>0
		print('fraction of age difference: {:.3f}'.format(np.sum(sel)/len(ldage)), sumy(ldage[sel]))
		print('fraction of common parent: {:.3f}'.format(d['com_parent']/len(ldage)))
		plt.figure()
		plt.hist(np.log10(ldage[sel])-6, ne, alpha=alp, density=True)
		plt.xlabel(r'$\log(\Delta t_{\mathrm{BH}}\ [\mathrm{Myr}])$')
		plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.tight_layout()
		plt.savefig(rep+'Dage_dis.pdf')
		plt.close()
		
		Ebref = 0.5*Msun*1e12
		plt.figure()
		his, edge, pat = plt.hist(np.log10(d['Eb']), ne, alpha=alp)#, density=True)
		y1, y2 = np.min(his)/2, np.max(his)*1.3
		plt.plot(np.log10([Ebref]*2), [y1, y2], 'k--', label=r'$\log[(1/2)\langle M_{\star}\sigma_{\star}^{2}\rangle]$')
		plt.legend()
		plt.ylim(y1, y2)
		plt.xlabel(r'$\log(E_{\rm b}\ [\mathrm{erg}])$')
		plt.ylabel(r'Counts')
		#plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.tight_layout()
		plt.savefig(rep+'DEb_dis.pdf')
		plt.close()
		
		ldage = d['Dage_']
		sel = ldage>0
		print('fraction of delayed binary formation: {:.3f}'.format(np.sum(sel)/len(ldage)), sumy(ldage[sel]))
		sel_ = ldage > 1e7
		print('fraction of BBHs from separate halos: {:.3f}'.format(np.sum(sel_)/len(ldage)))
		#print('fraction of common parent: {:.3f}'.format(d['com_parent']/len(ldage)))
		plt.figure()
		plt.hist(np.log10(ldage[sel])-6, ne, alpha=alp, density=True)
		plt.xlabel(r'$\log(\Delta t_{\mathrm{BH}}\ [\mathrm{Myr}])$')
		plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.tight_layout()
		plt.savefig(rep+'DtBBH_dis.pdf')
		plt.close()
		
		nsfb = np.linspace(2, 3.5, ne)
		plt.figure()
		ld1, ld2 = d['ldens']
		plt.hist(np.log10(ld1), nsfb, alpha=alp/2, density=True, label='Primary')
		plt.hist(np.log10(ld2), nsfb, alpha=alp/2, density=True, label='Secondary')
		if len(lnsf)>0:
			plt.hist(np.log10(lnsf), nsfb, alpha=alp/2, density=True, label='All BHs')
		plt.xlabel(r'$\log(n_{\mathrm{SF}}\ [\mathrm{cm^{-3}}])$')
		plt.ylabel(r'Probability density')
		plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'SFdens_dis.pdf')
		plt.close()
	
	lx, ly, yer = d['Mrate']
	plt.figure()
	plt.errorbar(lx[ly>0], ly[ly>0]*1e9, yerr=yer[ly>0]*1e9, xerr=dz/2, fmt='*')
	plt.xlabel(r'$z_{\mathrm{BBH}}$')
	plt.ylabel(r'$\dot{n}_{\mathrm{BBH}}\ [\mathrm{Gpc^{-3}\ yr^{-1}}]$')
	plt.yscale('log')
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'merger_rate.pdf')
	else:
		plt.savefig(rep+'merger_rate_col.pdf')
	plt.close()

	lzGW, lz, lm, lm1, lm2, le = d['events']
	
	ltdelay = np.array([TZ(z1) - TZ(z2) for z1, z2 in zip(lzGW, lz)])/YR/1e6
	plt.figure()
	plt.hist(np.log10(ltdelay), ne, alpha=alp, density=True)
	plt.xlabel(r'$\log(t_{\mathrm{GW}}\ [\mathrm{Myr}])$')
	plt.ylabel(r'Probability density')
	#plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'tGW_dis.pdf')
	plt.close()
	
	#if mode==0:
	tickloc = np.linspace(np.min(lzGW), np.max(lzGW), 5)[:-1]
	ticklab = ['{:.2f}'.format(np.log10(TZ(z)/1e9/YR)) for z in tickloc]
	plt.figure(figsize=(12,4))
	ax = plt.subplot(131)
	plt.scatter(lzGW, lm, c=lz, cmap = plt.cm.coolwarm, s=s, alpha=alp)
	cb = plt.colorbar()
	cb.ax.set_title('$z_{\mathrm{BBH}}$')#, y=-0.1)
	plt.xlabel(r'$z_{\mathrm{GW}}$')
	plt.ylabel(r'$M_{1}+M_{2}\ [\mathrm{M_{\odot}}]$')
	plt.yscale('log')
	#plt.xlim(xlim)
	#if mode==0:
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(tickloc)
	ax2.set_xticklabels(ticklab)
	ax2.set_xlabel(r'$\log(t\ [\mathrm{Gyr}])$')
	
	ax = plt.subplot(132)
	plt.scatter(lzGW, lm, c=lm2/lm1, cmap = plt.cm.cool, s=s, alpha=alp, label=r'$\gamma={}$'.format(gamma))
	cb = plt.colorbar()
	cb.ax.set_title('$M_{2}/M_{1}$')#, y=-0.11)
	plt.clim(0, 1)
	plt.xlabel(r'$z_{\mathrm{GW}}$')
	plt.yscale('log')
	#plt.xlim(xlim)
	xmax = np.max(lzGW)
	ymax = np.max(lm)
	if (mode!=0) and (fsk>0):
		plt.text(xmax*0.2, ymax*0.9, r'$\gamma={}$'.format(gamma)+r', $f_{\mathrm{fk}}='+str(fsk)+'$')
	else:
		plt.text(xmax*0.7, ymax*0.9, r'$\gamma={}$'.format(gamma))
	#if mode==0:
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(tickloc)
	ax2.set_xticklabels(ticklab)
	ax2.set_xlabel(r'$\log(t\ [\mathrm{Gyr}])$')
	
	ax = plt.subplot(133)
	plt.scatter(lzGW, lm, c=le, s=s, alpha=alp)
	cb = plt.colorbar()
	if mode==0:
		cb.ax.set_title('$e$')#, y=-0.1)
		plt.clim(0, 1)
	else:
		cb.ax.set_title('$a_{0}\ [\mathrm{AU}]$')
	plt.xlabel(r'$z_{\mathrm{GW}}$')
	plt.yscale('log')
	#plt.xlim(xlim)
	#if mode==0:
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(tickloc)
	ax2.set_xticklabels(ticklab)
	ax2.set_xlabel(r'$\log(t\ [\mathrm{Gyr}])$')
	
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'bhmerger_'+str(selab)+'_'+str(rg)+'.pdf')
	else:
		#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.savefig(rep+'bhmerger_col.pdf')
	#plt.show()
	plt.close()
	
	if len(lsens)>0:
		lfp = d['fpeak']
		lstrain = 2*d['strain']*d['fpeak']
		f1, f2 = 1e-4, 1e3
		y1, y2 = 1e-24, 1e-16
		plt.figure()
		#plt.scatter(lfp, lstrain, c=lzGW, s=s, alpha=alp, cmap=plt.cm.cool, zorder=2)
		for lf, lh22, m1, m2 in zip(d['lf'], d['lh22'], lm1, lm2):
			if (m1<9e3) and (m2<9e3):
				plt.plot(lf, 2*lf*lh22, alpha=0.2, color='gray', zorder=0)
			elif m2>9e3:
				plt.plot(lf, 2*lf*lh22, alpha=0.4, color='r', zorder=1)
			else:
				plt.plot(lf, 2*lf*lh22, alpha=0.4, color='orange', zorder=1)
		#"""
		lex = []
		i = 0
		for detec in d['snr']:
			ex = detec['examp']
			m, z, lf, lh22, fpeak, hpeak = ex
			#plt.plot(lf, 2*lf*lh22, color='red', lw = 2.5, ls=lls[i], alpha=0.4, zorder=1)
			lex.append([fpeak, 2*fpeak*hpeak, m, z])
			print('Typical source for {}: M = {} Msun, z = {}'.format(lins[i], m, z))
			i += 1
		#lex = np.array(lex).T
		#n1 = len(lfp)
		#n2 = len(lex[0])
		#lfp = np.hstack([lfp, lex[0]])
		#lstrain = np.hstack([lstrain, lex[1]])
		#lzGW = np.hstack([lzGW, lex[3]])
		#lm = np.hstack([lm, lex[2]])
		#ms = np.hstack([np.ones(n1)*s, np.ones(n2)*s*1])
		ms = s
		#"""
		plt.scatter(lfp, lstrain, c=lzGW, s=ms, alpha=alp, cmap=plt.cm.cool, zorder=2)
		#plt.scatter(lfp, lstrain, c=lm, s=ms, alpha=alp, cmap=plt.cm.cool, zorder=2)
		cb = plt.colorbar()
		cb.ax.set_title(r'$z_{\rm{GW}}$')
		#cb.ax.set_title(r'$M\ [\mathrm{M_{\odot}}]$')
		i = 0
		#if len(LIGO)>0:
		#	plt.loglog(LIGO[0], LIGO[1]*LIGO[0]**0.5, 'k-', label='LIGO-Livingston', lw=0.5)
		for sens, ins in zip(lsens,lins):
			plt.plot(sens[0], sens[1]*sens[0]**0.5, label=ins, ls=lls[i], color='k', lw=llw[i])
			i += 1
		z = 7
		DL = DZ(z)*(1+z)
		M1, M2 = 1e5, 1e5
		fRD = fRD_a(afin(0,0,M1,M2), M1+M2)/(1+z)
		lf = np.geomspace(f1, fRD*2, 1000)
		h22 = PMwf(lf, M1, M2, DL, z)[0]
		z = 7
		DL = DZ(z)*(1+z)
		M1, M2 = 1e4, 100
		fRD = fRD_a(afin(0,0,M1,M2), M1+M2)/(1+z)
		lf_ = np.geomspace(f1, fRD*2, 1000)
		h22_ = PMwf(lf_, M1, M2, DL, z)[0]
		plt.loglog(lf, 2*h22*lf, alpha=0.6, color='r', lw=3, ls='--', zorder=3)
		plt.text(1e-3, 3e-19, 'DCBH-DCBH', color='r', bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 1})
		plt.loglog(lf_, 2*h22_*lf_, alpha=0.6, color='orange', lw=3, ls='-.', zorder=3)
		plt.text(7e-3, 1e-22, r'$\mathrm{DCBH-BH_{\mathrm{PopIII}}}$', color='orange', bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 1})
		plt.xlabel(r'$f\ [\rm{Hz}]$')
		plt.ylabel(r'Characteristic strain')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(f1, f2)
		plt.ylim(y1, y2)
		#plt.legend()
		plt.legend(bbox_to_anchor=(0., 0.79, 1., 1.), loc=3, ncol=2, mode="expand", borderaxespad=0.)
		plt.tight_layout()
		plt.savefig(rep+'GWevents_'+str(selab)+'.pdf')
		#plt.savefig(rep+'GWevents_'+str(selab)+'.png')
		plt.close()

lfmt = ['^', 'o', 's', 'D'] * 2

def mplothis(ld, rep, z1 = 4, z2 = 20, nz = nb, alp = 0.7, cs = oi, m1 = 6e2, m2 = 3e3, nm = nb, ne = nb, mode = 0, lref = [], llab=[], Mtot=1e5, lMs=[],dind=0,lins=[]):
	plt.figure()
	i = 0
	y1 = 100
	y2 = 0
	x2 = 0
	for d in ld:
		if mode==2:
			fsk = d['sk']
		else:
			gamma = d['gamma']
		if False:#len(d['snr'])>0:
			lx, ly, yer = d['snr'][dind]['GWrate']
		else:
			lx, ly, yer = d['GWrate']
		dz = d['dz']
		if y1>np.min(ly[ly>0]):
			y1 = np.min(ly[ly>0])
		if y2<np.max(ly):
			y2 = np.max(ly)
		if x2<np.max(lx):
			x2 = np.max(lx)
		if mode==0:
			plt.errorbar(lx[ly>0], ly[ly>0]*1e9, yerr=yer[ly>0]*1e9, xerr=dz/2, fmt=lfmt[i], label=r'$\gamma={:.1f}$'.format(gamma), alpha = alp)
		elif mode==1:
			plt.plot(lx[ly>0], ly[ly>0]*1e9, ls=lls[i], label=r'$\gamma={}$'.format(gamma))
			#plt.errorbar(lx[ly>0], ly[ly>0]*1e9, yerr=yer[ly>0]*1e9, ls=lls[i], label=r'$\gamma={}$'.format(gamma))
		else:
			plt.errorbar(lx[ly>0], ly[ly>0]*1e9, yerr=yer[ly>0]*1e9, xerr=dz/2, fmt=lfmt[i], label=r'$f_{\rm{sk}}='+str(fsk)+'$', alpha= alp)
		i += 1
	#xmin = np.min(lx[ly>0])
	#ymax = np.max(ly)
	#plt.text(xmin, ymax*1, r'$\gamma={}$'.format(gamma))
	y1, y2 = min(0.01, y1*1e9*0.5), max(y2*1e9 * 2, 1e3)
	x1, x2 = 0, min(x2*1.2, 10)
	#plt.fill_between([x1, x2], [1, 1], [1e2, 1e2], facecolor='gray', alpha=0.5, label=r'Kinugawa14, $z_{\mathrm{GW}}\simeq 0$')
	if len(lref)>0:
		i = 0
		for ref in lref:
			print('Pop III stellar mass density ({}): {:.0f} Msun Mpc^-3, ratio: {:.3f}'.format(llab[i], lMs[i], lMs[i]/Mtot))
			plt.plot(ref[0], ref[1], color='k', ls=lls[i], label=llab[i], lw=2)
			if i<2:
				plt.plot(ref[0], ref[1]*Mtot/lMs[i], color='k', ls=lls[i], lw=1)
			i += 1
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$z_{\mathrm{GW}}$')
	plt.ylabel(r'$\dot{n}_{\mathrm{GW}}\ [\mathrm{yr^{-1}\ Gpc^{-3}}]$')
	plt.yscale('log')
	plt.legend(loc=2)
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'GW_rate_dens_'+str(selab)+'.pdf')
	else:
		plt.savefig(rep+'GW_rate_dens_col_'+str(selab)+'.pdf')
	plt.close()
	
	zlim = ld[0]['Drate'][1]
	y2 = 0
	plt.figure()
	i = 0
	for d in ld:
		if mode==2:
			fsk = d['sk']
		else:
			gamma = d['gamma']
		dz = d['dz']
		if len(d['snr'])>0:
			lx = d['snr'][dind]['Dzbase']
			ly = d['snr'][dind]['Dlr']
		else:
			lx = d['Dzbase']
			ly = d['Dlr']
		if y2<np.max(ly):
			y2 = np.max(ly)
		if mode==2:
			plt.plot(lx[ly>0], ly[ly>0], ls = lls[i], label=r'$f_{\rm{sk}}='+str(fsk)+'$')
		else:
			plt.plot(lx[ly>0], ly[ly>0], ls = lls[i], label=r'$\gamma={}$'.format(gamma))
		i += 1
	y1, y2 = 0.1, y2 * 2
	x1, x2 = 0, zlim
	#plt.fill_between([x1, x2], [1, 1], [1e2, 1e2], facecolor='gray', alpha=0.5, label=r'Kinugawa14, $z_{\mathrm{GW}}\simeq 0$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.xlabel(r'$z_{\mathrm{lim}}$')
	if len(d['snr'])>0:
		inst = lins[dind]
		plt.ylabel(r'$\dot{N}_{\mathrm{'+inst+r'}}(z_{\rm{GW}}<z_{\rm{lim}})\ [\mathrm{yr^{-1}}]$')
	else:
		plt.ylabel(r'$\dot{N}_{\mathrm{GW}}(z_{\rm{GW}}<z_{\rm{lim}})\ [\mathrm{yr^{-1}}]$')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'GW_rate_'+str(selab)+'_'+inst+'.pdf')
	else:
		plt.savefig(rep+'GW_rate_col_'+str(selab)+'.pdf')
	plt.close()
		
	bins = np.linspace(z1, z2, nz+1)
	plt.figure()
	i = 0
	for d in ld:
		gamma = d['gamma']
		if len(d['snr'])>0:
			lz = d['snr'][dind]['Dlz']
		else:
			lz = d['Dlz']
		print('z_merge(D): ', sumy(lz))
		plt.hist(lz, bins, alpha = alp*2/3, color=cs[i], label=r'$\gamma={}$'.format(gamma), density=True)
		i += 1
	plt.xlabel(r'$z_{\mathrm{BBH}}$')
	plt.ylabel(r'Probability density')
	#plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'dis_Dzmerge_'+str(selab)+'.pdf')
	else:
		plt.savefig(rep+'dis_Dzmerge_col_'+str(selab)+'.pdf')
	plt.close()
	
	bins = np.linspace(0, zlim, nz+1)
	plt.figure()
	i = 0
	for d in ld:
		gamma = d['gamma']
		if len(d['snr'])>0:
			lz = d['snr'][dind]['DlzGW']
		else:
			lz = d['DlzGW']
		print('z_GW(D): ', sumy(lz))
		plt.hist(lz, bins, alpha = alp*2/3, color=cs[i], label=r'$\gamma={}$'.format(gamma), density=True)
		i += 1
	plt.xlabel(r'$z_{\mathrm{GW}}$')
	plt.ylabel(r'Probability density')
	#plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'dis_DGW_'+str(selab)+'.pdf')
	else:
		plt.savefig(rep+'dis_DGW_col_'+str(selab)+'.pdf')
	plt.close()
	
	m1 = 1e5
	m2 = 0
	for d in ld:
		if len(d['snr'])>0:
			lm = d['snr'][dind]['Dlm']
		else:
			lm = d['Dlm']
		ml, mh = np.min(lm), np.max(lm)
		if ml<m1:
			m1 = ml
		if mh>m2:
			m2 = mh
	bins = np.log10(np.geomspace(m1, m2, nm+1))
	plt.figure()
	i = 0
	for d in ld:
		gamma = d['gamma']
		if len(d['snr'])>0:
			lm = d['snr'][dind]['Dlm']
		else:
			lm = d['Dlm']
		print('M(D): ', sumy(lm))
		plt.hist(np.log10(lm), bins, alpha = alp*2/3, color=cs[i], label=r'$\gamma={}$'.format(gamma), density=True)
		i += 1
	plt.xlabel(r'$\log(M\ [\mathrm{M}_{\odot}])$')
	plt.ylabel(r'Probability density')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if mode==0:
		plt.savefig(rep+'dis_DM_'+str(selab)+'.pdf')
	else:
		plt.savefig(rep+'dis_DM_col_'+str(selab)+'.pdf')
	plt.close()
	
	if mode==0:
		bins = np.linspace(0, 1, ne+1)
		plt.figure()
		i = 0
		for d in ld:
			gamma = d['gamma']
			if len(d['snr'])>0:
				le = d['snr'][dind]['Dle']
			else:
				le = d['Dle']
			print('ecc(D): ', sumy(le))
			plt.hist(le, bins, alpha = alp*2/3, color=cs[i], label=r'$\gamma={}$'.format(gamma), density=True)
			i += 1
		plt.xlabel(r'$e$')
		plt.ylabel(r'Probability density')
		plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'dis_Decc_'+str(selab)+'.pdf')
		plt.close()
		
		bins = np.linspace(0, 1, ne+1)
		plt.figure()
		i = 0
		for d in ld:
			gamma = d['gamma']
			if len(d['snr'])>0:
				lq = d['snr'][dind]['Dlq']
			else:
				lq = d['Dlq']
			print('q(D): ', sumy(lq))
			plt.hist(lq, bins, alpha = alp*2/3, color=cs[i], label=r'$\gamma={}$'.format(gamma), density=True)
			i += 1
		plt.xlabel(r'$q\equiv M_{2}/M_{1}$')
		plt.ylabel(r'Probability density')
		#plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'dis_Dq_'+str(selab)+'.pdf')
		plt.close()
		
	if len(ld[0]['snr'])>0:
		for i in range(len(ld[0]['snr'])):
			inst = lins[i]
			plt.figure()
			bins = np.linspace(0, 5, ne+1)
			j = 0
			for d in ld:
				gamma = d['gamma']
				lsnr = d['snr'][i]['snr']
				lsnr = np.log10(lsnr[lsnr>0])
				plt.hist(lsnr, bins, alpha = alp*2/3, color=cs[j], label=r'$\gamma={}$'.format(gamma), density=True)
				j += 1
			plt.xlabel(r'$\log(\rm{SNR})$')
			plt.ylabel(r'Probability density')
			plt.yscale('log')
			plt.legend()
			plt.tight_layout()
			plt.savefig(rep+'dis_DSNR_'+inst+'_'+str(selab)+'.pdf')
			plt.close()
				
import numpy.random as nrdm

def gen_ap(n, rmax, rmin=0.1):
	y1 = np.log10(rmin)
	y2 = np.log10(rmax)
	r = nrdm.uniform(size=n)
	return 10**(r*(y2-y1)+y1)

#plt.figure()
#la = gen_ap(1000, 20, 0.1)
#plt.hist(la, 20)
#plt.show()

def mergerhis(l0, ds, mode = 0, gamma = 1.5, H = 17.5, nth = 1e1, beta = 2.0, X = 0.76, bfrac=1, rg = 0, zlim = 5, dz = 0.5, mrel = ms_mbh_default, zsn=4, lsens=[], snrlim=7, ffac=0.9, fage=0, V=V, h=0.6774, eps=0.02, vcir=0, Mfs = 1, amin = 0.1, a0 = 10*PC, seed=233, ms=350, fej = 0.43, bh_self=0, m0=586, mfac=1):
	h = ds['HubbleParam']
	lz = 1/l0[0] - 1
	ncol = len(l0)
	sel = lz >= zsn
	l = l0.T[sel]
	l = l.T
	lz = 1/l[0] - 1
	le = l[12]
	#efac = (1+le)/(1-le)
	lr0 = 2.8*eps/h/(1+lz) * 1e3 * 2
	#lr0 = 2.8*eps/h*1e3 
	
	nrdm.seed(seed)
	la0 = gen_ap(len(l[0]), lr0/(1-le), amin)
	lm1 = l[2]*UM/h#/Msun
	lm2 = l[7]*UM/h#/Msun
	#leps = lm1*lm2*Msun**2*GRA/(2*la0*PC)
	
	out = {}
	
	#Eb = GRA*lm1*lm2/(2*la0*PC) * (1-le)/(1+le)
	if ncol>21:
		if bh_self==0:
			Eb = -l[ncol-7]*(1-fej)**2
		else:
			Eb = -l[ncol-7]*(lm1/m0/Msun)*(lm2/m0/Msun)
	else:
		if bh_self==0:
			Eb = GRA*(ms*Msun)**2/(2*la0*PC) * (1-le)/(1+le)
		else:
			Eb = GRA*lm1*lm2/(2*la0*PC) * (1-le)/(1+le)
	Ecut = vcir**2*Mfs*Msun*0.5
	#print('amax = {} PC'.format((GRA*125*125*Msun**2)/Ecut/PC))
	sel = Eb > Ecut
	print('Fraction of HDB: {:.3f}'.format(np.sum(sel)/len(l[0])))
	out['le0'] = le
	out['la0'] = la0
	
	l = l.T[sel]
	l = l.T
	
	Nbbh = len(l)
	lm1 = l[2]*UM/h/Msun * mfac
	lm2 = l[7]*UM/h/Msun * mfac
	lz = 1/l[0] - 1
	le = l[12]
	out['le_hdb'] = le
	#lr0 = 2.8*eps/h/(1+lz) * 1e3
	#la0 = lr0*(1+le)/(1-le**2)
	out['Nbbh'] = Nbbh
	out['gamma'] = gamma
	out['rg'] = rg
	out['dz'] = dz
	out['snr'] = []
	
	if fage>0:
		lz1 = 1/l[ncol-2]-1
		lz2 = 1/l[ncol-1]-1
		lid1 = l[ncol-4]
		lid2 = l[ncol-3]
		ldens1 = l[ncol-6]
		ldens2 = l[ncol-5]
		if ncol>21:
			lEb = np.abs(l[ncol-7])
		else:
			lEb = Eb[sel]
		out['Eb'] = lEb
		#print(sumy(ldens1), sumy(ldens2))
		out['ldens'] = [ldens1[ldens1>0], ldens2[ldens2>0]]
		out['com_parent'] = sum(lid1==lid2)
		Dage = np.array([abs(TZ(z1)-TZ(z2))/YR for z1, z2 in zip(lz1, lz2)])
		Dage_ = np.array([abs(TZ(z)-TZ(max(z1,z2)))/YR for z, z1, z2 in zip(lz, lz1, lz2)])
		out['Dage'] = Dage
		out['Dage_'] = Dage_
		
	tlim = TZ(0)/1e9/YR
	lm = lm1 + lm2
	
	if len(lz)==0:
		out['Mrate'] = [[], [], []]
	else:	
		#z1, z2 = np.min(lz), np.max(lz)
		z1 = max(np.min(lz)-dz/2, 0)
		z2 = max(np.max(lz)+dz/2, z1 + dz)
		nzb = min(int((z2-z1)/dz), 20)
		if nzb<=0:
			nzb = 1
		zbase = np.linspace(z1, z2, nzb+1)
		dt = np.array([TZ(zbase[i]) - TZ(zbase[i+1]) for i in range(nzb)])/YR
		ly, lx0 = np.histogram(lz, zbase)
		yer = ly**0.5/dt/V
		ly = ly/dt/V
		lx = (lx0[1:] + lx0[:-1])/2
		out['Mrate'] = [lx, ly, yer]

	lsig0 = sigma_inf(lm)
	lrho0 = rho_inf(lm, gamma, mrel)
	if mode==0:
		lt = (np.array([TZ(z) for z in lz])/YR + np.array([tGW(m1+m2, m1, m2, e, H, gamma, a0) for m1, m2, e in zip(lm1, lm2, le)]))/1e9
		lsig = lsig0
		lrho = lrho0
	else:
		"""
		lrho1 = rho_inf(lm, gamma, mrel, rmax)
		lrho2 = rho_inf(lm, gamma, mrel, rmin)
		lrhou, lrhod = lrho1, lrho2
		if np.sum(lrhou)<np.sum(lrhod):
			lrhou, lrhod = lrho2, lrho1
		#print(sumy(lrhou/lrhod))
		rhoth = nth * PROTON / X
		lb = (l[14]/rhoth)**beta * l[14] * (l[14]>rhoth) + l[14] * (l[14]<=rhoth)
		lb = lb * (lb<=lrhou) + lrhou * (lb>lrhou)
		lb += (lrhod-lb) * (lb<lrhod)
		lrho = lb + lrhod * (lb<=0)
		"""
		lrho = rho_inf(lm, gamma, mrel, bfrac)
		lsig = l[13] + lsig0 * (l[13]<=0)
		lt = (np.array([TZ(z) for z in lz])/YR + np.array([coaltime(m1, m2, sig, rho, e, H) for m1, m2, sig, rho, e in zip(lm1, lm2, lsig, lrho, le)]))/1e9
	
	lagw = np.array([astar_GW(m1, m2, sig, rho, e) for m1, m2, sig, rho, e in zip(lm1, lm2, lsig, lrho, le)])
	
	sel = lt<tlim
	print('Nmerger: ', np.sum(sel))
	#print(lt, tlim, le)

	"""
	if mode!=0:
		rat0 = lsig/lsig0
		sel0 = rat0>=0
		rat1 = lrho/lrho0
		sel1 = rat1>=0
		#print('sigma_inf: ', l[13], sumy(l[13]))
		#print('rho_inf: ', lb, sumy(l[14]))
		#print('sig_ratio: ', sumy(rat0[sel0]))
		#print('rho_ratio: ', sumy(rat1[sel1]))
		plt.figure()
		plt.scatter(lm[sel0], rat0[sel0], c=lm2[sel0]/lm1[sel0], label=r'$\sigma_{\mathrm{sim}}/\sigma_{\mathrm{est}}$', s=128, alpha=0.7, marker='o')
		plt.scatter(lm[sel1], rat1[sel1], c=np.log10(lm2/lm1)[sel1], label=r'$\rho_{\mathrm{sim}}/\rho_{\mathrm{est}}$', s=128, alpha=0.7, marker='^')
		cb = plt.colorbar()
		cb.ax.set_title('$M_{2}/M_{1}$')#, y=-0.11)
		plt.yscale('log')
		plt.xscale('log')
		plt.ylim(1e-3, 1e3)
		plt.xlabel(r'$M_{1}+M_{2}\ [\mathrm{M_{\odot}}]$')
		plt.ylabel(r'Ratio')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'ratio_gamma'+str(gamma)+'.pdf')
		plt.close()
	"""

	d = np.array([l[1], lt, lz, lm1, lm2, le, np.ones(len(lz)), lagw]).T
	d = d[sel]
	if rg!=0:
		d = np.array(sorted(d, key=lambda x:x[0]))
		nme = len(d)
		for i in range(0, nme):
			ID = d[i][0]
			t = d[i][1]
			z = d[i][2]
			if d[i][6]<=0:
				continue
			for j in range(i+1, nme):
				if ID != d[j][0]:
					break
				if ((d[j][1]>t) or (d[j][6]<=0)) and (d[j][2]>z):
					temp = d[i][3]
					d[i][3] -= d[j][4]
					d[i][1] = (TZ(z)/YR + tGW(temp+d[i][4], d[i][3], d[i][4], d[i][5], H, gamma, a0))/1e9
					t = d[i][1]
					if rg > 1:
						if d[i][4] > d[j][4] * (rg-1):
							d[j][6] = 0
					d[j][3] = temp
					d[j][1] = (TZ(d[j][2])/YR + tGW(temp+d[j][4], d[j][3], d[j][4], d[j][5], H, gamma, a0))/1e9
		ltag = d[:,6]
		d = d[ltag>0]
	lID, lt, lz, lm1, lm2, le, ltag, lagw = d.T
	sel = lt<tlim
	d = d[sel]
	lID, lt, lz, lm1, lm2, le, ltag, lagw = d.T
	lm = lm1 + lm2
	lzGW = np.array([ZT(t) for t in np.log10(lt)])
	Nmg = len(lm)
	out['Nmg'] = Nmg
			
	#Nbbh = d['Nbbh']
	if Nmg>0:
		out['Nhh'] = np.sum((lm1>9e3)*(lm2>9e3))
		out['Nhl'] = np.sum((lm1>9e3)*(lm2<9e3))
		out['Nll'] = Nmg-out['Nhh']-out['Nhl']
	else:
		out['Nhh'] = out['Nhl'] = out['Nll'] = 0
			
	if len(lzGW)==0:
		out['Drate'] = [0, zlim]
	else:
		out['strain'] = np.array([PMwf(np.array([fRD_a(afin(0,0,m1,m2),m1+m2)*ffac/(1+z)]), m1, m2, DZ(z)*(1+z), z)[0][0] for m1, m2, z in zip(lm1, lm2, lzGW)])
		out['fpeak'] = np.array([fRD_a(afin(0,0,m1,m2), m1+m2)*ffac/(1+z) for m1, m2, z in zip(lm1, lm2, lzGW)])
		lf = np.geomspace(1e-4, 1e3, 100)
		#out['lh22'] = np.array([PMwf(lf[lf<2*fRD_a(afin(0,0,m1,m2),m1+m2)/(1+z)], m1, m2, DZ(z)*(1+z), z)[0] for m1, m2, z in zip(lm1, lm2, lzGW)])
		lsel = [((fGW(m1+m2, agw, e)/(1+z))<lf) * (lf<2*fRD_a(afin(0,0,m1,m2),m1+m2)/(1+z)) for m1, m2, z, agw, e in zip(lm1, lm2, lzGW, lagw, le)]
		out['lf'] = np.array([lf[sel] for sel in lsel])
		out['lh22'] =  np.array([PMwf(lf, m1, m2, DZ(z)*(1+z), z)[0] for m1, m2, z, lf in zip(lm1, lm2, lzGW, out['lf'])])
		#np.array([lf[lf<2*fRD_a(afin(0,0,m1,m2),m1+m2)/(1+z)] for m1, m2, z in zip(lm1, lm2, lzGW)])
		#print(len(lzGW))
		z1 = max(np.min(lzGW)-dz/2, 0)
		z2 = max(np.max(lzGW)+dz/2, z1 + dz)
		nzb = int((z2-z1)/dz)#, 30)
		if nzb<=0:
			nzb = 1
		zbase = np.linspace(z1, z2, nzb+1)
		#t2, t1 = np.max(lt), np.min(lt)
		#nzb = int((t2-t1)*1e9/ts)
		#if nzb<=0:
		#	nzb = 1
		#ltedge = np.linspace(min(t2+ts/1e15,tlim), t1-ts/1e15, nzb+1)
		#zbase = np.array([ZT(t) for t in np.log10(ltedge)])
		dt = np.array([TZ(zbase[i]) - TZ(zbase[i+1]) for i in range(nzb)])/YR
		ly, lx0 = np.histogram(lzGW, zbase)
		yer = ly**0.5/dt/V
		ly = ly/dt/V
		lx = (lx0[1:] + lx0[:-1])/2
		out['GWrate'] = [lx, ly, yer]
	
		lV = np.array([DZ(x2)**3 - DZ(x1)**3 for x2, x1 in zip(lx0[1:], lx0[:-1])])*4*np.pi/3/MPC**3
		intg = ly/(1+lx) * lV
		lrate = np.cumsum(intg[lx<=zlim])
		if len(lrate)>0:
			rate = lrate[-1]#np.sum(ly[lx<zlim] / (1+lx[lx<zlim]) * lV[lx<zlim])
			out['Dle'] = le[lzGW<zlim]
			out['Drate'] = [rate, zlim]
			out['Dlz'] = lz[lzGW<zlim]
			out['DlzGW'] = lzGW[lzGW<zlim]
			out['Dzbase'] = lx[lx<=zlim]
			out['Dlr'] = lrate
			out['Dlm'] = lm[lzGW<zlim]
			lq = lm2[lzGW<zlim]/lm1[lzGW<zlim]
			lq = lq * (lq<=1) + 1/lq * (lq>1)
			out['Dlq'] = lq
		else:
			print(lrate, lx, zlim, intg[lx<=zlim], intg)
			out['Drate'] = [0, zlim]
	
	out['events'] = [lzGW, lz, lm, lm1, lm2, le]
	
	lsnr = []
	if len(lsens)>0:
		dd0 = np.array([lzGW, lm1, lm2, lz, le, lagw]).T
		for sens in lsens:
			lzGW, lm1, lm2, lz, le, lagw = dd0.T
			snr = np.array([SNR(m1, m2, sens, z, agw, e0) for m1, m2, z, agw, e0 in zip(lm1, lm2, lzGW, lagw, le)])
			sel = snr>=snrlim
			dd = dd0[sel].T
			dsnr = {}
			dsnr['snr'] = snr
			lzGW, lm1, lm2, lz, le, lagw = dd
			if len(lzGW)<=0:
				dsnr['Drate'] = [0, zlim]
				dsnr['frac'] = 0
				lsnr.append(dsnr)
				continue
			lm = lm1 + lm2
			z1 = max(np.min(lzGW)-dz/2, 0)
			z2 = max(np.max(lzGW)+dz/2, z1 + dz)
			nzb = int((z2-z1)/dz)#, 30)
			if nzb<=0:
				nzb = 1
			zbase = np.linspace(z1, z2, nzb+1)
			dt = np.array([TZ(zbase[i]) - TZ(zbase[i+1]) for i in range(nzb)])/YR
			ly, lx0 = np.histogram(lzGW, zbase)
			yer = ly**0.5/dt/V
			ly = ly/dt/V
			lx = (lx0[1:] + lx0[:-1])/2
			dsnr['GWrate'] = [lx, ly, yer]
			lV = np.array([DZ(x2)**3 - DZ(x1)**3 for x2, x1 in zip(lx0[1:], lx0[:-1])])*4*np.pi/3/MPC**3
			intg = ly/(1+lx) * lV
			lrate = np.cumsum(intg)
			rate = lrate[-1]
			Nmg_ = len(lzGW)
			dsnr['Nmg'] = Nmg_
			dsnr['frac'] = Nmg_/Nmg
			dsnr['Dle'] = le
			dsnr['Drate'] = [rate, z2]
			dsnr['Dlz'] = lz
			dsnr['DlzGW'] = lzGW
			dsnr['Dzbase'] = lx
			dsnr['Dlr'] = lrate
			dsnr['Dlm'] = lm
			lq = lm2/lm1
			lq = lq * (lq<=1) + 1/lq * (lq>1)
			dsnr['Dlq'] = lq
			
			m1, m2 = np.median(lm1), np.median(lm2)
			agw, e0 = np.median(lagw), np.median(le)
			z = np.median(lzGW)
			ex_strain = PMwf(np.array([fRD_a(afin(0,0,m1,m2),m1+m2)*ffac/(1+z)]), m1, m2, DZ(z)*(1+z), z)[0][0]
			ex_fpeak = fRD_a(afin(0,0,m1,m2), m1+m2)*ffac/(1+z)
			fmin = fGW(m1+m2, agw, e0)/(1+z)
			#print('Minimum frequency: {} Hz'.format(fmin))
			lf = np.geomspace(fmin, 1e3, 100)
			ex_lh22 = PMwf(lf[lf<2*fRD_a(afin(0,0,m1,m2),m1+m2)/(1+z)], m1, m2, DZ(z)*(1+z), z)[0]
			ex_lf = lf[lf*(1+z)<2*fRD_a(afin(0,0,m1,m2),m1+m2)]
			dsnr['examp'] = [m1+m2, z, ex_lf, ex_lh22, ex_fpeak, ex_strain]
			lsnr.append(dsnr)
		out['snr'] = lsnr
	return out

def stellarb(sn, rep = './', base = 'snapshot', ext = '.hdf5', gamma = 1.5, sk = 0.1, zlim = 5, dz = 0.1, frm = 0.57, seed = 1234, mode=0):#, fbh = 0.8):
	np.random.seed(seed)
	ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	ad = ds.all_data()
	stbh = np.array(ad[('PartType3', 'Star Type')])
	selbh = stbh>=3e4
	ms = np.array(ad[('PartType3','Masses')].to('Msun'))[selbh]
	lz = 1/np.array(ad[('PartType3','StellarFormationTime')])[selbh] - 1
	sel = np.array(ad[('PartType3','BH Type')])[selbh]==1
	ms = ms[sel]
	lz = lz[sel]
	#mmaxbh = np.array(ad[('PartType3', 'BH Max mass')][sel])
	#minput = ms*fbh
	out = []
	for m, z in zip(ms, lz):
		if sk>0:
			l = binary_gen(m, gamma=gamma, sk=sk, frm=frm, adis=0, mode=mode)
		else:
			l = binary_gen(m, gamma=gamma, frm=frm, adis=1, mode=mode)
		out.append([z*np.ones(len(l[0])), *l])
	d = np.hstack(out)
	out = {}
	lz, lt, lm1, lm2, la0, le = d
	lm = lm1 + lm2
	ltGW = (lt + np.array([TZ(z) for z in lz])/YR)/1e9
	tlim = TZ(0)/1e9/YR
	sel = ltGW < tlim
	d = np.array([ltGW, lz, lm, lm1, lm2, la0]).T[sel]
	ltGW, lz, lm, lm1, lm2, la0 = d.T
	lzGW = np.array([ZT(t) for t in np.log10(ltGW)])
	out['events'] = [lzGW, lz, lm, lm1, lm2, la0]
	Nmg = len(lzGW)
	out['Nmg'] = Nmg
	out['gamma'] = gamma
	out['sk'] = sk
	out['dz'] = dz
	
	if len(lzGW)==0:
		out['Drate'] = [0, zlim]
	else:
		z1 = max(np.min(lzGW)-dz/2, 0)
		z2 = max(np.max(lzGW), z1 + dz)
		nzb = min(int((z2-z1)/dz), 300)
		if nzb<=0:
			nzb = 1
		zbase = np.linspace(z1, z2, nzb+1)
		dt = np.array([TZ(zbase[i]) - TZ(zbase[i+1]) for i in range(nzb)])/YR
		ly, lx0 = np.histogram(lzGW, zbase)
		yer = ly**0.5/dt/V
		ly = ly/dt/V
		lx = (lx0[1:] + lx0[:-1])/2
		out['GWrate'] = [lx, ly, yer]
		
		lV = np.array([DZ(x2)**3 - DZ(x1)**3 for x2, x1 in zip(lx0[1:], lx0[:-1])])*4*np.pi/3/MPC**3
		intg = ly/(1+lx) * lV
		lrate = np.cumsum(intg[lx<=zlim])
		rate = lrate[-1]#np.sum(ly[lx<zlim] / (1+lx[lx<zlim]) * lV[lx<zlim])
		out['Dla0'] = la0[lzGW<zlim]
		out['Drate'] = [rate, zlim]
		out['Dlz'] = lz[lzGW<zlim]
		out['DlzGW'] = lzGW[lzGW<zlim]
		out['Dzbase'] = lx[lx<=zlim]
		out['Dlr'] = lrate
		out['Dlm'] = lm[lzGW<zlim]

	if len(lz)==0:
		out['Mrate'] = [[], [], []]
	else:		
		z1 = max(np.min(lzGW)-dz/2, 0)
		z2 = max(np.max(lzGW), z1 + dz)
		nzb = min(int((z2-z1)/dz), 300)
		if nzb<=0:
			nzb = 1
		zbase = np.linspace(z1, z2, nzb+1)
		dt = np.array([TZ(zbase[i]) - TZ(zbase[i+1]) for i in range(nzb)])/YR
		ly, lx0 = np.histogram(lz, zbase)
		yer = ly**0.5/dt/V
		ly = ly/dt/V
		lx = (lx0[1:] + lx0[:-1])/2
		out['Mrate'] = [lx, ly, yer]
	return out

def bhtrack(sni, snf, rep = './', base = 'snapshot', ext = '.hdf5', sp = 1):
	ds0 = yt.load(rep+base+'_'+str(sni).zfill(3)+ext)
	ad0 = ds0.all_data()
	bhm = np.array(ad0[('PartType3', 'BH_Mass')])
	bhmacc = np.array(ad0[('PartType3', 'BH_Macc')])
	bhid = ad0[('PartType3','ParticleIDs')]
	maxm = np.max(bhm)
	sel = bhm==maxm
	ID = bhid[sel][0]
	lz = []
	lmbh = []
	lmacc = []
	lz.append(ds0['Redshift'])
	lmbh.append(maxm)
	lmacc.append(bhmacc[sel][0])
	sn = sni - sp
	while sn>=snf:
		ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
		keys = ds.field_list
		tag = np.sum([x[0] == 'PartType3' for x in keys])
		if tag>0:
			ad = ds.all_data()
			bhid = ad[('PartType3','ParticleIDs')]
			sel = bhid==ID
			if np.sum(sel)<=0:
				print('Lose track at z = {:.3f}, sn = {}'.format(ds['Redshift'],sn))
				break
			bhm = np.array(ad[('PartType3', 'BH_Mass')])[sel][0]
			bhmacc = np.array(ad[('PartType3', 'BH_Macc')])[sel][0]
			lz.append(ds['Redshift'])
			lmbh.append(bhm)
			lmacc.append(bhmacc)
			sn = sn - sp
		else:
			print('Lose track at z = {:.3f}, sn = {}'.format(ds['Redshift'],sn))
			break
	return [lz, lmbh, lmacc]

def bhhis(l, sni, rep = './', base = 'snapshot', ext = '.hdf5'):
	ds0 = yt.load(rep+base+'_'+str(sni).zfill(3)+ext)
	h = ds0['HubbleParam']
	ad0 = ds0.all_data()
	bhm = np.array(ad0[('PartType3', 'BH_Mass')])
	bhm0 = np.array(ad0[('PartType3', 'BH Mass')])
	bhmacc = np.array(ad0[('PartType3', 'BH_Macc')])
	bhid = ad0[('PartType3','ParticleIDs')]
	mdyn = np.array(ad0['PartType3', 'Masses'].to('Msun'))
	maxm = np.max(bhm)
	sel = bhm==maxm
	mdynmax = mdyn[sel][0]
	print('MBH_max = {} Msun (Mdyn = {} Msun)'.format(maxm, mdynmax))
	ID = bhid[sel][0]
	mseed = bhm0[sel][0]
	sel = l[1]==ID
	d = l.T[sel]
	d = np.array(sorted(d, key=lambda x:x[0]))
	d = d.T
	lz = 1/d[0]-1
	lm1 = d[2]*UM/h/Msun
	lm2 = d[7]*UM/h/Msun
	lm = lm1 + lm2
	lmm = np.cumsum(lm2)
	lm2_ = np.hstack([[0], lm2[:-1]])
	lm += (lm - lmm < mseed) * lm2_
	lmacc = lm - lmm - mseed #lm1[1:] - (lm1[:-1]+lm2[:-1])
	#print(d[0][1:]-d[0][:-1])
	
	plt.figure()
	plt.plot(lz, lm, label='Total')
	plt.plot(lz, lmm, '--', label='Merge', marker='.')
	plt.plot(lz, lmacc, '-.', label='Accreted')
	plt.yscale('log')
	plt.ylim(1e2, np.max(lm)*1.2)
	plt.legend()
	plt.xlabel(r'$z$')
	plt.ylabel(r'$M\ [\mathrm{M_{\odot}}]$')
	plt.tight_layout()
	plt.savefig(rep+'Maxhis.pdf')
	plt.close()
	return ID
	
#Mup = lambda z: 2.5e7*((1+z)/10)**-1.5
#Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074 #1e6*((1+z)/10)**-2
	
def group_info(ds, obj, rep = './', mode = 0, mfac = 1.0, rfac = 0.2, mlab = 0):
	z = ds['Redshift']
	if mlab==0:
		Mmin = 0.0#Mdown(z)
	else:
		Mmin = Mup(z)
	lh = obj.halos
	datah = []
	for halo in lh:
		M = halo.masses['total'].to('Msun')*mfac
		if M<Mmin:
			continue
		Ms_ = halo.masses['stellar'].to('Msun')
		Mg = halo.masses['gas'].to('Msun')
		if len(halo.galaxy_index_list)>0:
			gal = halo.central_galaxy
			pos = gal.pos.to('kpccm/h')
			Ms = gal.masses['stellar'].to('Msun')
			R = gal.radii['baryon_half_mass'].to('kpccm/h')
		else:
			pos = halo.pos.to('kpccm/h')
			R = halo.radii['baryon_half_mass'].to('kpccm/h')
			if R<=0:
				R = halo.radii['total_half_mass'].to('kpccm/h')
			Ms = Ms_
		ad = ds.sphere(pos, (R*rfac, 'kpccm/h'))
		stbh = np.array(ad[('PartType3', 'Star Type')])
		selbh = stbh>=3e4
		lbh = ad[('PartType3', 'BH_Mass')][selbh]
		if len(lbh)>0:
			if mode==0:
				Mbh = np.max(lbh)
			else:
				Mbh = np.sum(lbh)
		else:
			Mbh = 0
		datah.append([Mbh, Ms, R.to('kpc'), M, Ms_, Mg])
	datah = np.array(datah).T
	d = {}
	d['z'] = z
	d['data'] = datah
	return d
	
def gal_info(ds, obj, rep = './', mode = 0, mfac = 1.0, rfac = 0.2, Mmin = 32*585):
	z = ds['Redshift']
	lg = obj.galaxies
	datag = []
	for gal in lg:
		Ms = gal.masses['stellar'].to('Msun')*mfac
		if Ms<Mmin:
			continue
		Mg = gal.masses['gas'].to('Msun')
		pos = gal.pos.to('kpccm/h')
		R = gal.radii['baryon_half_mass'].to('kpccm/h')
		ad = ds.sphere(pos, (R*rfac, 'kpccm/h'))
		stbh = np.array(ad[('PartType3', 'Star Type')])
		selbh = stbh>=3e4
		lbh = ad[('PartType3', 'BH_Mass')][selbh]
		if len(lbh)>0:
			if mode==0:
				Mbh = np.max(lbh)
			else:
				Mbh = np.sum(lbh)
		else:
			Mbh = 0
		datag.append([Mbh, Ms, R.to('kpc'), Mg])
	datag = np.array(datag).T
	d = {}
	d['z'] = z
	d['data'] = datag
	print('Minimum stellar mass: {:.3f} 10^4 Msun'.format(np.min(datag[1])/1e4))
	return d

#Mbh_Ms0 = lambda x: 10**(1.25*np.log10(x)-4.8)
Mbh_Ms1 = lambda x: 10**(1.4*np.log10(x)-6.45)
Mbh_Ms2 = lambda x: 10**(1.05*np.log10(x)-4.1)
	
def linear(x, b0, b1):
	return b0 + x*b1
	
def get_range(l, fac = 0.7):
	return np.min(l)*fac, np.max(l)/fac
	
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
	
datarep = '/media/friede/Seagate Backup Plus Drive/HighzBH/'
#datarep = './'

if __name__ == "__main__":
	sn = 23
	#npro = 168
	npro = 1152
	#npro = 672
	#npro = 408
	#rlab = 'NSFDBKzoom'
	#rlab = 'FDzoom_HR'
	#rlab = 'FDzoom'
	rlab = 'LB20' #'FDbox'
	#rlab = 'PS (this work)'
	#rep = 'NSFDBKzoom_Hseed/'
	#rep = 'NSFDBKzoom_Lseed/'
	#rep = 'FDzoom_NBH/'
	#rep = 'FDzoom_Lseed_comp/'
	#rep = 'FDzoom_Lseed_ADC/'#NDF/'#local/'
	#rep = 'FDzoom_Lseed_HR/'
	#rep = 'FDzoom_Lseed/'
	#rep = 'FDNzoom_Lseed/'
	#rep = 'FDzoom_Hseed/'
	#rep = 'FDNzoom_Hseed_FT/'
	#rep = 'FDNzoom_Hseed_UVB/'
	#rep = 'FDzoomHDB_Hseed/'
	#rep = 'SFtest/'
	rep = 'FDbox_Lseed/'
	repi = datarep+rep
	box = 1
	cr = 'comp/'
	gamma = 1.5
	ligoo2 = 0
	vcir = 0#1e6
	fej = 0#0.43
	mfac = 1#0.25
	abd = 1
	#subrep = 'UCD/'
	#subrep = 'HDB/'
	subrep = './'
	if abd>0:
		subrep = 'abd/'
	rep_ = rep+subrep
	if not os.path.exists(rep_):
		os.makedirs(rep_)
	
	zlim = 10
	snrlim = 10
	dz = 0.3
	selab = 0
	glab = 0
	ins = 1
	bfrac = 1.0
	fage = 1
	ncol = 19+2#+1
	nr = 20+1
	seed = 0
	
	gamma0 = 0
	gamma1 = 1.5
	ng = 4
	lg = np.linspace(gamma1, gamma0, ng)
	#lg = [1.5, 2, 2.5]
	
	fitlab = 0
	snap = 0
	fabd = 1

	Mh_Ms_Hseed = lambda x: 10**((np.log10(x)-1.99)/0.14)
	Mh_Ms_Lseed = lambda x: 10**((np.log10(x)-1.40)/0.13)
	if selab==0:
		rf = ms_mbh_default
	else:
		if seed==0:
			rf = Mh_Ms_Lseed
		else:
			rf = Mh_Ms_Hseed

	mode = int(sys.argv[1])
	if mode==-1:
		data = loaddata(sn, sn, repi)
		ds0 = data['ds']
		#plt.figure()
		#ax = new_3Dax(111)
		#dis3D(data, 0, ax)
		#plt.tight_layout()
		#plt.show()
		#plt.close()
	else:
		ds0 = yt.load(repi+'snapshot_'+str(sn).zfill(3)+'.hdf5')
	zsn = ds0['Redshift']
	h = ds0['HubbleParam']
	#print(SFgas(sn, rep))
	#print(ds0.field_list)
	xlm = [25, 4]

	if snap>0:
		M1, M2, lnsf, fabd = smass(ds0, rep_, abd=abd)
		print('Snapshot: M(PopIII) = {:.0f} Msun Mpc^-3, M(PopII) = {:.0f} Msun Mpc^-3'.format(M1/V, M2/V))
		print('Fraction of real Pop III BHs: {:.3f}'.format(fabd))
		MpopIII = M1

	sfr0III = np.array(retxt_nor(repi+'popIII_sfr.txt', 4, 0, 0))
	sfr1III = np.array(retxt_nor(cr+'popIII_sfr.txt', 4, 0, 0))
	sfr0II = np.array(retxt_nor(repi+'popII_sfr.txt', 4, 0, 0))
	sfr1II = np.array(retxt_nor(cr+'popII_sfr.txt', 4, 0, 0))

	lm1 = sfr0III[1] * sfr0III[2]
	lm2 = sfr0II[1] * sfr0II[2]
	M1 = np.sum(lm1[sfr0III[0]<=1/(1+zsn)])/V
	M2 = np.sum(lm2[sfr0II[0]<=1/(1+zsn)])/V
	print('OTF: M(PopIII) = {:.0f} Msun Mpc^-3, M(PopII) = {:.0f} Msun Mpc^-3'.format(M1, M2))
	if snap==0:
		MpopIII = M1*V
		lnsf = []
	lm1 = sfr1III[1] * sfr1III[2]
	lm2 = sfr1II[1] * sfr1II[2]
	M1 = np.sum(lm1[sfr1III[0]<=1/(1+zsn)])/(4/0.6774)**3/fac0
	M2 = np.sum(lm2[sfr1II[0]<=1/(1+zsn)])/(4/0.6774)**3/fac0
	print('Old: M(PopIII) = {:.0f} Msun Mpc^-3, M(PopII) = {:.0f} Msun Mpc^-3'.format(M1, M2))
	
	ref0 = np.array(retxt(cr+'SF16.txt', 3, 0, 0))
	ref1 = np.array(retxt(cr+'edges_bottom.txt', 2, 0, 0))
	ref2 = np.array(retxt(cr+'edges_top.txt', 2, 0, 0))
	ref3 = np.array(retxt(cr+'sf18_sfrd.txt', 3, 0, 0)) 
	ref4 = np.array(retxt(cr+'sf18_sfrd_evo.txt', 3, 0, 0)) 
	ref5 = np.array(retxt(cr+'popIII_sfr_s2_skx.txt', 4, 0, 0))
	ref6 = np.array(retxt(cr+'popII_sfr_s2_skx.txt', 4, 0, 0))

	lm1 = ref5[1] * ref5[2]
	lm2 = ref6[1] * ref6[2]
	M1 = np.sum(lm1[ref5[0]<=1/(1+zsn)])/(4/0.6774)**3
	M2 = np.sum(lm2[ref6[0]<=1/(1+zsn)])/(4/0.6774)**3
	print('JJ: M(PopIII) = {:.0f} Msun Mpc^-3, M(PopII) = {:.0f} Msun Mpc^-3'.format(M1, M2))
	
	des0 = np.array(retxt(cr+'deS_popIII_sfr.txt',2,0,0))
	des1 = np.array(retxt(cr+'deS_popIII1_sfr.txt',2,0,0))
	i0 = interp1d(des0[0],np.log(des0[1]))
	i1 = interp1d(des1[0],np.log(des1[1]))
	zb0 = np.linspace(3, 12, 30)
	zb1 = np.linspace(12.1, 29, 50)
	sfrds0 = np.exp(i0(zb0))
	sfrds1 = np.exp(i0(zb1)) + np.exp(i1(zb1))
	zb = np.hstack([zb0, zb1])
	sfrds = np.hstack([sfrds0, sfrds1]) * 0.5

	htwg0 = np.array(retxt(cr+'Htwg_popIII_sfr.txt',2,0,0))
	htwg1 = np.array(retxt(cr+'Htwg_popII_sfr.txt',2,0,0))
	lz0 = np.array([ZT(np.log10(x)) for x in htwg0[0]])
	lz1 = np.array([ZT(np.log10(x)) for x in htwg1[0]])
	#print(sumy(lz0), sumy(lz1))
	hsfr0 = interp1d(lz0, np.log(htwg0[1]))
	hsfr1 = interp1d(lz1, np.log(htwg1[1]))
	zh0 = np.linspace(7.1, 28,  50)
	zh1 = np.linspace(5.7, 7, 20)
	hsfrd0 = np.exp(hsfr0(zh0)) + np.exp(hsfr1(zh0))
	hsfrd1 = np.exp(hsfr1(zh1))
	zh = np.hstack([zh1, zh0])
	hsfrd = np.hstack([hsfrd1, hsfrd0])

	lzJ = np.linspace(4, 25, 1000)
	thrat = Mup(lzJ)/Mdown(lzJ)
	Jcritu = ((thrat-1)/6.96)**(1/0.47)/(4*np.pi)
	Jcritd = ((10-1)/6.96)**(1/0.47)/(4*np.pi) * np.ones(lzJ.shape[0])
	Jc = 0.85
		
	lz = 1/sfr0II[0]-1
	J1, J2, JLW = LWbg(lz, sfr0III[3], sfr0II[3])
	plt.figure()
	plt.plot(lz, J1, '--', label='Pop III')
	plt.plot(lz, J2, '-.', label='Pop II')
	plt.plot(lz, JLW, 'k-', label='Total')
	plt.fill_between(lzJ, Jcritu, Jcritd, fc='gray', alpha=0.5, label=#r'$J_{\rm crit}$'+', for\n'+
	r'$\hat{M}_{\rm th}^{\rm mol}(J_{\rm LW,bg})\sim 10M_{\rm th}^{\rm mol}-M_{\rm th}^{\rm atom}$')
	plt.plot([4, 25], [Jc]*2, 'k:', label=r'$k_{\rm H_{2},des}=k_{\rm H_{2},form}$')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$J_{\mathrm{LW,bg}}\ [10^{-21}\mathrm{erg\ s^{-1}\ cm^{-2}\ Hz^{-1}\ sr^{-1}}]$')
	plt.yscale('log')
	#plt.xlim(xlm)
	plt.xlim(4, 25)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'JLW_bg.pdf')
	plt.close()
	
	totxt(rep+'FLW_z.txt',[lz, 4*np.pi*JLW])

	Nion2, Nion3 = 1e47, 1e48
	t2, t3 = 1e7, 3e6
	l2 = Nion2 * sfr0II[3] * t2 #*(1+lz)**3
	l3 = Nion3 * sfr0III[3] * t3 #*(1+lz)**3
	lt = l2+l3
	lC = CH(lz)
	lrefu = SFR_reion(lz, lC, fesc=0.7) * Nion2 * t2
	lrefd = SFR_reion(lz, lC, fesc=0.1) * Nion2 * t2
	lref = SFR_reion(lz, lC, fesc=0.3) * Nion2* t2
	y1, y2 = 1e47, 1e54
	zion = 6.
	plt.figure()
	plt.plot(lz, l3, '--', label='Pop III')
	plt.plot(lz, l2, '-.', label='Pop II')
	plt.plot(lz, lt, 'k-', label='Total')
	#plt.plot(lz, lref, 'k:')
	plt.fill_between(lz, lrefu, lrefd, fc='gray', alpha=0.5,  label='Reionization\n'+r'$(f_{\rm esc}\sim 0.1-0.7)$')
	#plt.plot([zion]*2, [y1, y2], )
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\dot{n}_{\rm ion}\ \rm [s^{-1}\ Mpc^{-3}]$')
	plt.yscale('log')
	#plt.xlim(xlm)
	plt.xlim(4, 25)
	plt.ylim(y1, y2)
	plt.legend(loc=3)
	plt.tight_layout()
	plt.savefig(rep+'Nion_bg.pdf')
	plt.close()

	metrat = np.array(retxt(cr+'metal_ratio.txt',6,0,0))
	lzm, rma = metrat[:2]
	ith = 1
	rvf = metrat[2+ith]
	plt.figure()
	plt.plot(lz, J1/JLW, label=r'$J_{\rm LW,bg}$')
	plt.plot(lz, l3/lt, '--', label=r'$\dot{n}_{\rm ion}$')
	plt.plot(lzm, rma, '-.', label=r'$\langle Z\rangle$')
	plt.plot(lzm, rvf, ':', label=r'$\mathcal{F} (Z>10^{-4}\ \mathrm{Z_{\odot}})$')
	plt.plot(xlm, [0.1]*2, 'k-', lw=0.5)
	plt.plot(xlm, [0.2]*2, 'k-', lw=0.5)
	plt.plot(xlm, [0.5]*2, 'k-', lw=0.5)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm PopIII}$')
	#plt.yscale('log')
	#plt.xlim(xlm)
	plt.xlim(4, 25)
	plt.ylim(0, 1)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'fPopIII_bg.pdf')
	plt.close()
	

	if fitlab!=0:
		fit = curve_fit(sfrtf, 1/sfr0III[0]-1, sfr0III[3])#, [2, -4, 12, -6])
		para = fit[0]
		print('Pop III:', para)
		zcut = 4
		lt = np.linspace(10, TZ(zcut)/YR/1e6, 10000)/1e3
		lz = np.array([ZT(t) for t in np.log10(lt)])
		lsfrd = sfrtf(lz, *para)
		Mextra = np.trapz(lsfrd, lt)*1e9
		print('Extrapolated total Pop III mass density: {:.0f} Msun Mpc^-3'.format(Mextra))
		MpopIII = Mextra*V
	else:
		Mextra = MpopIII/V

	lMs = []
	lt = np.array([TZ(z) for z in zb])/YR
	Ms = abs(np.trapz(sfrds, lt)*2)
	lMs.append(Ms)
	lMs.append(Ms)
	lt = np.array([TZ(z) for z in lz0])/YR
	lMs.append( abs(np.trapz(htwg0[-1], lt)) )

	J13 = np.array(retxt(cr+'Johnson2013.txt',2))
	T07 = np.array(retxt(cr+'Tornatore07.txt',2))
	sfrNLW = np.array(retxt_nor(cr+'popIII_sfr_NLW.txt', 4, 0, 0))
	
	Xu16 = np.array(retxt(cr+'Xu2016.txt',2))
	t0 = TZ(7.6)/YR/1e6
	lt = (t0-Xu16[0])/1e3
	lzx16 = [ZT(np.log10(t)) for t in lt]
	
	S18d = np.array(retxt(cr+'sarmento18_d.txt',2))
	S18u = np.array(retxt(cr+'sarmento18_u.txt',2))
	S18m = (S18d[1] + S18u[1])*0.5
	S18s = np.abs(S18u[1]-S18d[1])*0.5
	
	CJ20 = np.array(retxt(cr+'chatterjee20.txt',2))
	
	plt.figure()
	z2 = 26
	psfr(sfr0III, rlab+' (Pop III)', mode=1)
	if fitlab!=0:
		lzf = np.linspace(0, z2, 100)
		plt.plot(lzf, sfrtf(lzf, *para), lw=3, alpha = 0.5, label=r'Fit')#: $a(1+z)^{b}/\{1+[(1+z)/c]^{d}\}$,'+'\n'+r'$a={:.0f}$, $b={:.0f}$, $c={:.0f}$, $d={:.0f}$'.format(*para))
	#psfr(sfr1III, 'LB19', '--', V=fac0*(4/0.6774)**3)
	#psfr(sfrNLW, 'No LW (FDzoom)', (0, (10, 5)), mode=1, fz=1)
	#plt.plot(T07[0], T07[1], ls=(0, (10, 5)), label='TL07', color='c')
	plt.plot(J13[0], J13[1], 'g--', label='JCS13')
	plt.plot(lzx16, Xu16[1], '-.', label='XH16', color='orange')
	#plt.plot(lz0, htwg0[-1], '-.', label='HT16', color='orange')
	#plt.fill_between(S18d[0], S18d[1], S18u[1], fc='r', alpha=0.5)
	plt.plot(*S18d, 'r:', label='SR18')
	plt.plot(*S18u, 'r:')
	#plt.plot(1/ref5[0]-1, ref5[3], ls=(0, (3,1,1,1)), label='JJ19', color='k')
	#plt.plot(zb, sfrds, 'k:', label='deS11')
	plt.plot(1/sfr0II[0]-1, sfr0II[3]+sfr0III[3], ls=(0, (10, 5)), label=rlab+' (total)')
	#plt.plot(ref0[0], 10**ref0[1], 'o', label='FS16: observed', color='k')
	#plt.plot(ref0[0], 10**ref0[2], '^', label='FS16: estimated', color='k')
	#plt.plot(CJ20[0], 10**CJ20[1], 'k-', lw=3, label='CA20')
	lzf = np.linspace(0, 10, 100)
	plt.fill_between(lzf, sfrbf(lzf)*10**0.2, sfrbf(lzf)/10**0.2, facecolor='gray', label='MR14', alpha=0.5)
	plt.yscale('log')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\dot{\rho}_{\star,\rm PopIII}\ [\mathrm{M}_{\odot}\ \mathrm{yr^{-1}\ Mpc^{-3}}]$')
	#plt.ylabel(r'$\mathrm{SFRD}\ [\mathrm{M}_{\odot}\ \mathrm{yr^{-1}\ Mpc^{-3}}]$')
	#plt.xlim(xlm)
	plt.xlim(0, z2)
	plt.ylim(1e-6, 10)
	plt.legend(loc=1, ncol=2)#, fontsize=13)
	plt.tight_layout()
	plt.savefig(rep+'popIIIsfrd.pdf')
	plt.close()

	lm0 = np.cumsum(sfr0III[1]*sfr0III[3])
	plt.figure()
	plt.plot(1/sfr0III[0]-1, lm0)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\rho_{\star,\mathrm{PopIII}}\ [\mathrm{M_{\odot}\ Mpc^{-3}}]$')
	plt.yscale('log')
	plt.xlim(xlm)
	plt.tight_layout()
	plt.savefig(rep+'rho_popIII.pdf')
	plt.close()
	
	if fitlab!=0:
		lc = [0.015, 2.7, 2.9]
		sfrtf_ = sfrd_gen(lc)
		fit = curve_fit(sfrtf_, 1/sfr0II[0]-1, sfr0II[3], [4.3, 1, 11])
		para = fit[0]
		print('Pop II:', para)
	plt.figure()
	psfr(sfr0II, rlab, mode=1)
	#psfr(sfr1II, 'LB19', '--', V=fac0*(4/0.6774)**3)
	plt.plot(1/ref6[0]-1, ref6[3], 'r--', label='JJ19')
	plt.plot(lz1, htwg1[-1], 'k-.', label='HT16')
	if fitlab!=0:
		plt.plot(zb, sfrtf_(zb, *para), lw=3, alpha = 0.5, label=r'Fit: $\frac{a(1+z)^{b}}{1+[(1+z)/c]^{d}}\exp([1-(1+z)^{e}]/f)$'+'\n'+r'$a={:.2f}$, $b={:.2f}$, $c={:.2f}$'.format(*lc)+'\n'+r'$d={:.2f}$, $e={:.2f}$, $f={:.2f}$'.format(*para))
	plt.yscale('log')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\mathrm{SFRD}\ [\mathrm{M}_{\odot}\ \mathrm{yr^{-1}\ Mpc^{-3}}]$')
	plt.xlim(xlm)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'popIIsfrd.pdf')
	plt.close()
	
	lm1 = np.cumsum(sfr0II[1]*sfr0II[3])
	plt.figure()
	plt.plot(1/sfr0II[0]-1, lm1)
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\rho_{\star,\mathrm{PopII}}\ [\mathrm{M_{\odot}\ Mpc^{-3}}]$')
	plt.yscale('log')
	plt.xlim(xlm)
	plt.tight_layout()
	plt.savefig(rep+'rho_popII.pdf')
	plt.close()
	
	lfcol = np.array(retxt('collapse_rat_z.txt',4,0,0))
	fcol0 = interp1d(lfcol[0],np.log10(lfcol[1]))
	fcol1 = interp1d(lfcol[0],np.log10(lfcol[2]))
	rho0 = rhom(1)/Msun*MPC**3
	lsfe = np.cumsum((sfr0II[1]+sfr0III[1])*sfr0II[3]/rho0)
	lsfe_ = (sfr0II[1]+sfr0III[1])*sfr0II[3]/rho0
	plt.figure()
	lz0 = 1/sfr0II[0]-1
	sel = (lz0<np.max(lfcol[0])) * (lz0>np.min(lfcol[0]))
	zbase = lz0[sel]
	eta1 = lsfe[sel]/10**fcol1(zbase)
	eta0 = lsfe[sel]/10**fcol0(zbase)
	eta1_ = np.zeros(len(eta1))
	eta1_[0] = eta1[0]
	eta1_[1:] = lsfe_[sel][1:]/(10**fcol1(zbase[1:])-10**fcol1(zbase[:-1]))
	plt.plot(zbase, eta1_, label=r'PS, instaneous')
	plt.plot(zbase, eta1, label=r'PS: top-hat')
	plt.plot(zbase, eta0, '--', label='Tinker08')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\langle\eta\rangle\equiv \rho_{\star}/\bar{\rho}_{\rm m}$')
	#plt.ylabel(r'$\rho_{\star,\mathrm{PopII}}\ [\mathrm{M_{\odot}\ Mpc^{-3}}]$')
	plt.yscale('log')
	plt.xlim(xlm)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'eta_z.pdf')
	plt.close()
	totxt('eta_z.txt', [zbase, eta1, eta0],0,0,0)
	totxt('eta_z_ins.txt', [zbase, eta1_],0,0,0)
	
	e1 = interp1d(*ref1)
	e2 = interp1d(*ref2)
	lz = np.linspace(5, 25, 100)
	
	zs = np.linspace(15, xlm[1], 100)
	ts = np.array([TZ(z)/1e9/YR for z in zs])
	t0 = TZ(0)/1e9/YR
	zs_ = np.linspace(10, xlm[1], 100)
	sfrds0 = sfrgb(zs_, -2.41-2.09)
	sfrds1 = sfrgb(zs_, -2.41+1.87)
	#sfrds = sfrmd(ts, t0)
	
	if fitlab!=0:
		sfrtf_ = sfrd_gen(lc)
		fit = curve_fit(sfrtf_, 1/sfr0II[0]-1, sfr0II[3]+sfr0III[3], [4.3, 1, 11])
		para = fit[0]
		print('Total:', para)	
	plt.figure()
	psfr_tot(sfr0II, sfr0III, rlab, mode=1)
	#if fitlab!=0:
	#	plt.plot(zb, sfrtf(zb, *para), lw=3, alpha = 0.5, label=r'Fit')#: $\frac{a(1+z)^{b}}{[1+(1+z)/c]^{d}}$'+'\n'+r'$a={:.2f}$, $b={:.2f}$, $c={:.2f}$, $d={:.2f}$'.format(*para))
	plt.plot(1/ref6[0]-1, ref5[3]+ref6[3], 'r--', label='JJ19')
	plt.plot(zh, hsfrd, 'k-.', label='HT16')
	#plt.plot(zs, sfrds, 'k:', label='SL04')
	plt.fill_between(zs_, sfrds0, sfrds1, facecolor=oi[2], label='WJ14', alpha=0.3)
	plt.text(9, 6e-2, 'GRB', color='g', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
	plt.fill_between(zs, sfrbf(zs)*10**0.2, sfrbf(zs)/10**0.2, facecolor=oi[1], label='MP14', alpha=0.4)
	plt.text(7, 5e-3, 'UVLF', color='b', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
	plt.fill_between(lz, e1(lz), e2(lz), facecolor=oi[0], alpha=0.3, label='MJ19')
	plt.text(20, 1e-3, '21-cm', color='orange', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
	plt.plot(ref0[0], 10**ref0[1], 'o', label='SF16: observed', color='k')
	plt.plot(ref0[0], 10**ref0[2], '^', label='SF16: estimated', color='k')
	#plt.fill_between(ref3[0], 10**ref3[1], 10**ref3[2], facecolor=oi[1], alpha=0.5, label=r'SF18: $\alpha=-2.35$')
	#plt.fill_between(ref4[0], 10**ref4[1], 10**ref4[2], facecolor=oi[2], alpha=0.5, label=r'SF18: $\alpha>-2.35$')
	plt.yscale('log')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\mathrm{SFRD}\ [\mathrm{M}_{\odot}\ \mathrm{yr^{-1}\ Mpc^{-3}}]$')
	plt.xlim(xlm)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'sfrd.pdf')
	plt.close()

	if mode==-2:
		obj = recaesar(sn, repi)
		Mmin = 0
		bhmode = 1
		rfac = 0.5
		rela = gal_info(ds0, obj, rep, bhmode, rfac=rfac, Mmin=Mmin)
		plotrela(rela, rep, bhmode, 'gal', sn)
		mlab = 1
		rela = group_info(ds0, obj, rep, bhmode, rfac=rfac, mlab=mlab)#, Mmin=Mmin)
		plotrela(rela, rep, bhmode, 'halo', sn)
		halo_star(rela, rep, sn)
		
		#rela = group_info(ds0, obj, rep, bhmode, rfac=rfac, mlab=0)
		#halo_star(rela, './', sn, Mmin = 0)
	
	demo = 0
	tlw = 1
	wh = 6.5
	ax = [0, 1]
	#ax = [1, 2]
	#ax = [2,1]
	if box==0:
		thk = 400
		cen = np.array([1940, 1960, 1850])
		ext = 1000
	else:
		thk = 800
		cen = np.array([2e3, 2e3, 8e2])
		ext = 4e3
	if demo>0:
		lflag = 0
		ext = 1000
		asp = 4/3
		lss = [1, 8, 64]
	else:
		lflag = 1
		asp = 1
		lss = [1, 32, 64]
	lext = np.array([ext, ext/asp, thk])
	#le, re = [0, 0, 675-thk/2], [4000, 4000, 675+thk/2]
	le, re = cen - lext/2, cen + lext/2
	#le, re = [1500, 1500, 1850], [2500, 2500, 2150]
	#le, re = [1950, 1850, 1940], [2050, 1950, 1970]
	nump = 1e6
	if mode==0:		
		#"""
		pm = 1
		#met = metal(sn, rep, rlab, Zsun=Zsun, mode=1, nump=1e7)
		#print(met)
		ds0 = phase(sn, repi, indm=1, nump=1e7, mode=pm)
		#"""
		if box==1:
			#hc = rep+'halos_0.0_box.ascii'
			hc = rep+'halos_0.0_box_'+str(sn)+'.ascii'
			nf = 11
			hd0 = rockstar(np.array(retxt(hc,nf,19,0)))
		else:
			hd0 = caesar_obj(recaesar(sn, repi), 0)
		if tlw>0:
			bh = 1
			temp_bg(sn, ds0, le, re, ax, rep = rep_, mode=bh, lss=lss, lflag=lflag, wh=wh, nump=nump)
			metal_bg(sn, ds0, le, re, ax, rep = rep_, mode=1, Zsun=Zsun, nump=nump, s=8)
			if box==1:
				denspro(sn, ds0, hd0, le, re, ax, nb = ext, rep = rep_, mode=0, lflag=lflag, nump=0, hs=1)
				#a=0
			else:
				denspro(sn, ds0, hd0, le, re, ax, nb = ext, rep = rep_, mode=1, lflag=lflag, nump=0, hs=0)
			if demo==0:
				LW_bg(sn, ds0, le, re, ax, rep = rep_, mode=0, nump=nump)
		else:
			denspro(sn, ds0, hd0, le, re, ax, nb = ext, rep = rep_, mode=1, lflag=lflag, nump=nump)
		
		#denspro(sn, ds0, le, re, [1,2], nb = 100, rep = rep, mode=1)
		#denspro(sn, ds0, le, re, [2,0], nb = 100, rep = rep, mode=1)
		#denspro(sn, ds0, le, re, ax, nb = 100, rep = rep)
		#main(sn, nbh = 3, R = 3, alp = 0.1, ms = 0.3, rep=rep)
		
	#d = loaddata(sn, sn, rep)
	#ds0 = d['ds']

	#"""
	
	if mode==1:
		llab = ['BK17: FS1', 'BK17: FS2', 'HT16: IMF 1->100']	
		BK1 = np.array(retxt(cr+'BK_GWR1.txt',2,0,0))
		BK2 = np.array(retxt(cr+'BK_GWR2.txt',2,0,0))
		HT0 = np.array(retxt(cr+'HT_GWR.txt',2,0,0))
		lt0 = HT0[0]
		tmax = TZ(0)/1e9/YR
		sel = lt0<=tmax
		lt = lt0[sel]
		lz = [ZT(np.log10(t)) for t in lt]
		ly = HT0[1][sel]
		HT = np.array([lz, ly])
	
		LIGO_ = np.array(retxt(cr+'AdLIGOcurrent.txt',2,0,0))
		LIGO = np.array(retxt(cr+'AdLIGOdesign.txt',2,0,0))
		ET2 = np.array(retxt(cr+'ETxylophone.txt',2,0,0))
		deciHz = np.array(retxt(cr+'deciHz_op.txt',2,0,0))
		dhz = np.array([deciHz[0], deciHz[1]/deciHz[0]**0.5])
		LISA = np.array(retxt(cr+'LISA.txt',2,0,0))
		lisa = np.array([LISA[0], LISA[1]/LISA[0]**0.5])
		lsens = [LIGO, ET2, dhz, lisa]#, LIGO_]
		#lsens = [ET2, dhz]
		lins = ['AdLIGO', 'ETxylophone', 'DOoptimal','LISA']#, 'LIGO O2']
		#lins = ['ETxylophone', 'DOoptimal']
		if ligoo2==1:
			lsens.append(LIGO_)
			lins.append('LIGO O2')
		lsens0 = [LIGO, ET2, dhz, lisa, LIGO_]
	
		l = details(sn, zsn, repi, rep_, ncol, npro, abd=abd, h=h)# base='bhmergers_'+str(sn))
		Nb = len(l[0])
		print('Binary fraction: {:.3f} %'.format(Nb*586/MpopIII))
		#bhhis(l, sn, rep)
		ld = [mergerhis(l, ds0, rg = mode, gamma = g, dz=dz, mode=selab, mrel=rf, zsn=zsn, lsens=lsens,zlim=zlim,snrlim=snrlim, bfrac=bfrac, fage=fage,vcir=vcir, fej=fej, mfac=mfac) for g in lg]
		print('bfrac = {}'.format(bfrac))
		for d in ld:
			g = d['gamma']
			Nmg = d['Nmg'] #d['snr'][0]['Nmg']
			Nbbh = d['Nbbh']
			Nhh = d['Nhh']
			Nhl = d['Nhl']
			Nll = d['Nll']
			#le = d['snr'][0]['Dle']
			#lsnr = [dd['snr'] for dd in d['snr']]
			#[print('SNR ({}): '.format(lins[i]), sumy(lsnr[i])) for i in range(len(lins))]
			#print('Eccentricity: ', sumy(le))
			print('gamma = {}: Nmerger: {} ({} Mpc^-3), fraction: {}, efficiency: {} ({}) Msun^-1'.format(g, Nmg, Nmg/V, Nmg/Nbbh, Nll/MpopIII, Nmg/MpopIII))
			print('Nhh = {} ({:.3f}%), Nhl = {} ({:.3f}%), Nll = {} ({:.3f}%)'.format(Nhh, 1e2*Nhh/Nmg, Nhl, Nhl/Nmg*1e2, Nll, Nll/Nmg*1e2))
			i = 0
			for dsnr in d['snr']:
				rate, zmax = dsnr['Drate']
				frac = dsnr['frac']
				print('Detection rate (for z < {}): {:.3f} yr^-1, fraction: {:.3f}, for {}'.format(zmax, rate, frac, lins[i]))
				i += 1
		splothis(ld[0], rep_, 32, 0.7, selab=selab, lsens=lsens0, lins=lins, fage=fage, LIGO=LIGO_, lnsf=lnsf)
		#for ins in range(len(lins)):
		mplothis(ld, rep_, z1=zsn, lref=[BK1, BK2, HT], llab=llab, Mtot=Mextra, 
					lMs=lMs, lins=lins, dind = ins)
	
		if glab>0:
			#"""
			nr = 16
			lg = np.linspace(gamma0, gamma1, nr)
			ld = [mergerhis(l, ds0, rg = mode, gamma = g, dz=dz, mode=selab, mrel=rf, zsn=zsn, lsens=lsens, zlim=zlim,snrlim=snrlim,vcir=vcir, fej=fej, mfac=mfac) for g in lg]
			lrate0 = [d['snr'][0]['Drate'][0] for d in ld]
			lrate1 = [d['snr'][1]['Drate'][0] for d in ld]
			lrate2 = [d['snr'][2]['Drate'][0] for d in ld]
			lrate3 = [d['snr'][3]['Drate'][0] for d in ld]
			if ligoo2>0:
				lrate4 = [d['snr'][4]['Drate'][0] for d in ld]
			plt.figure()
			plt.plot(lg, lrate0, label='AdLIGO')
			plt.plot(lg, lrate1, '--', label='ETxylophone')
			plt.plot(lg, lrate2, '-.', label='DOoptimal')
			plt.plot(lg, lrate3, ':', label='LISA')
			if ligoo2>0:
				plt.plot(lg, lrate4, '-', label='LIGO O2', lw=3.0)
			plt.legend(loc=2)
			plt.xlabel(r'$\gamma$')
			plt.ylabel(r'$\dot{N}_{\mathrm{detection}}\ [\mathrm{yr^{-1}}]$')#)(z_{\mathrm{GW}}<'+str(zlim)+')\ [\mathrm{yr^{-1}}]$')
			plt.yscale('log')
			plt.tight_layout()
			plt.savefig(rep_+'rate_gamma_'+str(selab)+'.pdf')
			plt.close()
			#"""
			
			lrd = np.array([d['GWrate'][1][0]*(d['GWrate'][0][0]<dz) for d in ld])
			lrder = np.array([d['GWrate'][2][0]*(d['GWrate'][0][0]<dz) for d in ld])
			#lrd = np.array([d['GWrate'][1][0]*(d['GWrate'][0][0]<100) for d in ld])
			#lrder = np.array([d['GWrate'][2][0]*(d['GWrate'][0][0]<100) for d in ld])
			plt.figure()
			plt.errorbar(lg, lrd*1e9, yerr=lrder*1e9)
			plt.xlabel(r'$\gamma$')
			plt.ylabel(r'$\dot{n}_{\mathrm{GW}}(z_{\rm{GW}}\simeq 0)\ [\mathrm{yr^{-1}\ Gpc^{-3}}]$')
			plt.yscale('log')
			plt.tight_layout()
			plt.savefig(rep_+'rate_dens_gamma'+str(selab)+'.pdf')
			plt.close()
			
			if selab==1:
				lg = [1.5, 1.0, 0.5, 0]
				#nr = 20
				lbf = np.geomspace(0.2, 1.0, nr)
				md = [[mergerhis(l, ds0, rg = mode, gamma = g, dz=dz, mode=selab, mrel=rf, bfrac=bf, zsn=zsn, lsens=lsens,zlim=zlim,snrlim=snrlim,vcir=vcir, fej=fej, mfac=mfac) for bf in lbf] for g in lg]
				for ins in range(len(lins)):
					lrate = np.array([[d['snr'][ins]['Drate'][0] for d in ld] for ld in md])
			
					plt.figure()
					i = 0
					for r, g in zip(lrate, lg):
						plt.plot(lbf, r, ls=lls[i], label=r'$\gamma='+str(g)+'$')
						i += 1
					plt.xlabel(r'$f_{\rm{bulge}}$')
					plt.ylabel(r'$\dot{N}_{\mathrm{'+lins[ins]+r'}}\ [\mathrm{yr^{-1}}]$')
					#plt.ylabel(r'$\dot{N}_{\mathrm{GW}}(z_{\mathrm{GW}}<'+str(zlim)+')\ [\mathrm{yr^{-1}}]$')
					plt.yscale('log')
					#plt.xscale('log')
					plt.legend()
					plt.tight_layout()
					plt.savefig(rep_+'rate_bf_'+lins[ins]+'.pdf')
					plt.close()
			
				lrd = np.array([[d['GWrate'][1][0]*(d['GWrate'][0][0]<dz) for d in ld] for ld in md])
				plt.figure()
				i = 0
				for r, g in zip(lrd*1e9, lg):
					plt.plot(lbf, r, ls=lls[i], label=r'$\gamma='+str(g)+'$')
					i += 1
				plt.xlabel(r'$f_{\rm{bulge}}$')
				plt.ylabel(r'$\dot{n}_{\mathrm{GW}}(z_{\rm{GW}}\simeq 0)\ [\mathrm{yr^{-1}\ Gpc^{-3}}]$')
				#plt.xscale('log')
				plt.yscale('log')
				plt.legend()
				plt.tight_layout()
				plt.savefig(rep_+'rate_dens_bf.pdf')
				plt.close()
	#"""
	if mode==4:
		dt = 10
		#lR = [2, 1, 0.5, 0.25, 0.1, 0.05]
		lR = [0.03, 0.3, 1, 3, 10]
		Zmode=1
		if Zmode==0:
			dZ = metal_popIII(ds0, dt, lR, Zsun=Zsun)
			totxt(rep_+'lR.txt', [lR],0,0,0)
			totxt(rep_+'Zg1.txt', dZ['Zg'][0],0,0,0)
			totxt(rep_+'Zs1.txt', dZ['Zs'][0],0,0,0)
			totxt(rep_+'Zg2.txt', dZ['Zg'][1],0,0,0)
			totxt(rep_+'Zs2.txt', dZ['Zs'][1],0,0,0)
			totxt(rep_+'dg1.txt', dZ['dg'][0],0,0,0)
			totxt(rep_+'ds1.txt', dZ['ds'][0],0,0,0)
			totxt(rep_+'dg2.txt', dZ['dg'][1],0,0,0)
			totxt(rep_+'ds2.txt', dZ['ds'][1],0,0,0)
			totxt(rep+'Zself1.txt', [dZ['Z'][0]])
			totxt(rep+'Zself2.txt', [dZ['Z'][1]])
		else:
			lR = retxt(rep_+'lR.txt',1,0,0)[0]
			Zg1 = np.array(retxt(rep_+'Zg1.txt', len(lR),0,0))
			Zs1 = np.array(retxt(rep_+'Zs1.txt', len(lR),0,0))
			Zg2 = np.array(retxt(rep_+'Zg2.txt', len(lR),0,0))
			Zs2 = np.array(retxt(rep_+'Zs2.txt', len(lR),0,0))
			dg1 = np.array(retxt(rep_+'dg1.txt', len(lR),0,0))
			ds1 = np.array(retxt(rep_+'ds1.txt', len(lR),0,0))
			dg2 = np.array(retxt(rep_+'dg2.txt', len(lR),0,0))
			ds2 = np.array(retxt(rep_+'ds2.txt', len(lR),0,0))
			Z1 = np.array(retxt(rep_+'Zself1.txt', 1,0,0))
			Z2 = np.array(retxt(rep_+'Zself2.txt', 1,0,0))
			dZ = {}
			dZ['lr'] = lR
			dZ['Zg'] = [Zg1, Zg2]
			dZ['Zs'] = [Zs1, Zs2]
			dZ['dg'] = [dg1, dg2]
			dZ['ds'] = [ds1, ds2]
			dZ['Z'] = [Z1, Z2]
		popIII_Zdis(dZ, zsn, dt, rep_)
		envmode=Zmode
		#mass = BH_mass(rep_, sn, ds0, nbin=30, Rcm = 0.1, rho0=0.4, mode=envmode, beta=1, typef=abd)
	
	if mode==2:
		#nbh = countBH(ds0, abd=abd)
		#print('z = {:.2f}, N(SBH) = {}, N(DCBH) = {}'.format(*nbh))
		#bhind = BHmass_spec(sn, ds0, rep_, mode = 1, abd=abd)
		
		#print('z = {:.2f}, M_BH(seed) = {} Msun, M_BH = {} Msun'.format(*mass))
		#ds1 = phase(sn, cr, indm=0)
		#bhind = 0
		#ID = mass[3][bhind]
		#hsml = mass[4][bhind]
	#if mode==2:
		#plot3d(rep+'dis_'+str(sn)+'.png', ID, hsml, ds0)
	
		bh1 = np.array(retxt_nor(repi+'BH1_fr.txt', 4, 0, 0))
		sel = bh1[0]<=1/(1+zsn)
		Mseed = np.sum(bh1[1][sel] * bh1[2][sel])*fabd
		bh2 = np.array(retxt_nor(repi+'BH2_fr.txt', 4, 0, 0))
		sel = bh2[0]<=1/(1+zsn)
		Mseed += np.sum(bh2[1][sel] * bh2[2][sel])*fabd
		print('Total seed mass: {} Msun Mpc^-3'.format(Mseed/V))
	
		bhinfo = np.array(retxt_nor(repi+'blackholes.txt',7,0,0))
		lz = 1/bhinfo[0]-1
		
		zacc = 5
		sel = lz>=zacc
		m = bhinfo[2] * UM/Msun/h*fabd
		mdot = bhinfo[4]*fabd
		print('Total BH mass: {} Msun Mpc^-3'.format(m[sel][-1]/V))
		lt = np.array([TZ(z)/YR for z in lz])
		macc = np.sum(mdot[sel][:-1] * (lt[sel][1:]-lt[sel][:-1])) #np.trapz(mdot[sel], lt[sel])
		macc_ = m[sel][-1] - Mseed
		print('Total accreted mass: {} ({}) Msun Mpc^-3'.format(macc/V, macc_/V))
		print('Accreted ratio: {} ({})'.format(macc/m[sel][-1], macc_/m[sel][-1]))
		#print('NBH_seed/NBH: {}'.format(bhinfo[1][-1]/(nbh[1]+nbh[2])))
	
	
		ylm = [1, 1e5]
		plt.figure()
		plt.plot(lz[m>0], m[m>0]/V)
		plt.xlabel(r'$z$')
		plt.ylabel(r'$\rho_{\mathrm{BH}}\ [\mathrm{M_{\odot}\ Mpc^{-3}}]$')
		plt.yscale('log')
		plt.xlim(xlm)
		plt.ylim(ylm)
		plt.tight_layout()
		plt.savefig(rep+'rhoBH_z.pdf')
		plt.close()
		
		plt.figure()
		plt.plot(lz[mdot>0], mdot[mdot>0]/V)
		plt.xlabel(r'$z$')
		plt.ylabel(r'$\dot{\rho}_{\mathrm{acc}}\ [\mathrm{M_{\odot}\ yr^{-1}\ Mpc^{-3}}]$')
		plt.yscale('log')
		plt.xlim(xlm)
		plt.tight_layout()
		plt.savefig(rep_+'rhodotacc_z.pdf')
		plt.close()
		
		mbh = 120
		Eddrat = mdot[mdot>0]/np.array([max(int(mtt/mbh),1)*Macc_edd(min(mbh,mtt), 0.125) for mtt in m[mdot>0]])
		plt.figure()
		plt.plot(lz[mdot>0], Eddrat)
		plt.xlabel(r'$z$')
		plt.ylabel(r'$\langle\dot{M}_{\mathrm{acc}}/\dot{M}_{\mathrm{Edd}}\rangle\equiv'
		+r'\dot{\rho}_{\mathrm{acc}}/\dot{\rho}_{\mathrm{Edd}}$')
		plt.yscale('log')
		plt.xlim(xlm)
		plt.tight_layout()
		plt.savefig(rep_+'Eddrat_z.pdf')
		plt.close()
		print('Overall Eddington ratio: ', np.median(Eddrat))
		
		#mdot_ = (mdot[:-1] + mdot[1:])/2
		MJ14 = np.array(retxt(cr+'rhoacc_XrayB_MJ14.txt',2,0,0))
		ltmj = MJ14[0]/1e3
		lzmj = np.array([ZT(np.log10(t)) for t in ltmj])
		facc1, facc2, facc3 = 4.74e-6, 1.35e-5, 1.55e-4
		lmacc = np.cumsum(mdot[:-1] * (lt[1:]-lt[:-1]))
		lz = lz[1:]
		x1, x2 = np.max(lzmj), 4
		plt.figure()
		y1, y2 = 1e-4, 1e5
		plt.plot(lz[lmacc>0], lmacc[lmacc>0]/V, label='ISM, LB20', color='k')
		plt.plot(lzmj, MJ14[1], ls=(0, (10, 5)), label='XRB, MJ14')
		lm0 = np.cumsum(sfr0III[1]*sfr0III[3])
		plt.plot(1/sfr0III[0]-1, lm0*facc1, '--', label='XRB, FD')
		plt.plot(1/sfr0III[0]-1, lm0*facc2, '-.', label='XRB, FD_close')
		plt.plot(1/sfr0III[0]-1, lm0*facc3, ':', label='XRB, FD_greif')
		plt.fill_between([x1, x2], [0.66e4, 0.66e4], [1.4e4, 1.4e4], facecolor='gray', label='SR12: $z=5$', alpha=0.5)
		#plt.plot([5, 5], [y1, y2], 'k:')
		plt.xlabel(r'$z$')
		plt.ylabel(r'$\rho_{\mathrm{acc}}\ [\mathrm{M_{\odot}\ Mpc^{-3}}]$')
		plt.yscale('log')
		#plt.xlim(xlm)
		plt.xlim(x1, x2)
		plt.ylim(y1, y2)
		plt.legend(loc=1)
		plt.tight_layout()
		plt.savefig(rep_+'rhoacc_z.pdf')
		plt.close()
		
		rat = lmacc/m[1:]
		plt.figure()
		plt.plot(lz[rat>0], rat[rat>0])
		plt.plot(xlm, [1,1], 'k--')
		plt.xlabel(r'$z$')
		plt.ylabel(r'$\rho_{\mathrm{acc}}/\rho_{\mathrm{BH}}$')
		plt.yscale('log')
		plt.xlim(xlm)
		plt.tight_layout()
		plt.savefig(rep_+'ratio_acc_BH_z.pdf')
		plt.close()
	
	"""
		dz = 0.1
		#selab = 1
		ld = [stellarb(sn, rep, gamma=g, dz=dz, sk=sk, frm=frm, mode=selab) for g in [1.5, 1.0, 0.5]]
		#ld = [stellarb(sn, rep, gamma=gamma, dz=dz, sk=k, frm=frm) for k in [1e-3, 1e-2, 1e-1]]
		for d in ld:
			k = d['sk']
			g = d['gamma']
			rate, zlim = d['Drate']
			Nmg = d['Nmg']
			la0 = d['Dla0']
			print('a0: ', sumy(la0))
			print('Nmerger: {} ({} Mpc^-3)'.format(Nmg, Nmg/V))
			print('Detection rate (with gamma = {}, fsk = {}, for z < {}): {} yr^-1'.format(g, k, zlim, rate))
		splothis(ld[0], rep, 32, 0.5, mode=1)
		mplothis(ld, rep, z1=zsn, mode=1, m1 = 70, m2 = 300)
	
		
		nr = 20
		lg = np.linspace(0.5, 1.5, nr)
		lrate = [stellarb(sn, rep, gamma=g, dz=dz, sk=sk, frm=frm)['Drate'][0] for g in lg]
		plt.figure()
		if sk>0:
			plt.plot(lg, lrate, label=r'$f_{\mathrm{sk}}='+str(sk)+'$')
		else:
			plt.plot(lg, lrate)#, label=r'$f_{\mathrm{sk}}='+str(sk)+'$')
		plt.xlabel(r'$\gamma$')
		plt.ylabel(r'$\dot{N}_{\mathrm{GW}}(z_{\mathrm{GW}}<'+str(zlim)+')\ [\mathrm{yr^{-1}}]$')
		plt.legend()
		plt.tight_layout()
		plt.savefig(rep+'rate_gamma_col.pdf')
		plt.close()
		

	lg = [1.5, 1.0, 0.5]
	if mode==3:
		nr = 20
		x1, x2 = 0.001, 0.01
		lsk = np.geomspace(x1, x2, nr)
		out = [lsk]
		for g in lg:
			lrate = [stellarb(sn, rep, gamma=g, dz=dz, sk=k, frm=frm)['Drate'][0] for k in lsk]
			out.append(lrate)
		totxt(rep+'rate_fsk.txt', out, ['fsk']+[str(g) for g in lg], 1, 0)
	if mode==4:
		out = retxt(rep+'rate_fsk.txt', len(lg)+1, 1, 0)
		lsk = out[0]
		x1, x2 = np.min(lsk), np.max(lsk)
		#y1, y2 = 0.2, 1e3
		plt.figure()	
		for i in range(len(lg)):
			lrate = out[i+1]
			plt.plot(lsk, lrate, label=r'$\gamma='+str(lg[i])+'$', ls=lls[i])
			i += 1
		plt.xlabel(r'$f_{\mathrm{sk}}$')
		plt.ylabel(r'$\dot{N}_{\mathrm{GW}}(z_{\mathrm{GW}}<'+str(zlim)+')\ [\mathrm{yr^{-1}}]$')
		plt.legend()
		plt.yscale('log')
		plt.xscale('log')
		plt.xlim(x1, x2)
		#plt.ylim(y1, y2)
		plt.tight_layout()
		plt.savefig(rep+'rate_sk_col.pdf')
		plt.close()
	"""

	#print(fac, V*fac)
	
	
	"""
	sni, snf = 41, 21
	bht = bhtrack(sni, snf, rep, sp = 5)
	plt.figure()
	plt.plot(bht[0], bht[1], label=r'$M_{\mathrm{BH}}$', marker='.')
	#plt.plot(bht[0], bht[2], '--', label=r'$M_{\mathrm{acc}}$')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$M\ [\mathrm{M_{\odot}}]$')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'Maxhis.pdf')
	plt.close()
	"""
	
	"""
	sn1, sn2 = 73, 74
	R = 10
	ds1 = yt.load(rep+'snapshot_'+str(sn1).zfill(3)+'.hdf5')
	ds2 = yt.load(rep+'snapshot_'+str(sn2).zfill(3)+'.hdf5')
	pos = np.array(countBH(ds2)[4][bhind])
	plot3d(rep+'pre.pdf', pos, R, ds1)
	plot3d(rep+'post.pdf', pos, R, ds2)
	"""
	
	"""
	lnbh = []
	for n in range(40, 50):
		ds0 = yt.load(rep+'snapshot_'+str(n).zfill(3)+'.hdf5')
		nbh = countBH(ds0)
		lnbh.append(nbh)
	for x in lnbh:
		print('z = {:.2f}, N(SBH) = {}, N(DCBH) = {}'.format(*x))#N(DCBH,Z<Zth) = {}'.format(*x))
	"""
	
	"""
	ID = np.array(countBH(ds0)[4][bhind])
	for sn in range(72, 75):
		R = 3
		ds = yt.load(rep+'snapshot_'+str(sn).zfill(3)+'.hdf5')
		plot3d(rep+'dis_'+str(sn)+'.pdf', ID, R, ds)
		main(sn, nbh = 3, R = 3, alp = 0.1, ms = 0.3, rep=rep)
	"""
	
	"""
	bhind = 2
	mode = 0
	fac1 = 10.
	fac2 = 5.
	sns = 49
	snf = 47
	ds0 = yt.load(rep+'snapshot_'+str(sns).zfill(3)+'.hdf5')
	pos = np.array(countBH(ds0)[3][bhind])
	obj = recaesar(sns, rep)
	halo, rat = findhost(pos, obj, fac1, mode=mode)
	his = track(halo, sns, snf, fac2, rep=rep)
	prop = haloget(his)
	print('DCBH position: ',  pos)
	print('Host halo center: ', prop['pos'][0])
	print('r/r200: ', rat)
	print('Match fraction: ', his[2])
	print('Stellar masses: ', prop['mstr'])
	totxt(rep+'HostInfo.txt', [prop[k] for k in prop if k is not 'pos'], [k for k in prop if k is not 'pos'], 1, 0)
	
	plt.figure()
	plt.plot(his[1], prop['mtot'])
	plt.plot(his[1], Mup(his[1]), 'k--', label='Atomic')
	plt.legend()
	plt.yscale('log')
	plt.xlim(np.max(his[1]), np.min(his[1]))
	plt.xlabel(r'Redshift')
	plt.ylabel(r'$M_{h}\ [\mathrm{M_{\odot}}]$')
	plt.tight_layout()
	#plt.show()
	plt.savefig(rep+'AccHis_'+str(bhind)+'.pdf')
	plt.close()
	"""
	
	#"""
	if mode==3:
		lze = np.array([8, 7, 6, 5.5, 5, 4.5, 4])
		# 17, 18, 19, 20, 21, 22, 23
		z0 = 24
		ze = 4
		dt = 10e6
		ntot = 24
		ns = 10
		#nr = int((ntot-ns)/2)
		nr = ntot - ns - len(lze)
		z1 = 9
		a1 = 1/(1+z1)
		t1 = TZ(z1)/YR
		t2 = t1 + dt
		lt = np.linspace(t1, t2, ns+1)
		lam = np.array([1/(ZT(np.log10(t/1e9))+1) for t in lt])
		la0 = np.linspace(1/(1+z0), lam[0], nr+1)
		#lae = np.linspace(lam[-1], 1/(1+ze), nr+1)
		lae = 1/(1+lze)
		la = np.hstack([la0, lam[1:-1], lae])
		totxt('output_{}_{}_{}.txt'.format(z0, z1, ze), [la], 0, 0, 0)
	#"""
