import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cosmology import *
from txt import *
import matplotlib
#import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import sys
import os

plt.style.use('test2')
# I use a file called test2.mplstyle to control the plot style
# see https://matplotlib.org/stable/tutorials/introductory/customizing.html
# if you do not want to use this style, please comment out this line
# or replace it with your own configuration file

plt.style.use('tableau-colorblind10')
from radcool import cool #(T, ntot, n, J_21, z, gamma, X, T0):

# halo mass threshold Mdown (Mup) for molecular (atomic) cooling from Trenti & Stiavelli 2009
Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5

# mass threshold of Pop III star formation under external effects
# j21: LW intensity, vbc: strength of gas dark matter streaming motion
def mth_anna(j21, vbc=0.8):
	logM = 5.562*(1+0.279*j21**0.5)
	s = 0.614*(1-0.56*j21**0.5)
	logM += s*vbc
	return 10**logM

# typical LW background intensity produced by stars from Greif & Bromm 2006
j21_z = lambda z: 10**(2-z/5)

nf = 11 # num. of columns in a halo catalog produced by rockstar
# https://bitbucket.org/gfcstanford/rockstar/src/main/

# get the halo mass [Msun/h], virial radius [kpc/h] and positions [Mpc/h]
# from a rockstar catalog (all lengths are in comoving units)
def rockstar(hd):
	return np.array([hd[2], hd[4], *hd[8:11]*1e3])

H_FRAC = 0.75
# gas temperature [K] given the specific energy u and ionized fraction Y 
# mode=0: u is in cgs units, =1: u is in code units (defined in cosmology.py)
def temp(u, Y=0, mode=0, X = H_FRAC, gamma = 5.0/3):
	M = PROTON*4.0/(3*X+1+4*X*Y)
	if mode>0:
		U = u*UE/UM
	else:
		U = u
	tem = M*U*(gamma-1)/BOL
	return tem

# Eddington accretion rate in Msun/YR given the BH mass MBH [Msun], 
# radiative efficency eps and mean molecular weight mu
def Macc_edd(MBH, eps = 0.125, mu = 1.22):
	return 4*np.pi*GRA*PROTON*mu/SPEEDOFLIGHT/SIGMATH/eps * YR * MBH

# Jeans mass [Msun] given gas density n [cm^-3] and temperature T [K]
def MJeans(n, T, X=0.75):
	y = (375*BOL**3/(4*np.pi*(PROTON*mmw(X=X))**4*GRA**3))**0.5*(T**3/n)**0.5/Msun
	return y
	#return 2e3*(T/200)**1.5/(n/1e4)**0.5

# free-fall timescale [second] given the mass density [g/cm^-3]
def tff_func(rho):
	return (3*np.pi/(32*GRA*rho))**0.5

# IGM temperature, ds: input yt dataset of the simulation
def Tigm_func(ds):
	z = ds['Redshift']
	h = ds['HubbleParam']
	Om = ds['Omega0']
	ad = ds.all_data()
	lm0 = np.array(ad[('PartType0', 'Masses')].to('Msun'))
	lm1 = np.array(ad[('PartType1', 'Masses')].to('Msun'))
	lT = temp(ad[('PartType0','InternalEnergy')].to('cm**2/s**2'),ad[('PartType0','Primordial e-')])
	lh2 = np.array(ad[('PartType0', 'Primordial H2')])
	rhob = np.array(ad[('PartType0', 'Density')].to('g/cm**3'))
	rhoigm = rhom(1/(1+z), Om, h)*lm0[0]/lm1[0]
	sel = (rhob<rhoigm*10)*(rhob>0.1*rhoigm)
	Tmean = np.average(lT[sel], weights=1.0/rhob[sel])
	#xh2 = np.min(lh2) 
	sel = (rhob<rhoigm*1.1)*(rhob>0.9*rhoigm)
	xh2 = np.average(lh2[sel], weights=1.0/rhob[sel])
	Tref = T_b(z)
	return np.array([z, Tmean, Tref, xh2])

# temperature [K] - (hydrogen) density [cm^3] phase diagram 
# sn: snapshot index, ds: yt dataset, ad: simulation particle data 
# rep: output folder, nb: number of bins, rn/rT: density/temperature range
# edgen/edgeT: percentiles used to set the range if rn/rT is not given
def phasedg(sn, ds, ad, rep, nb, rn=[], rT=[], edgen=[0, 100], edgeT=[0, 100]):
	z = ds['Redshift']
	nhd = np.array(ad[('PartType0', 'Density')].to('g/cm**3'))*H_FRAC/PROTON
	lT = temp(ad[('PartType0','InternalEnergy')].to('cm**2/s**2'),ad[('PartType0','Primordial e-')])
	
	if len(rT)==0:
		rT = np.percentile(np.log10(lT), edgeT)
	if len(rn)==0:
		rn = np.percentile(np.log10(nhd), edgen)

	plt.figure()
	plt.subplot(111)
	plt.hist2d(np.log10(nhd),np.log10(lT),bins=nb,norm=LogNorm(),range=[rn,rT]) #np.log10([rn,rT]))
	plt.plot([],[],'.',markersize=1e-3, label=r'$z={:.2f}$'.format(z))
	plt.legend(fontsize=16, loc=4)
	plt.xlabel(r'$\log(n_{\mathrm{H}}\ [\mathrm{cm^{-3}}])$',size=16)
	plt.ylabel(r'$\log(T\ [\mathrm{K}])$',size=16)
	plt.tight_layout()
	plt.savefig(rep+'Tn_'+str(sn)+'.png', dpi=500)

# cosmic web in terms of projected distribution of particles
# le, re: positions of the left and right corners of the region to be plotted
# hd: halo catalog, must have been processed by the function rockstar
# ax: define the 2D plane of projection, 0: x, 1: y, 2: z
# norm: normalization of the sizes of haloes on the plot
# mcut: mass threshold above which BH particles (PartType=3) will be shown
# mhmin: mass threshold above which haloes are plotted
# mnorm: normalization of the sizes of BHs on the plot
# fic: show initial positions (1) or not (0)
# cmc: positions in comoving (1) or physical (0) coordinates
# note that this function returns the positions, radii and masses 
# of haloes above mhmin, and the index of the target halo among them
def cosweb(sn, ds, ad, le, re, hd, rep = './', nb = [1000, 1000], ax = [0, 1], norm = 1e6, mcut=10, mhmin = 1e5, mnorm=10, fic=0, cmc=0):
	bs = ds['BoxSize']
	z = ds['Redshift']
	h = ds['HubbleParam']
	posd = np.array(ad[('PartType1','Coordinates')].to('kpccm/h')).T
	posg = np.array(ad[('PartType0','Coordinates')].to('kpccm/h')).T
	if cmc>0:
		llb = [r'$x\ [\mathrm{kpc\ (comoving)}]$', r'$y\ [\mathrm{kpc\ (comoving)}]$', r'$z\ [\mathrm{kpc\ (comoving)}]$']
		tophy = 1.0/h
	else:
		llb = [r'$x\ [\mathrm{kpc}]$', r'$y\ [\mathrm{kpc}]$', r'$z\ [\mathrm{kpc}]$']
		tophy = 1.0/h/(1+z)
	if len(hd)>0:
		posh = hd[2:5]
		lm0 = hd[0]
		lsel = [(le[i]<posh[i])*(posh[i]<re[i]) for i in range(3)]
		sel = lsel[0]*lsel[1]*lsel[2]
		#print(np.max(lm0[sel])/mhmin)
		sel = sel*(lm0>mhmin)
	else:
		sel = [0]
		posh = []
		lr = []
	# select the target halo from all haloes above mhmin
	# or only those specified in halo_index.txt
	# the target halo is (by default) the halo with the denest gas particle
	if np.sum(sel)>0:
		if (os.path.exists(rep+'halo_index.txt')):
			hind = np.array(retxt(rep+'halo_index.txt',1)[0], dtype='int')
		else:
			hind = range(int(np.sum(sel)+0.1))
		lm = hd[0][sel][hind]/h
		lr = hd[1][sel][hind]
		posh = posh.T
		posh = posh[sel][hind]
		#maxmhind = np.argmax(lm)
		idens = np.argmax(ad['PartType0', 'Density'])
		cen = posg.T[idens]
		#print(cen/h)
		mmax = 0
		for i in range(len(lm)):
			if (np.sum((posh[i]-cen)**2)**0.5<lr[i]):
				if lm[i]>mmax:
					mmax = lm[i]
					maxmhind = i
		print('Target halo mass: {:.4e} Msun'.format( lm[maxmhind] ))
		posh = posh.T
		halo_label=r'$M_{}={:.1f}\times 10^5\ {}$, $R_{}={:.0f}\ {}$, $z={:.1f}$'.format(r'{\rm h}',lm[maxmhind]/1e5,r'\rm M_{\odot}', r'{\rm vir}',1e3*lr[maxmhind]/h/(1+z), r'\rm pc', z)
	else:
		hind = []
		lm = []# hd[0]/h
		lr = [] #hd[1]
		maxmhind = -1
		print('No massive halos')
	#print('Candidate halos: positions: ', posh.T, 'masses: ', lm)
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType3' for x in keys])
	if tag>0:
		poss = np.array(ad[('PartType3','Coordinates')].to('kpccm/h'))
		lmbh = np.array(ad[('PartType3', 'BH_Mass')])
		lnbh = np.array(ad[('PartType3', 'BH_NProgs')])
		dcbh = lmbh>mcut
		print('Number of BHs: ', np.sum(lnbh[dcbh]))
		#print('BH masses: ', lmbh)
		poss = poss[dcbh].T
	else:
		dcbh = [0]
	#print(np.sum(dcbh))
	
	plt.figure(figsize=(11,5.5))
	ax1 = plt.subplot(121)
	ax1.set_aspect('equal')
	plt.hist2d(posd[ax[0]]*tophy,posd[ax[1]]*tophy,bins=nb,norm=LogNorm(),cmap=plt.cm.Blues)
	#cb = plt.colorbar()
	#cb.set_label(r'$\log(N)$')
	if len(hind)>0:
		plt.scatter(posh[ax[0]]*tophy,posh[ax[1]]*tophy,s=lr**2/norm,edgecolors='r',facecolors='none',lw=0.5)
	if np.sum(dcbh)>0:
		plt.scatter(poss[ax[0]]*tophy,poss[ax[1]]*tophy, s=lmbh[dcbh]/mnorm, edgecolors='k', facecolors='k',lw=1)
		#print('DCBH!')
	plt.xlabel(llb[ax[0]],size=16)
	plt.ylabel(llb[ax[1]],size=16)
	plt.xlim(le[ax[0]]*tophy, re[ax[0]]*tophy)
	plt.ylim(le[ax[1]]*tophy, re[ax[1]]*tophy)
	plt.title(r'Dark matter',size=16)
#	plt.tight_layout()
	ax2 = plt.subplot(122)
	ax2.set_aspect('equal')
	plt.hist2d(posg[ax[0]]*tophy,posg[ax[1]]*tophy,bins=nb,norm=LogNorm(),cmap=plt.cm.Purples)
	#cb = plt.colorbar()
	#cb.set_label(r'$\log(N)$')
	
	# plot initial positions of particles in the haloes specified by hind
	# the initial positions are derived in another code and stored in partlist files
	if len(hind)>0:
		if fic>0:
			for i in range(len(lm)):
				if (os.path.exists(rep+'partlist_'+str(i)+'.txt')):
					post = np.array(retxt(rep+'partlist_'+str(i)+'.txt',3))*bs
					plt.scatter(*post[ax]*tophy, s=1, edgecolors='g',facecolors='g',lw=0,alpha=0.5)
		plt.scatter(posh[ax[0]]*tophy,posh[ax[1]]*tophy,s=lr**2/norm,edgecolors='r',facecolors='none',lw=0.5)
	
	if np.sum(dcbh)>0:
		plt.scatter(poss[ax[0]]*tophy,poss[ax[1]]*tophy, s=lmbh[dcbh]/mnorm, edgecolors='k', facecolors='k',lw=1)
	if len(hind)>0:
		#plt.plot([],[], '.', markersize=1e-3, label=halo_label)
		#plt.legend(fontsize=16)
		x, y = (le[ax] + 0.03*(re[ax]-le[ax]))*tophy
		plt.text(x, y, halo_label, size=16)
	plt.xlabel(llb[ax[0]],size=16)
	plt.ylabel(llb[ax[1]],size=16)
	plt.xlim(le[ax[0]]*tophy, re[ax[0]]*tophy)
	plt.ylim(le[ax[1]]*tophy, re[ax[1]]*tophy)
	plt.title(r'Gas',size=16)
	#plt.suptitle(r'Cosmic web at $z=$'+str(int(ds['Redshift']*100)/100),size=16)
	plt.tight_layout()
	plt.savefig(rep+'cswb_'+str(ax[0])+str(ax[1])+'_'+str(sn)+'.png', dpi=500)
	plt.close()
	return posh, lr, lm, maxmhind

# star formation history inferred from stellar and BH particles
# you can also get the SFH from popIII_sfr.txt and popII_sfr.txt
# mh: mass of the target halo or high-res region (for a zoom-in simulation) 
# or the total mass in the box
# dt: time resolution in Myr, edge and up: y axis range
# log: linear (0) or log (1) scale for x axis
# ntmax: maximum num. of time bins
# zcut: print total stellar masses formed above zcut
def SFhistory(sn, mh, ds, ad, rep='./', dt = 1, edge=0.5, log=1, ntmax=100, up=10, zcut=15):
	z = ds['Redshift']
	redlab = r'$z={:.1f}$'.format(z)
	mhlab = r'$M_{\rm total}='+'{:.2e}'.format(mh)+r'\ \rm M_{\odot}$'
	t0 = TZ(z)/YR/1e6
	m0 = np.array(ad[('PartType4', 'Masses')].to('Msun'))[0]
	sa = np.hstack([ad[('PartType4', 'StellarFormationTime')], ad[('PartType3', 'StellarFormationTime')]])
	stype = np.hstack([ad[('PartType4', 'Star Type')], ad[('PartType3', 'Star Type')]])
	sel3 = stype >= 30000
	sel2 = np.logical_not(sel3)
	t3 = np.array([TZ(x) for x in 1/sa[sel3]-1])/YR/1e6
	t2 = np.array([TZ(x) for x in 1/sa[sel2]-1])/YR/1e6
	m3tot = np.sum(sa[sel3]<1/(1+zcut))*m0
	m2tot = np.sum(sa[sel2]<1/(1+zcut))*m0
	print('Total mass of Pop III/II stars: {:.2e}/{:.2e} Msun, ratio={:.2e} for z>{}'.format(m3tot, m2tot, m3tot/m2tot, zcut))
	t = np.hstack([t3, t2])
	ti = np.min(t)
	nt0 = int(abs(t0-ti)/dt)+1
	nt = min(ntmax, nt0)
	#print('Num of timebins: ', nt)
	print('Delay time: {:.1f} Myr'.format(np.min(t2)-np.min(t3)))
	ti_ = t0-nt0*dt
	if log>0:
		ed = np.geomspace(ti_, t0, nt+1)
	else:
		ed = np.linspace(ti_, t0, nt+1)
	h2, ed = np.histogram(t2, ed)
	h3, ed = np.histogram(t3, ed)
	his, ed = np.histogram(t, ed)
	base = midbin(ed)
	norm = m0/(dt*1e6)
	y1, y2 = norm*edge, np.max(his*norm)/edge*up
	plt.figure()
	plt.plot(base, his*norm, 'k', label='Total\n'+redlab+', '+mhlab, zorder=0)
	plt.plot(base, h3*norm, '--', label='Pop III', zorder=2)
	plt.plot(base, h2*norm, '-.', label='Pop II', zorder=1)
	plt.xlabel(r'$t\ [\rm Myr]$')
	plt.ylabel(r'$\dot{M}_{\star}\ [\rm M_{\odot}\ yr^{-1}]$')
	plt.yscale('log')
	if log>0:
		plt.xscale('log')
	plt.xlim(ti_, t0)
	plt.ylim(y1, y2)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'SFR_t_'+str(sn)+'.pdf')
	plt.close()
	
# distribution of tidal tensor elements
# only relevant for simulations with TIDAL_TENSOR turned on in Config.sh
# ind: plot label, lind: indices of elements 
def tidaltensor_dis(sn, ind, ds, ad, nb=100, lind=[0,1,2,4,5,8], rep='./'):
	z = ds['Redshift']
	h = ds['HubbleParam']
	if ind==0:
		dtt = np.array(ad[('PartType0','TidalTensorPS')])
	else:
		dtt = np.array(ad[('PartType4','TidalTensorPS')])
		dtt1 = np.array(ad[('PartType3','TidalTensorPS')])
	tt = np.hstack(dtt.T[lind])*(1e5*h*(1+z)/KPC)**2*(1e9*YR)**2
	t1, t2 = np.quantile(tt, [0.01, 0.99])
	bins = np.linspace(t1, t2, nb+1)
	plt.figure()
	if ind==0:
		plt.hist(tt, bins, alpha=0.5, label='Gas')
	else:
		tt1 = np.hstack(dtt1.T[lind])*(1e5*h*(1+z)/KPC)**2*(1e9*YR)**2
		plt.hist(tt1, bins, alpha=0.5, label='BH')
		plt.hist(tt, bins, histtype='step', lw=1.5, label='Star')
	plt.xlabel(r'$T_{ij}\ [\rm Gyr^{-2}]$')
	plt.ylabel(r'Probability density [a. u.]')
	plt.legend()
	plt.xlim(t1, t2)
	plt.yscale('log')
	plt.title(r'$z={:.2f}$'.format(z))
	plt.tight_layout()
	plt.savefig(rep+'tt_dis_'+str(sn)+'_'+str(ind)+'.pdf')
	plt.close()

# velocity dispersion and gas density around BH particles
# esoft: size of the region around the BH particle in comoving kpc/h
def bhvel(ds, ad, esoft=0.028, gamma=5.0/3.0, X=0.75):
	pos3 = ad[('PartType3','Coordinates')]
	vel3 = np.array(ad[('PartType3', 'Velocities')].to('km/s'))
	nbh = len(pos3)
	y = np.zeros(nbh)
	z = np.zeros(nbh)
	for i in range(nbh):
		sp = ds.sphere(pos3[i], (esoft, 'kpccm/h'))
		ldens = np.array(sp['PartType0', 'Density'].to('g/cm**3'))*X/PROTON
		if len(ldens)>0:
			lm0 = np.array(sp[('PartType0', 'Masses')].to('Msun'))
			vel0 = np.array(sp[('PartType0', 'Velocities')].to('km/s'))
			cs2 = np.array(sp[('PartType0','InternalEnergy')].to('km**2/s**2')*(gamma-1)*gamma)
			lv0 = np.array([(np.sum((x-vel3[i])**2)) for x in vel0]) + cs2
			y[i] = (np.sum(lv0*lm0)/np.sum(lm0))**0.5
			z[i] = np.sum(lm0)/np.sum(lm0/ldens)
	#zsnap = ds['Redshift']
	#print(sp[('PartType0', 'Velocities')][0]/(1+zsnap)**0.5, sp[('PartType0', 'Velocities')][0].to('km/s'))
	return y, z

# plot various profiles around the densest gas particle with in sp
# sp: similar to ad, but is only used to identify the center
# (usually ad is the whole box, sp is the target halo)
# r: virial radius of the target halo in kpc/h (comoving)
# mh: mass of the target halo in Msun
# r1: minimum radius bin, rfac: defines the maximum radius bin with the virial radius
# nr: num. of radius bins
# mode: different ways of calculating gas density, =0 is faster but less accurate
# vfac: defines the range of y axis for velocity profiles
# xhd: abundance of deuterium 
# nosub: =1: ignore substructures that break the monotonicity of the density profile
# read=0: do the calculation and write results to files, 
# =1: read profile data from existing files 
# (usually you only need to run the function once with read=0 and always use 
# read=1 thereafter, unless you want to change the calculation itself)
# note that some ranges of plots are hard coded here for high-res simulations
def plot_profile(sn, ds, ad, sp, r, mh, rep, r1=1e-1, rfac=10, nr=50, mode=0, vfac=1.5, gamma=5.0/3.0, nth=1e5, X=0.75, xhd=2.38e-5, read=0, nosub=1):
	z = ds['Redshift']
	h = ds['HubbleParam']
	Om = ds['Omega0']
	halo_label=r'$M_{}={:.1f}\times 10^5\ {}$, $R_{}={:.0f}\ {}$, $z={:.1f}$'.format(r'{\rm h}',mh/1e5,r'\rm M_{\odot}', r'{\rm vir}',1e3*r/h/(1+z), r'\rm pc', z)
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType3' for x in keys])
	lm0 = np.array(ad[('PartType0', 'Masses')].to('Msun'))
	lm1 = np.array(ad[('PartType1', 'Masses')].to('Msun'))
	r2 = r*rfac/h*1e3/(1+z)
	rvir = r/h*1e3/(1+z)
	if read==0:
		#cen = np.array(np.average(ad[('PartType0','Coordinates')].to('pc'), 0, weights=ad[('PartType0','Density')]**2))
		lT = np.array(temp(ad[('PartType0','InternalEnergy')].to('cm**2/s**2'),ad[('PartType0','Primordial e-')]))
		pos0 = np.array(ad[('PartType0','Coordinates')].to('pc'))
		pos1 = np.array(ad[('PartType1','Coordinates')].to('pc'))
		idens = np.argmax(sp['PartType0', 'Density'])
		posg = np.array(sp[('PartType0','Coordinates')].to('pc'))
		cen = posg[idens]
		lr0 = np.array([(np.sum((x-cen)**2))**0.5 for x in pos0])
		#r1 = max(np.min(lr0[lr0>0])*0.99, r1)
		lr1 = np.array([(np.sum((x-cen)**2))**0.5 for x in pos1])
		vel0 = np.array(ad[('PartType0', 'Velocities')].to('km/s'))
		cs2 = np.array(ad[('PartType0','InternalEnergy')].to('km**2/s**2')*(gamma-1)*gamma)
		vel1 = np.array(ad[('PartType1', 'Velocities')].to('km/s'))
		lh2 = np.array(ad[('PartType0', 'Primordial H2')])
		lhd = np.array(ad[('PartType0', 'Primordial HD')])
		le = np.array(ad[('PartType0', 'Primordial e-')])
		mgas = np.sum(lm0[lr0<r2])
		mdm = np.sum(lm1[lr1<r2])
		print('Gas fraction: {:.2f}'.format(mgas/mdm*lm1[0]/lm0[0]))
		vcen = (np.average(vel0, 0)*mgas + np.average(vel1, 0)*mdm)/(mgas+mdm)
		lv0 = np.array([(np.sum((x-vcen)**2)) for x in vel0])
		lv0_ = np.array([(np.sum((x-vcen)**2)) for x in vel0]) + cs2
		lv1 = np.array([(np.sum((x-vcen)**2)) for x in vel1])
		#print(r2)
		red = np.array([0] + [x for x in np.geomspace(r1, r2, nr+1)])
		rbase = (red[:-1] + red[1:])*0.5
		if tag>0:
			pos3 = np.array(ad[('PartType3','Coordinates')].to('pc'))
			vel3 = np.array(ad[('PartType3', 'Velocities')].to('km/s'))
			lv3 = np.array([(np.sum((x-vcen)**2)) for x in vel3])
			lr3 = np.array([(np.sum((x-cen)**2))**0.5 for x in pos3])
			lm3 = np.array(ad[('PartType3', 'Masses')].to('Msun'))
			lnbh = np.array(ad[('PartType3', 'BH_NProgs')])
			print('Num. of PBHs in the halo: {} ({})'.format(np.sum(lr3<rvir), np.sum(lnbh[lr3<rvir])))
		
		vshell = (4*np.pi/3*(red[1:]**3-red[:-1]**3))
		his0, ed = np.histogram(lr0, red, weights=lm0)
		hish2, ed = np.histogram(lr0, red, weights=lm0*lh2)
		hishd, ed = np.histogram(lr0, red, weights=lm0*lhd)
		hise, ed = np.histogram(lr0, red, weights=lm0*le)
		if mode>0:
			ldens = np.array(ad['PartType0', 'Density'].to('Msun/pc**3'))
			hisd, ed = np.histogram(lr0, red, weights=lm0/ldens)
			gasdens = np.zeros(nr+1)
			gasdens[his0>0] = his0[his0>0]/hisd[his0>0]
		else:
			gasdens = his0/vshell
		densmax = nth/X*PROTON/Msun*PC**3
		hisv0, ed = np.histogram(lr0, red, weights=lm0*lv0)
		hisv0_, ed = np.histogram(lr0, red, weights=lm0*lv0_)
		hisT, ed = np.histogram(lr0, red, weights=lm0*lT)
		gash2, gashd, gase = np.zeros(nr+1), np.zeros(nr+1), np.zeros(nr+1)
		gasvel = np.zeros(nr+1)
		gasvel_ = np.zeros(nr+1)
		sel0 = his0>0
		gasvel[sel0] = (hisv0[sel0]/his0[sel0])**0.5
		gasvel_[sel0] = (hisv0_[sel0]/his0[sel0])**0.5
		gasT = np.zeros(nr+1)
		gasT[sel0] = hisT[sel0]/his0[sel0]
		gash2[sel0] = hish2[sel0]/his0[sel0]
		gashd[sel0] = hishd[sel0]/his0[sel0]
		gase[sel0] = hise[sel0]/his0[sel0]
		print('Central gas temperature: {:.1f} K'.format(gasT[sel0][0]))
		his1, ed = np.histogram(lr1, red, weights=lm1)
		hisv1, ed = np.histogram(lr1, red, weights=lm1*lv1)
		dmvel = np.zeros(nr+1)
		sel1 = his1>0
		dmvel[sel1] = (hisv1[sel1]/his1[sel1])**0.5
		#dmvel = (hisv1/his1)**0.5
		dmdens = his1/vshell
		out = [rbase, vshell, gasdens, gasvel, gasvel_, gasT, gash2, gashd, gase]
		out += [dmdens, dmvel]
		if tag>0 and len(lm3)>0:
			his3, ed = np.histogram(lr3, red, weights=lm3)
			sel = lr3<r1
			if np.sum(sel)>0:
				#his3[0] += np.sum(lm3[sel])
				print('Num. of PBH at the center {:.2e} pc: {} ({})'.format(r1, np.sum(sel), np.sum(lnbh[sel])))
			mpbh = np.sum(his3)
			pbhdens = his3/vshell
			hisv3, ed = np.histogram(lr3, red, weights=lm3*lv3)
			pbhvel = np.zeros(nr+1)
			sel3 = his3>0
			pbhvel[sel3] = (hisv3[sel3]/his3[sel3])**0.5
			#pbhvel = (hisv3/his3)
			sigma_pbh = (np.sum(pbhvel**2*his3)/np.sum(his3))**0.5
			print('PBH velocity dispersion: {:.2f} km s^-1'.format(sigma_pbh))
			#norm3 = int(mgas/mpbh)
			out += [pbhdens, pbhvel]
		totxt(rep+'prodata_'+str(sn)+'.txt', out)
	else:
		if tag>0:
			rbase, vshell, gasdens, gasvel, gasvel_, gasT, gash2, gashd, gase, dmdens, dmvel, pbhdens, pbhvel = np.array(retxt(rep+'prodata_'+str(sn)+'.txt',13))
		else:
			rbase, vshell, gasdens, gasvel, gasvel_, gasT, gash2, gashd, gase, dmdens, dmvel = np.array(retxt(rep+'prodata_'+str(sn)+'.txt',11))
		nr = len(rbase)-1
	
	vmax = np.max(dmvel)
	norm1 = 1 #int(np.sum(gasdens*vshell)/np.sum(dmdens*vshell))
	xh = 0.93
	dpro = gasdens + dmdens #his0+his1
	#rho = (gasdens+dmdens)*Msun/PC**3
	if tag>0:# and np.sum(pbhdens)>0:
		vmax = max(vmax, np.max(pbhvel))
		dpro += pbhdens #his3
		norm3 = int(np.sum(gasdens*vshell)/np.sum(pbhdens*vshell))
		#rho += (pbhdens)*Msun/PC**3
	menc = np.cumsum(dpro*vshell)
	rho = dpro*Msun/PC**3
	gasn = gasdens*Msun/PC**3/PROTON/mmw(X=X)
	rhofloor = rhom(1/(1+z), Om, h)
	nfloor = rhofloor*lm0[0]/lm1[0]/PROTON/mmw(X=X)
	gasn[gasn<nfloor] = nfloor 
	rho[rho<rhofloor] = rhofloor
	lmj = MJeans(gasn, gasT)
	ltff = tff_func(rho)/YR/1e6
	ny = np.zeros((nr+1, 17))
	ny[:,0] = (1.0 - gase - 2.0*gash2)*xh
	ny[:,1] = gase*xh
	ny[:,5] = gase*xh
	ny[:,3] = gash2*xh
	ny[:,6] = 1.0-xh
	ny[:,9] = xhd*xh
	ny[:,11] = gashd * xhd*xh
	dTdt = -np.array([cool(gasT[i], gasn[i], ny[i]*gasn[i], 0, z, gamma, X, 2.73) for i in range(nr+1)])
	ltcool = gasT/dTdt/YR/1e6
	
	lflag = np.ones(nr+1)
	refn = gasn[0]
	for i in range(nr):
		if gasn[i+1]>refn:
			lflag[i+1] = 0
		else:
			refn = gasn[i+1]
	if nosub:
		sel = lflag>0.5
	else:
		sel = lflag>-1
	
	if gasT[0]>200:
		n1, n2, nn = 1e-2, 1e5, 8
	else:
		n1, n2, nn = 1e-2, 1e4, 7
	plt.figure(figsize=(5.5,7))
	ax1 = plt.subplot(211)
	plt.loglog(gasn[sel]*xh, gasT[sel], 'k-', label='$T$')#, label=halo_label)
	y1 = 1e2 #min(1e2, np.min(gasT))
	y2 = max(np.max(gasT)*1.1,2e3)
	plt.legend(loc=2)
	#plt.xlabel(r'$n_{\rm H}\ [\rm cm^{-3}]$')
	plt.ylabel(r'$T\ [\rm K]$')
	plt.ylim(y1, y2)
	y1, y2, ny = 1e-7, 1e-1, 7
	ax2 = ax1.twinx()
	ax2.loglog(gasn[sel]*xh, gash2[sel], '--', label=r'$n_{\rm H_{2}}/n_{\rm H}$')#r'$\rm H_{2}$')
	ax2.loglog(gasn[sel]*xh, gashd[sel], '-.', label=r'$n_{\rm HD}/n_{\rm D}$')#r'$\rm HD$')
	ax2.loglog(gasn[sel]*xh, gase[sel], ':', label=r'$n_{\rm e^{-}}/n_{\rm H}$')#r'$\rm e^{-}$')
	ax2.set_ylabel('Abundance') #(r'$x_{k}\equiv n_{k}/n_{\rm H}$')
	ax2.set_ylim(y1, y2)
	ax2.set_yticks(np.geomspace(y1, y2, ny))
	plt.legend(loc=3)
	plt.xlim(n1, n2)
	plt.xticks(np.geomspace(n1, n2, nn))
	plt.title(halo_label, size=16)
	y1, y2 = 1e-2, 10
	ax1 = plt.subplot(212)
	plt.plot([n1, n2], [1]*2, 'k-', lw=0.5)
	plt.loglog(gasn[sel]*xh, menc[sel]/lmj[sel], 'k-', label=r'$M_{\rm enc}/M_{\rm J}$')
	plt.ylabel(r'$M_{\rm enc}/M_{\rm J}$')
	plt.ylim(y1, y2)
	plt.legend(loc=2)
	plt.xlabel(r'$n_{\rm H}\ [\rm cm^{-3}]$')
	y1, y2, ny = 1e-4, 1e2, 7
	ax2 = ax1.twinx()
	ax2.loglog(gasn[sel]*xh, 1/ltcool[sel], '--', label=r'$1/t_{\rm cool}$')
	ax2.loglog(gasn[sel]*xh, (gamma-1)/ltff[sel], '-.', label=r'$(\gamma-1)/t_{\rm ff}$')
	plt.legend(loc=4)
	ax2.set_ylabel(r'$d\ln T/dt\ [\rm Myr^{-1}]$')
	ax2.set_ylim(y1, y2)
	ax2.set_yticks(np.geomspace(y1, y2, ny))
	plt.xlim(n1, n2)
	plt.xticks(np.geomspace(n1, n2, nn))
	"""
	ax1 = plt.subplot(313)
	#plt.plot([n1, n2], [1]*2, 'k:')
	plt.loglog(gasn*xh, 1/ltcool, label=r'$1/t_{\rm cool}$')
	plt.loglog(gasn*xh, (gamma-1)/ltff, '--', label=r'$(\gamma-1)/t_{\rm ff}$')
	#plt.loglog(gasn*xh, gamma/ltff, '-.', label=r'$\gamma/t_{\rm ff}$')
	plt.legend(loc=4)
	plt.xlabel(r'$n_{\rm H}\ [\rm cm^{-3}]$')
	plt.ylabel(r'$d\ln T/dt\ [\rm Myr^{-1}]$')
	plt.xlim(n1, n2)
	plt.xticks(np.geomspace(n1, n2, nn))
	plt.ylim(y1, y2)
	plt.yticks(np.geomspace(y1, y2, ny))
	"""
	plt.tight_layout()
	plt.savefig(rep+'coolpro_'+str(sn)+'.pdf')
	plt.close()
	
	#plt.figure()
	plt.figure(figsize=(5.5,7))
	ax1 = plt.subplot(211)
	#ax1.set_aspect('equal')
	y1, y2, ny = 1e-3, 1e4, 8
	plt.loglog(rbase, gasdens, label='Gas')
	plt.loglog(rbase, dmdens*norm1, '--', label=r'DM')# $\times {:.2e}$'.format(norm1))
	if tag>0:# and np.sum(pbhdens)>0:
		plt.loglog(rbase, pbhdens*norm3, '-.', label=r'PBH $\times {}$'.format(norm3))
	#plt.loglog([r1,r2],[densmax]*2,'k-',lw=0.5)
	plt.legend(loc=1)
	#plt.xlabel(r'$r\ [\rm pc]$')
	plt.ylabel(r'$\rho\ [\rm M_{\odot}\ pc^{-3}]$')
	plt.xlim(r1*0.45, r2)
	plt.ylim(y1, y2)
	plt.yticks(np.geomspace(y1, y2, ny))
	plt.title(halo_label, size=16)
	ax1 = plt.subplot(212)
	#ax1.set_aspect('equal')
	plt.plot(rbase, gasvel, label='Gas')
	plt.plot(rbase, dmvel*norm1, '--', label=r'DM')# $\times {:.2e}$'.format(norm1))
	if tag>0:# and np.sum(pbhdens)>0:
		plt.plot(rbase, pbhvel, '-.', label=r'PBH')# $\times {:.2e}$'.format(norm3))
	plt.plot(rbase, gasvel_, ':', label='Gas ($\sqrt{\sigma^{2}+c_{s}^{2}}$)')
	plt.legend(ncol=2, loc=1)
	plt.xscale('log')
	plt.xlabel(r'$r\ [\rm pc]$')
	plt.ylabel(r'$\sigma\ [\rm km\ s^{-1}]$')
	plt.xlim(r1*0.45, r2)
	plt.ylim(0, vmax*vfac)
	plt.tight_layout()
	plt.savefig(rep+'denspro_'+str(sn)+'.pdf')
	plt.close()
	
	"""
	y1, y2, ny = 1e-8, 1e-1, 8
	plt.figure()
	plt.loglog(rbase, gash2, label=r'$n_{\rm H_{2}}/n_{\rm H}$')
	plt.loglog(rbase, gashd, '--', label=r'$n_{\rm HD}/n_{\rm D}$')
	plt.loglog(rbase, gase, '-.', label=r'$n_{\rm e^{-}}/n_{\rm H}$')
	plt.legend(loc=4)
	plt.xlabel(r'$r\ [\rm pc]$')
	plt.ylabel(r'Abundance (relative)')
	plt.xlim(r1, r2)
	plt.ylim(y1, y2)
	plt.yticks(np.geomspace(y1, y2, ny))
	plt.tight_layout()
	plt.savefig(rep+'chemipro_'+str(sn)+'.pdf')
	plt.close()
	"""
	#plt.tight_layout()
	#plt.savefig(rep+'temppro_'+str(sn)+'.pdf')
	#plt.close()
	"""
	plt.loglog(rbase, np.cumsum(gasdens*vshell), label='Gas')
	if mode>0:
		plt.loglog(rbase, np.cumsum(gasdens1*vshell), ':', label='Gas (smoothed)')
	plt.loglog(rbase, np.cumsum(dmdens*vshell)*norm1, '--', label=r'DM')# $\times {:.2e}$'.format(norm1))
	if tag>0 and len(lm3)>0:
		plt.loglog(rbase, np.cumsum(pbhdens*vshell)*norm3, '-.', label=r'PBH')# $\times {:.2e}$'.format(norm3))
	plt.legend()
	plt.xlabel(r'$r\ [\rm pc]$')
	plt.ylabel(r'$M_{\rm enc}\ [\rm M_{\odot}]$')
	plt.xlim(r1, r2)
	"""

if __name__=="__main__":
	test = 1
	sn =  0
	plotcswb = 1 # plot cosmic web or not
	plotpro = 1 # plot profiles or not
	fic = 0
	fmet=0 # plot metallicity distribution or not (using yt)
	fac = 1 # set the halo mass threshold to be fac * Mdown
	fzoom = 0 # zoom-in simulation or not
	check = True # check for low-res particle contamination, only for fzoom=1
	ytplot = 1 # make additional plots with yt
	cmc = 1 # comoving units
	read = 0 # read profile data
	printfield = 0 # print data fields of the yt dataset
	fcen = 1 # set center to the mass center (=0) or the location of the densest
				# gas particle (=1)
	bhstat = 1 # calculate gas properties around BHs (see bhvel)
	nosub = 0 # see plot_profile
	mulax = 0 # =1: make cosmic web plots along three directions of projection
				# =0: only do the xy plane
	if len(sys.argv)>1:
		sn = int(sys.argv[1]) # the first argument is the snapshot index
	if len(sys.argv)>2:
		read = int(sys.argv[2]) 
	
	pid = 0

	if test==0:
		#rep = '../N256L014/'
		#rep = '../N128L014/'
		#rep = '../N128L014_nu2/'
		#rep = '../ref_runs/N128L014_zoom_pbh_sfdbk/' #25
		#rep = '../N128L014_zoom_cdm/' #14
		#rep = '../N128L014_zoom_cdm_cmb/' #16
		#rep = '../N128L014_zoom_cdm_streaming/' #18
		rep = '../N128L014_zoom_pbh_dmo/' #18 8 13 25
		#rep = '../N128L014_zoom_pbh_cmb/' # 9
		#rep = '../N128L014_zoom_pbh_all/' # 6
		#rep = '../N128L014_zoom_pbh_short/' #14
		#rep = '../N128L014_zoom_pbh_m100/' #6
		#rep = '../N128L014_zoom_pbh_uptb/' #17
		#rep = '../N128L014_zoom_pbh_nfdbk/' #5
		#rep = '../N128L014_zoom_pbh_wfdbk/' #7
		#rep = '../N128L014_zoom_pbh_sfdbk/' #9
		#rep = '../N128L014_zoom_pbh_streaming/' #10
		#rep = '../N128L014_nu2_zoom_cdm/' #25
		#rep = '../N128L014_nu2_zoom_cdm_cmb/' #27
		#rep = '../N128L014_nu2_zoom_pbh1/' # 23 23 33 30
		#rep = '../N128L014_nu2_zoom_pbh_cmb/' # 28
		#rep = '../N128L014_nu2_zoom_pbh_all/' # 14
		#rep = '../N128L014_nu2_zoom_pbh_short/' # 14
		#rep = '../N128L014_nu2_zoom_pbh_m100/' #17
		#rep = '../N128L014_nu2_zoom_pbh_uptb/' #27
		#rep = '../N128L014_nu2_zoom_pbh_nfdbk/' #15
		L = 140*KPC
	else:
		#rep = '../N32L025_nu3_z300/'
		#rep = '../N32L025_nu3_dens/'
		#rep = '../testparam/'
		#rep = '../N32L025_nu3_pbh_nfdbk/'
		#rep = '../N32L025_nu3_pbh_sfdbk/'
		#rep = '../N32L025_nu3_zoom_cdm/'
		#rep = '../N32L025_nu3_zoom_pbh/'
		#plotcswb = 0
		#fic = 0
		rep = '../N32L025_nu3_cdm/'
		#rep = '../N128L1_nu2/'
		L = 250*KPC
		#L = 1e3*KPC
		cmc = 1
		zcut = 4
		#pid=822248
		#pid=805608

	fname = rep + 'snapshot_'+str(sn).zfill(3)+'.hdf5'
	if plotcswb>0:
		hc = 'halos_0.0_'+str(sn)+'.ascii'
		# put the raw halo catalog from rockstar in the same folder of simulation data
		print(hc)
		if (os.path.exists(rep+hc)):
			halocata = rockstar(np.array(retxt(rep+hc,nf,19,0)))
		else:
			halocata = []

	ds = yt.load(fname) # create yt dataset
	
	z = ds['Redshift']
	t = TZ(z)/YR/1e6
	print('z = {:.2f}, t = {:.2f} Myr'.format(z, t))
	h = ds['HubbleParam']
	
	ad0 = ds.all_data() # get all particles in the box

	print('Particle mass: ', ad0[('PartType0','Masses')][0].to('Msun'), ad0[('PartType1','Masses')][0].to('Msun'))
	
	ligm = Tigm_func(ds)
	print('IGM properties:', ligm)
	if printfield>0:
		for key in ds:
			print(key)
		l = [x for x in ds.field_list] # if x[0]=='PartType0']
		print(l)
	
	"""
	# show the properties of the particle pid
	sel = ad0[('PartType0','ParticleIDs')]==pid
	if np.sum(sel)>0:
		ldens = ad0[('PartType0','Density')].to('g/cm**3')/1.22/PROTON
		lT = np.array(temp(ad0[('PartType0','InternalEnergy')].to('cm**2/s**2'),ad0[('PartType0','Primordial e-')]))
		print('n={:.2e} cm^-3, T={:.1f} K'.format( ldens[sel][0], lT[sel][0]))
	else:
		print('Could not find particle {}'.format(pid))
	"""
	
	mnorm = 10
	bs = ds['BoxSize']
	if test>0:
	#	ext, thk, asp = 250, 250, 1
	#	cen = [125, 125, 125]
		xb, yb = 200, 200
		mcut = 0
		#mcut = 1000
		mnorm = 333
	else:
		xb, yb = 1000, 1000
		mcut = 0
	if fzoom==0:
		ext, thk, asp = bs, bs, 1
		cen = [bs/2]*3
	else:
		ext, thk, asp = 10, 10, 1
		if fcen==0:
			cen = np.array(np.average(ad0[('PartType0','Coordinates')], 0, weights=ad0[('PartType0','Density')]**2))
		else:
			idens = np.argmax(ad0['PartType0', 'Density'])
			pos0 = np.array(ad0[('PartType0','Coordinates')])
			cen = pos0[idens]
		xb, yb = 500, 500
		#cen = 75.5595365, 33.8665082, 65.4746631
		

	cen_ = np.array(cen)
	lext = np.array([ext, ext/asp, thk])
	le, re = cen_ - lext/2, cen_ + lext/2

	ad = ds.box(le, re) # get particles in the box defined by le and re
	
	rn = [-5, 5]
	rT = [1, 6]
	#if sn>0:
	phasedg(sn, ds, ad, rep, int(0.5*(xb+yb)))#, rn=rn, rT=rT)

	norm2 = 3 # haloes are shown with sizes norm2 * virial radius on the plots 
	if fzoom>0:
		norm2 = 1
	norm1 = 6.8e-2*(ext/1e3)**2 * (3/norm2)**2 # this is chosen for the specific test2 style
	if mulax>0:
		lax = [[0,1], [1,2], [2,0]]
	else:
		lax = [[0, 1]]
	if plotcswb>0:
		mhmin = Mdown(z)*h*fac
		print('Halo mass threshold: {:.2e} Msun'.format(mhmin/h/fac))
		for ax in lax:
			posh, lr, lm, hind = cosweb(sn, ds, ad, le, re, halocata, rep, [xb, yb], ax=ax, mhmin=mhmin, norm=norm1, mcut=mcut, fic=fic, cmc=cmc, mnorm=mnorm)
		#print(posh.T/h)
		#print(lm)
		#print(lr)
		#plot_profile(sn, ds, ad, lr[0], rep)
		#lm0 = np.array(ad0[('PartType0', 'Masses')].to('Msun'))
		#lm1 = np.array(ad0[('PartType1', 'Masses')].to('Msun'))
		#print(np.sum(lm0)/np.sum(lm1), lm0[0]/lm1[0])
		if len(lr)>0:
			sp = ds.sphere(posh.T[hind], (lr[hind], 'kpccm/h'))
			idens = np.argmax(sp['PartType0', 'Density'])
			pos0 = np.array(sp[('PartType0','Coordinates')])
			cen = pos0[idens]
		if plotpro>0 and len(lr)>0:
			plot_profile(sn, ds, ad0, sp, lr[hind], lm[hind], rep, read=read, nosub=nosub)
	
	#exit()
	
	# you can also make plots with functions from yt 
	if ytplot:
		#"""
		ax = [0,1]
		prj = yt.ParticleProjectionPlot(ds, int(3-np.sum(ax)+0.5), ('PartType1', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
		prj.set_unit(('PartType1', 'Masses'), 'Msun/(kpccm/h)**2')
		prj.set_buff_size((xb, yb))
		prj.annotate_title('$z={:.2f}$'.format(z))
		if plotcswb>0 and len(lr)>0:
			for i in range(len(lr)):
				x, y = posh[ax[0]][i]-cen[ax[0]], posh[ax[1]][i]-cen[ax[1]]
				prj.annotate_sphere([x, y], radius=(lr[i]*norm2, 'kpccm/h'), coord_system="plot",circle_args={"color": "red", "linewidth": 1})
		prj.save(rep+'dmdis_'+str(sn)+'.png')

		ax = [0,1]
		prj = yt.ParticleProjectionPlot(ds, int(3-np.sum(ax)+0.5), ('PartType0', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
		prj.set_unit(('PartType0', 'Masses'), 'Msun/(kpccm/h)**2')
		prj.set_buff_size((xb, yb))
		prj.annotate_title('$z={:.2f}$'.format(z))
		if plotcswb>0 and len(lr)>0:
			for i in range(len(lr)):
				x, y = posh[ax[0]][i]-cen[ax[0]], posh[ax[1]][i]-cen[ax[1]]
				prj.annotate_sphere([x, y], radius=(lr[i]*norm2, 'kpccm/h'), coord_system="plot",circle_args={"color": "red", "linewidth": 1})
		prj.save(rep+'gasdis_'+str(sn)+'.png')
		#"""

		#"""
		plot = yt.ParticlePhasePlot(ad, ('PartType0', 'Density'), ('PartType0', 'InternalEnergy'), [('PartType0', 'Masses')], weight_field=None, x_bins=xb, y_bins=yb)
		plot.set_unit(('PartType0', 'Density'), 'g/cm**3')
		plot.set_unit(('PartType0', 'InternalEnergy'), 'km**2/s**2')
		plot.set_unit(('PartType0', 'Masses'), 'Msun')
		plot.set_log(('PartType0', 'Density'), True)
		#plot.set_zlim('all', 9e3, 1e9)
		#plot.annotate_text(xpos=0, ypos=0, text='O')
		#ax = plot.plots[('PartType0', 'Masses')].axes
		#ax.set_ylim(9e3, 1e9)
		plot.annotate_title('$z={:.2f}$'.format(z))
		plot.save(rep+'phasedg_'+str(sn)+'.png')
		#"""
	if fzoom>0:
		print('Halo center: ', cen/h)
		
	# check if there are BH particles (PartType3)
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType3' for x in keys]) 
	if tag>0:
		if len(lr)==0:
			sp = ad
		#if ytplot:
		#	extbh, thkbh = ext/asp, thk
			#extbh, thkbh = bs, bs
		#	prj = yt.ParticleProjectionPlot(ds, 2, ('PartType3', 'Masses'), center=cen, width=((extbh, 'kpccm/h'), (extbh, 'kpccm/h')), depth = thkbh, density=True)#, data_source=ad)
		#	prj.set_unit(('PartType3', 'Masses'), 'Msun/(kpccm/h)**2')
		#	prj.set_buff_size((xb, yb))
		#	prj.annotate_title('$z={:.2f}$'.format(z))
		#	prj.save(rep+'pbhdis_'+str(sn)+'.png')
				
		lmbh = np.array(ad0[('PartType3', 'Masses')].to('Msun'))
		lmbh0 = np.array(ad0[('PartType3', 'BH Mass')])
		mmax = np.max(lmbh)
		mmin = np.min(lmbh0)
		nmg = np.sum(np.array(lmbh[lmbh>2*mmin]/mmin-1.0, dtype='int'))
		print('Maximum (minimum) PBH mass: {} ({}) Msun'.format(mmax, mmin))
		print('Num. of mergers: {}'.format(nmg))
		if (os.path.exists(rep+'blackholes.txt')):
			bhinfo = np.array(retxt(rep+'blackholes.txt',7,0,0))
			lz = 1/bhinfo[0]-1
			#V = (bs/h/1e3)**3
			zacc = 5
			sel = lz>=zacc
			m = bhinfo[2] * 1e10/h
			mdot = bhinfo[4]
			print('Total BH mass: {} Msun'.format(m[sel][-1]))
			lt = np.array([TZ(z)/YR for z in lz])
			macc = m[sel][-1] - m[sel][0] #np.sum(mdot[sel][:-1] * (lt[sel][1:]-lt[sel][:-1])) #np.trapz(mdot[sel], lt[sel])
			macc_ = np.sum(lmbh) - np.sum(lmbh0) - nmg*mmin #len(ad[('PartType3','Masses')]) * mpbh
			print('Total accreted mass: {} ({})'.format(macc, macc_))
			meddacc = Macc_edd(m[0], eps = 0.059, mu = 1.22) * (lt[sel][-1]-lt[sel][0])
			print('Edd ratio: {} ({})'.format(macc/meddacc, macc_/meddacc))
			print('Accreted ratio: {} ({})'.format(macc/m[sel][-1], macc_/m[sel][-1]))
		if bhstat:
			lvt, lnH = bhvel(ds, sp)
			print('BH velocity: {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(np.min(lvt), np.max(lvt), np.mean(lvt), np.median(lvt)))
			print('BH gas density: {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(np.min(lnH), np.max(lnH), np.mean(lnH), np.median(lnH)))

	dind = np.argmax(sp[('PartType0', 'Density')].to('g/cm**3'))
	print('Maximum density: {:.2e} cm^-3'.format(np.array(H_FRAC*sp[('PartType0', 'Density')][dind].to('g/cm**3'))/PROTON))
	T = np.array(temp(sp[('PartType0', 'InternalEnergy')][dind].to('cm**2/s**2'),sp[('PartType0','Primordial e-')][dind]))
	print('Temperature of dense gas: {:.1f} K'.format(T))

	# check contamination of low-res particles (PartType5)
	if fzoom>0 and test==0 and check:			
		Nlowres = len(sp['PartType5','Masses'])
		if Nlowres>0:
			ax = [0,1]
			prj = yt.ParticleProjectionPlot(ds, int(3-np.sum(ax)+0.5), ('PartType5', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
			prj.set_unit(('PartType5', 'Masses'), 'Msun/(kpccm/h)**2')
			prj.set_buff_size((xb, yb))
			prj.annotate_title('$z={:.2f}$'.format(z))
			if plotcswb>0 and len(lr)>0:
				for i in range(len(lr)):
					x, y = posh[ax[0]][i]-cen[ax[0]], posh[ax[1]][i]-cen[ax[1]]
					prj.annotate_sphere([x, y], radius=(lr[i]*norm2, 'kpccm/h'), coord_system="plot",circle_args={"color": "red", "linewidth": 1})
			prj.save(rep+'lowresdis_'+str(sn)+'.png')
			mlow = np.array(np.max(ad['PartType5','Masses'].to('Msun')))
			print('Maximum mass of low-res particles: {:.2e} Msun'.format(mlow))
		print('Num. of low-res particles: {}'.format(Nlowres))
	
	# plot metallicity distribution with yt
	if fmet>0:
		ax = [0, 1]
		prj = yt.ParticleProjectionPlot(ds, 2, ('PartType0', 'Metallicity_00'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, weight_field=('PartType0', 'Masses'))#, data_source=ad)
		prj.set_zlim('all', 1e-6*0.02, 0.2)
		prj.set_buff_size((xb, yb))
		prj.annotate_title('$z={:.2f}$'.format(z))
		if plotcswb>0 and len(lr)>0:
			for i in range(len(lr)):
				x, y = posh[ax[0]][i]-cen[ax[0]], posh[ax[1]][i]-cen[ax[1]]
				prj.annotate_sphere([x, y], radius=(lr[i]*norm2, 'kpccm/h'), coord_system="plot",circle_args={"color": "red", "linewidth": 1})
		prj.save(rep+'metaldis_'+str(sn)+'.png')
	
	# check if there are stellar particles (PartType4)
	keys = ds.field_list
	tag = np.sum([x[0] == 'PartType4' for x in keys]) 
	if tag==0:
		print('No active stars!')
		exit() # stop here if there are no stars

	# plot stellar mass distribution
	prj = yt.ParticleProjectionPlot(ds, 2, ('PartType4', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
	prj.set_unit(('PartType4', 'Masses'), 'Msun/(kpccm/h)**2')
	prj.set_buff_size((xb, yb))
	prj.annotate_title('$z={:.2f}$'.format(z))
	prj.save(rep+'stardis_'+str(sn)+'.png')
	#prj.save('snapshot_'+str(sn).zfill(3)+'_stellar.png')

	ad = ds.all_data()
	if fzoom>0:
		mtot = np.sum(ad0['PartType0','Masses'].to('Msun'))+np.sum(ad0['PartType1','Masses'].to('Msun'))
	else:
		mtot = rhom(1) * (L/h)**3/Msun
		#mtot0 = np.array(np.sum(ad0['PartType0','Masses'].to('Msun'))+np.sum(ad0['PartType1','Masses'].to('Msun')))
		#print('check mass conservation:', mtot/mtot0)
	SFhistory(sn, mtot, ds, ad, dt=10, up = 20, rep=rep, zcut=zcut)
	#tidaltensor_dis(sn, 0, ds, ad, rep=rep)
	#tidaltensor_dis(sn, 1, ds, ad, rep=rep)
	
"""
sc = yt.create_scene(ds, lens_type='perspective', field=('PartType0', 'Density'))
#im, sc = yt.volume_render(ds, ('PartType0', 'Density'), fname='rendering.png')

sc.camera.width = (1000, 'kpccm/h')

sc.camera.switch_orientation()

source = sc[0]

source.tfh.set_log(True)

source.tfh.grey_opacity = False

source.tfh.plot(rep+'transfer_function.png', profile_field=('PartType0', 'Density'))

sc.save('rendering.png')#, sigma_clip=6.0)
"""
