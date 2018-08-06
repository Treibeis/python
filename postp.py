from txt import *
import yt
import caesar
from matplotlib.colors import LogNorm
from yt import YTQuantity
import matplotlib.patches as pltp
from matplotlib.collections import PatchCollection
from scipy.optimize import *
from sklearn.neighbors import NearestNeighbors
from process import *
import time
import sys

lls = ['-', '--', '-.', ':']
lls = lls*2

Hfrac = 0.76
XeH = 0.0
XeHe = 0.0

UL = 3.085678e21
#UL = 1.0
UM = 1.989e43
#UM = 1.0
UV = 1.0e5
#UV = 1.0

UT = UL/UV
UD = UM/UL**3
UP = UM/UL/UT**2
UE = UM*UL**2/UT**2

G = GRA*UM*UT**2/UL**3

np_rotation = lambda x: np.array(x).T

def temp(u, Y, X = H_FRAC, gamma = 5.0/3):
	M = PROTON*4.0/(3*X+1+4*X*Y)
	U = u*UE/UM
	tem = M*U*(gamma-1)/BOL
	return tem

def mmw(xeH = XeH, xeHe = XeHe, X = Hfrac):
	xh = 4*X/(1+3*X)
	return 4.0/(1.0+3*X)/(xh*(1+xeH)+(1-xh)*(1+xeHe))

# dd["density"].to_equivalent("cm**-3", "number_density",mu=mmw())

def cosmicT(z = 99, zd = 200.0, zre = 1100):
	Tre = 2.73*(1+zre)
	if z>=zd:
		T = Tre*(1+z)/(1+zre)
	else:
		Td = Tre*(1+zd)/(1+zre)
		T = Td*(1+z)**2/(1+zd)**2
	return T

def bg_density(z):
	Omegab = 0.048
	h = 0.6774
	rhoc = (h*100/1e3/UT)**2 * 3/8/np.pi/GRA
	return rhoc*Omegab*(1+z)**3

def gas_data(sn, density_th = 50, temperature_th = 500.0, rep = './', base_dir = './', base = 'snapshot', ext = '.hdf5'):
	data = {}
	#filename = rep+base_dir+'_'+str(sn).zfill(3)+'/'+base+'_'+str(sn).zfill(3)+'.'+str(0)+ext
	filename = rep+base_dir+base+'_'+str(sn).zfill(3)+ext
	ds = yt.load(filename)
	ad = ds.all_data()
	z = ds['Redshift']
	dens = (ad[('PartType0','Density')].to('g/cm**3') > density_th*bg_density(z)) * (temp(ad[('PartType0','InternalEnergy')],ad[('PartType0','Primordial HII')]) < temperature_th)
	coordinates = ad[('PartType0','Coordinates')][dens].to('kpccm/h')
	velocities = ad[('PartType0','Velocities')][dens].to('km/s')
	temperatures = temp(ad[('PartType0','InternalEnergy')][dens],ad[('PartType0','Primordial HII')][dens])
	densities = ad[('PartType0','Density')][dens].to('g/cm**3')
	num_densities = ad[('PartType0','Density')][dens].to_equivalent("cm**-3", "number_density",mu=mmw(0.0))
	H2_abundances = ad[('PartType0','Primordial H2')][dens]
	HD_abundances = ad[('PartType0','Primordial HD')][dens]
	data['pos'] = coordinates
	data['vel'] = velocities
	data['T'] = temperatures
	data['rho'] = densities
	data['n'] = num_densities
	data['H2'] = H2_abundances
	data['HD'] = HD_abundances
	data['ds'] = ds
	return data

def density_peak(data, nn = 64):
	X = np.array(data['pos'])
	rho = np.array(data['rho'])
	nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)
	peak_index = [rho[i]==max(rho[indices[i]]) for i in range(X.shape[0])]		
	d = {}
	d['pos'] = X[peak_index]
	d['index'] = peak_index
	return d

r_smooth = lambda sp: float(np.max(((3*sp[('PartType0','Masses')].to('g')/sp[('PartType0','Density')].to('g/cm**3')/4/np.pi)**(1/3)).to('kpc')))
cloud_mass = lambda sp: np.sum(sp[('PartType0','Masses')].to('Msun')) + np.sum(sp[('PartType3','Masses')].to('Msun'))
cloud_temp = lambda sp: np.average(temp(sp[('PartType0','InternalEnergy')],sp[('PartType0','Primordial HII')]))
cloud_mu = lambda sp: 1.0/np.average(1.0/mmw(sp[('PartType0','Primordial HII')]))
def cloud_rho(posc, sp, z, h = 0.6774):
	rad = np.max((np.array(sp[('PartType0','Coordinates')].to('kpccm/h')) - posc)**2)**0.5
	if len(sp[('PartType3','Masses')])>0:
		rad = max(rad, np.max((np.array(sp[('PartType0','Coordinates')].to('kpccm/h')) - posc)**2)**0.5)
	r = (r_smooth(sp)+rad/(1+z)/h)*UL
	mass = float(cloud_mass(sp).to('g'))
	return  mass/(4*np.pi*r**3/3.0)
def cloud_Jeans(posc, sp, z, h = 0.6774):
	T = cloud_temp(sp)
	mu = cloud_mu(sp)
	rho = cloud_rho(posc, sp, z, h)
	l_Jeans = float((15*BOL*T/4/np.pi/GRA/rho/mu/PROTON)**0.5)
	return 4*np.pi*l_Jeans**3*rho/3.0

def JM(T = 2e2, n = 1e4, mu = 1.22):
	rho = n*mu*PROTON
	l_Jeans = float((15*BOL*T/4/np.pi/GRA/rho/mu/PROTON)**0.5)
	return 4*np.pi*l_Jeans**3*rho/3.0 * 1e10/UM

def clouds(data, peaks, nn = 2, min_N = 32, rmax = 3.0, h = 0.6774, wall = 600, scale = 30, hsml = 0.02875):
	#xh = 4*X/(1+3*X)
	ds = data['ds']
	z = ds['Redshift']
	redshift = int(z*100)/100
	ad = ds.all_data()
	sink_num = len(ad[('PartType3','Masses')])
	gas_mass = ad[('PartType0','Masses')][0]
	#peaks = density_peak(data)
	print('number of density peaks: {}'.format(len(peaks['pos'])))
	radius = np.array(((3*gas_mass/data['rho'][peaks['index']]/4/np.pi)**(1/3)).to('kpccm/h'))
	cloud_index = np.zeros(radius.shape[0])
	cloud1 = []
	cloud2  =[]
	Mc_MJ1 = []
	Mc_MJ2 = []
	r_ratio1 = []
	r_ratio2 = []
	#ifsink1 = []
	#ifsink2 = []
	#sinkID = [0]
	nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(peaks['pos'])
	distances, indices = nbrs.kneighbors(peaks['pos'])		
	
	sink_pos = np.array(ad[('PartType3','Coordinates')].to('kpccm/h'))
	sink_rad = np.array([0.2 for i in range(sink_num)])
	sink_index = np.zeros(sink_num)
	nbrs = NearestNeighbors(n_neighbors=nn*2, algorithm='ball_tree').fit(sink_pos)
	distances0, indices0 = nbrs.kneighbors(sink_pos)
	gas_rad = []
	lsr = []
	lssp = []
	
	for i in range(sink_num):
		flag = np.max(distances0[i][1:] < sink_rad[indices0[i]][1:])
		if flag==0:
			posc = np.array(sink_pos[i])
			lpos = np.vstack([peaks['pos'],[posc]])
			nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(lpos)
			distances1, indices1 = nbrs.kneighbors(lpos)
			low = distances1[len(peaks['pos'])][1]
			sp = ds.sphere(posc, (2*low, 'kpccm/h'))
			rad = np.min(np.array(((3*gas_mass/sp[('PartType0','Density')]/4/np.pi)**(1/3)).to('kpccm/h')))
			if low>rmax:
				sink_ratio = ad[('PartType3','Masses')][i]/gas_mass
				print('sink {}, bounds [kpccm/h]: {}, num_P: {}, compact'.format(i, (rad, low, distances0[i][1]), sink_ratio))
				sink_index[i] = 2
				gas_rad.append(rad)
				lsr.append(sink_ratio)
				lssp.append(sp)
				continue
			def ratio1(rad):
				sp = ds.sphere(posc, (rad, 'kpccm/h'))
				mass = float(cloud_mass(sp).to('g'))
				mass_Jeans = cloud_Jeans(posc, sp, z, h)
				return -mass/mass_Jeans
			if distances0[i][1]<rmax:
				up = max(rad*scale, distances0[i][1])
			else:
				up = min(rad*scale, distances0[i][1])
			bd = (low*1.001, max(up,2.0*low))
			res = minimize_scalar(ratio1, bounds=bd, method = 'bounded')
			sp = ds.sphere(posc, (res.x, 'kpccm/h'))
			num_part = sp[('PartType0','Masses')].shape[0]
			if res.x>0.9*max(up,2.0*low):
				print('Unreasonable bound: {}'.format(bd))
				bd = (low, max(up,2.0*low)*2.0)
				res = minimize_scalar(ratio1, bounds=bd, method = 'bounded')
				sp = ds.sphere(posc, (res.x, 'kpccm/h'))
			#sp0 = ds.sphere(peaks['pos'][i], (up/2.0, 'kpccm/h'))
			#num_part = sp[('PartType0','Masses')].shape[0]
			#num_part0 = sp0[('PartType0','Masses')].shape[0]
				if res.x>0.9*max(up,2.0*low)*2.0:
					print('Failed to correct the bound: {}'.format(bd))
					sink_ratio = ad[('PartType3','Masses')][i]/gas_mass
					print('sink {}, bounds [kpccm/h]: {}, num_P: {}, compact'.format(i, (rad, low, distances0[i][1]), sink_ratio))
					sink_index[i] = 2
					gas_rad.append(rad)
					lsr.append(sink_ratio)
					lssp.append(sp)
					continue
			if num_part < min_N:# or (-res.fun<1 and res.x>= min(distances[i][1],rmax)):
				print('sink {}, radius : {} [kpccm/h], bounds [kpccm/h]: {}, num_p: {}, Mcloud/MJ: {}, discarded'.format(i, res.x, (rad, low, distances0[i][1]), num_part, -res.fun))
				continue
			sink_index[i] = 1
			sink_rad[i] = res.x
			cloud1.append(sp)
			r_ratio1.append(res.x/rad)
			Mc_MJ1.append(-res.fun)
			#ifsink.append(1)
			print('sink {}, radius : {} [kpccm/h], bounds [kpccm/h]: {}, num_p: {}, Mcloud/MJ: {}'.format(i, res.x, (rad, low, distances0[i][1]), num_part, -res.fun))
	cloud_pos = np.vstack([peaks['pos'][cloud_index > 0],sink_pos[sink_index==1]])
	cloud_rad = np.hstack([radius[cloud_index > 0],sink_rad[sink_index==1]])
	compact_rad = (hsml+np.array(gas_rad))/h/(1+z)# + np.array([r_smooth(sp) for sp in lssp])
	compact_tem = np.array([cloud_temp(sp) for sp in lssp])
	cloud = np.hstack([cloud2, cloud1])
	d = {}
	d['pos'] = np.vstack([cloud_pos,sink_pos[sink_index==2]])
	d['radcm'] = np.hstack([cloud_rad,[hsml for i in range(len(gas_rad))]])
	d['cloud'] = cloud
	d['ratio'] = np.hstack([Mc_MJ2,Mc_MJ1,lsr])
	d['Mc'] = np.hstack([np.array([cloud_mass(sp) for sp in cloud]),ad[('PartType3','Masses')][sink_index==2].to('Msun')])
	d['rad'] = np.hstack([np.array([r_smooth(sp) for sp in cloud]) + cloud_rad/h/(1+z),compact_rad])
	d['rho'] = d['Mc']*UM/1e10/(4*np.pi*(d['rad']*UL)**3/3.0)
	d['T'] = np.hstack([np.array([cloud_temp(sp) for sp in cloud]),compact_tem])
	d['r_ratio'] = np.hstack([r_ratio2,r_ratio1,hsml/np.array(gas_rad)])
	d['ifsink'] = np.hstack([[1 for i in range(len(cloud_pos))],[2 for i in range(len(gas_rad))]])
	totxt('sinks_z'+str(redshift)+'.txt',np.vstack([ np_rotation(d['pos']), [d['radcm'], d['rad'], d['ratio'], d['Mc'], d['rho'], d['T'], d['r_ratio'], d['ifsink']] ]), ['x [kpccm/h]', 'y', 'z', 'radcm [kpc/h]', 'rad [kpc]', 'ratio', 'Mc [Msun]', 'rho [g cm**-3]', 'T [K]', 'r_ratio', 'ifsink'],1,0)
	
	start = time.time()
	count = 1
	sink_pos_clouds = sink_pos[sink_index==1]
	sink_num_clouds = len(sink_pos_clouds)
	for i in range(radius.shape[0]):
		flag = np.max(distances[i][1:] < radius[indices[i][1:]])
		posc = np.array(peaks['pos'][i])
		lpos = np.vstack([sink_pos_clouds,[posc]])
		nbrs = NearestNeighbors(n_neighbors=nn*2, algorithm='ball_tree').fit(lpos)
		distances2, indices2 = nbrs.kneighbors(lpos)
		flag_ = np.max(distances2[sink_num_clouds][1:] < sink_rad[indices2[sink_num_clouds][1:]])
		if flag>0 or flag_>0:
			#cloud_index[i] = 0.0
			continue
		if radius[i]*scale>rmax:
			print('cloud {}, bounds [kpccm/h]: {}, not dense enough'.format(i, (radius[i], distances[i][1])))
			continue
		posc = peaks['pos'][i]
		def ratio(rad):
			sp = ds.sphere(posc, (rad, 'kpccm/h'))
			mass = float(cloud_mass(sp).to('g'))
			mass_Jeans = cloud_Jeans(posc, sp, z, h)
			return -mass/mass_Jeans
		if distances[i][1]<rmax:
			up = max(radius[i]*scale, distances[i][1])
		else:
			up = min(radius[i]*scale, distances[i][1])
		bd = (min(radius[i],hsml), up)#max(distances[i][1],rmax))
		res = minimize_scalar(ratio, bounds=bd, method = 'bounded')
		sp = ds.sphere(peaks['pos'][i], (res.x, 'kpccm/h'))
		num_part = sp[('PartType0','Masses')].shape[0]
		#if res.x<rmax:
		#	sp0 = ds.sphere(peaks['pos'][i], (up, 'kpccm/h'))
		#	num_part0 = sp0[('PartType0','Masses')].shape[0]
		if res.x>0.9*up:
			print('Unreasonable bound: {}'.format(bd))
			bd = (min(radius[i],hsml), up/2.0)
			res = minimize_scalar(ratio, bounds=bd, method = 'bounded')
			sp = ds.sphere(peaks['pos'][i], (res.x, 'kpccm/h'))
			#sp0 = ds.sphere(peaks['pos'][i], (up/2.0, 'kpccm/h'))
			#num_part = sp[('PartType0','Masses')].shape[0]
			#num_part0 = sp0[('PartType0','Masses')].shape[0]
			if res.x>0.9*up/2.0:
				print('Failed to correct the bound: {}'.format(bd))
				num_part = -1
		if res.x>=rmax or num_part < min_N:# or (-res.fun<1 and res.x>= min(distances[i][1],rmax)):
			print('cloud {}, radius : {} [kpccm/h], bounds [kpccm/h]: {}, num_p: {}, Mcloud/MJ: {}, discarded'.format(i, res.x, (radius[i], distances[i][1]), num_part, -res.fun))
			continue
		sink = int(len(sp[('PartType3','Masses')])>0)
		if sink==1:
			print('cloud {}, with sink'.format(i))
			continue			
		cloud_index[i] = 1.0
		print('cloud {}, radius : {} [kpccm/h], bounds [kpccm/h]: {}, num_p: {}, Mcloud/MJ: {}, ifsink: {}'.format(i, res.x, (radius[i], distances[i][1]), num_part, -res.fun, sink))
		cloud2.append(sp)
		rad = res.x#np.max((np.array(sp[('PartType0','Coordinates')].to('kpccm/h')) - posc)**2)**0.5
		r_ratio2.append(rad/radius[i])
		radius[i] = rad
		Mc_MJ2.append(-res.fun)
		#ifsink.append(sink)
		#if sink==1:
		#	for x in sp[('PartType3','ParticleIDs')]:
		#		sinkID.append(x)
		now = time.time()
		if int((now-start)/wall) >= count:
			cloud_pos = np.vstack([peaks['pos'][cloud_index > 0],sink_pos[sink_index==1]])
			cloud_rad = np.hstack([radius[cloud_index > 0],sink_rad[sink_index==1]])
			compact_rad = (hsml+np.array(gas_rad))/h/(1+z)# + np.array([r_smooth(sp) for sp in lssp])
			compact_tem = np.array([cloud_temp(sp) for sp in lssp])
			cloud = np.hstack([cloud2, cloud1])
			d = {}
			d['pos'] = np.vstack([cloud_pos,sink_pos[sink_index==2]])
			d['radcm'] = np.hstack([cloud_rad,[hsml for i in range(len(gas_rad))]])
			d['cloud'] = cloud
			d['ratio'] = np.hstack([Mc_MJ2,Mc_MJ1,lsr])
			d['Mc'] = np.hstack([np.array([cloud_mass(sp) for sp in cloud]),ad[('PartType3','Masses')][sink_index==2].to('Msun')])
			d['rad'] = np.hstack([np.array([r_smooth(sp) for sp in cloud]) + cloud_rad/h/(1+z),compact_rad])
			d['rho'] = d['Mc']*UM/1e10/(4*np.pi*(d['rad']*UL)**3/3.0)
			d['T'] = np.hstack([np.array([cloud_temp(sp) for sp in cloud]),compact_tem])
			d['r_ratio'] = np.hstack([r_ratio2,r_ratio1,hsml/np.array(gas_rad)])
			d['ifsink'] = np.hstack([[0 for i in range(len(radius[cloud_index > 0]))],[1 for i in range(len(sink_pos[sink_index==1]))],[2 for i in range(len(gas_rad))]])
			totxt('clouds_z'+str(redshift)+'_'+str(count)+'.txt',np.vstack([np_rotation(d['pos']), [d['radcm'], d['rad'], d['ratio'], d['Mc'], d['rho'], d['T'], d['r_ratio'], d['ifsink']]]), ['x [kpccm/h]', 'y', 'z', 'radcm [kpc/h]', 'rad [kpc]', 'ratio', 'Mc [Msun]', 'rho [g cm**-3]', 'T [K]', 'r_ratio', 'ifsink'],1,0)
			count += 1
	
	cloud_pos = np.vstack([peaks['pos'][cloud_index > 0],sink_pos[sink_index==1]])
	cloud_rad = np.hstack([radius[cloud_index > 0],sink_rad[sink_index==1]])
	compact_rad = (hsml+np.array(gas_rad))/h/(1+z)# + np.array([r_smooth(sp) for sp in lssp])
	compact_tem = np.array([cloud_temp(sp) for sp in lssp])
	cloud = np.hstack([cloud2, cloud1])
	d = {}
	d['pos'] = np.vstack([cloud_pos,sink_pos[sink_index==2]])
	d['radcm'] = np.hstack([cloud_rad,[hsml for i in range(len(gas_rad))]])
	d['cloud'] = cloud
	d['ratio'] = np.hstack([Mc_MJ2,Mc_MJ1,lsr])
	d['Mc'] = np.hstack([np.array([cloud_mass(sp) for sp in cloud]),ad[('PartType3','Masses')][sink_index==2].to('Msun')])
	d['rad'] = np.hstack([np.array([r_smooth(sp) for sp in cloud]) + cloud_rad/h/(1+z),compact_rad])
	d['rho'] = d['Mc']*UM/1e10/(4*np.pi*(d['rad']*UL)**3/3.0)
	d['T'] = np.hstack([np.array([cloud_temp(sp) for sp in cloud]),compact_tem])
	d['r_ratio'] = np.hstack([r_ratio2,r_ratio1,hsml/np.array(gas_rad)])
	d['ifsink'] = np.hstack([[0 for i in range(len(radius[cloud_index > 0]))],[1 for i in range(len(sink_pos[sink_index==1]))],[2 for i in range(len(gas_rad))]])
	totxt('clouds_z'+str(redshift)+'.txt',np.vstack([ np_rotation(d['pos']), [d['radcm'], d['rad'], d['ratio'], d['Mc'], d['rho'], d['T'], d['r_ratio'], d['ifsink']] ]), ['x [kpccm/h]', 'y', 'z', 'radcm [kpc/h]', 'rad [kpc]', 'ratio', 'Mc [Msun]', 'rho [g cm**-3]', 'T [K]', 'r_ratio', 'ifsink'],1,0)
	return d

def read_clouds(ds, mmin = 3e5, rep = './', h = 0.6774, X = 0.76):
	z = ds['Redshift']
	redshift = int(z*100)/100
	fn = rep+'clouds_z'+str(redshift)+'.txt'
	cloud_data = np.array(retxt(fn,11,1,0))
	d = {}
	massive = cloud_data[6] > mmin
	d['pos'] = np_rotation(cloud_data[:3])[massive]
	d['radcm'] = cloud_data[3][massive]
	#cloud = [ds.sphere(d['pos'][i], (d['radcm'][i], 'kpccm/h')) for i in range(len(d['pos']))]
	#d['cloud'] = cloud
	d['ifsink'] = cloud_data[10][massive]#np.array([int(len(sp[('PartType3','Masses')])>0) for sp in cloud])
	d['Mc'] = cloud_data[6][massive] #np.array([cloud_mass(sp) for sp in cloud])
	d['ratio'] = cloud_data[5][massive] #d['Mc']*UM/1e10/np.array([cloud_Jeans(d['radcm'][i], cloud[i], z, h) for i in range(len(d['pos']))])
	d['rad'] = cloud_data[4][massive] #np.array([r_smooth(sp) for sp in cloud]) + d['radcm']/h/(1+z)
	d['rho'] = cloud_data[7][massive] #d['Mc']*UM/1e10/(4*np.pi*(d['rad']*UL)**3/3.0)
	d['T'] = cloud_data[8][massive] #np.array([cloud_temp(sp) for sp in cloud])
	d['r_ratio'] = cloud_data[9][massive]
	#totxt(rep+fn,np.vstack([ np_rotation(d['pos']), [d['radcm'], d['rad'], d['ratio'], d['Mc'], d['rho'], d['T']] ]), ['x [kpccm/h]', 'y', 'z', 'radcm [kpc/h]', 'rad [kpc]', 'ratio', 'Mc [Msun]', 'rho [g cm**-3]', 'T [K]'],1,0)
	return d
	
def projected_dis(sn, pos_base, pos_peak = [], rep = './', ls = 20.0, lc = 'g', ref = [0.718, 0.397, 0.206], ax = [0,1], nb = 500, boxs = 4000):
	llb = [r'$x\ [h^{-1}\mathrm{kpc}]$', r'$y\ [h^{-1}\mathrm{kpc}]$', r'$z\ [h^{-1}\mathrm{kpc}]$']
	posb = np_rotation(to_center(pos_base, np.array(ref)*boxs, boxs))
	plt.figure()
	plt.hist2d(posb[ax[0]],posb[ax[1]],bins=nb,norm=LogNorm(),cmap=plt.cm.hot)
	cb = plt.colorbar()
	plt.xlabel(llb[ax[0]])
	plt.ylabel(llb[ax[1]])
	if pos_peak!=[]:
		posp = np_rotation(to_center(pos_peak, np.array(ref)*boxs, boxs))
		plt.scatter(posp[ax[0]],posp[ax[1]],s=ls,edgecolors=lc,facecolors='none',lw=1)
	plt.tight_layout()
	plt.savefig(rep+'density_peak_'+str(sn)+'_'+str(ax[0])+str(ax[1])+'.pdf')
	#plt.show()

lmodel = ['CDM', 'WDM(3 kev)', 'DVX30']

def phase(data, sn, indm = 1, rep = './', edge = [0.01, 99.99], nbin = 100):
	ds = data['ds']
	z = ds['Redshift']
	rT = np.percentile(np.log10(data['T']), edge)
	rn = np.percentile(np.log10(data['n']), edge)
	rx = np.percentile(np.log10(data['H2']), edge)
	rxd = np.percentile(np.log10(data['HD']), edge)
	plt.figure()
	#plt.subplot(111)
	plt.hist2d(np.log10(data['n']),np.log10(data['T']),bins=100,norm=LogNorm(),range=[rn,rT])
	cb = plt.colorbar()
	cb.set_label(r'$\log(N)$')
	plt.xlabel(r'$\log(n\ [\mathrm{cm^{-3}}])$')
	plt.ylabel(r'$\log(T\ [\mathrm{K}])$')
	plt.title(r'$T-n$ of cold gas for '+lmodel[indm]+' at $z=$'+str(int(ds['Redshift']*100)/100),size=14)
	plt.tight_layout()
	plt.savefig(rep+'Tn_cold_'+lmodel[indm]+'_'+str(sn)+'.pdf')
	#plt.show()
	plt.figure()
	#plt.subplot(111)
	plt.hist2d(np.log10(data['n']),np.log10(data['H2']),bins=100,norm=LogNorm(),range=[rn,rx])
	cb = plt.colorbar()
	cb.set_label(r'$\log(N)$')
	plt.ylabel(r'$\log([\mathrm{H_{2}/H}])$')
	plt.xlabel(r'$\log(n\ [\mathrm{cm^{-3}}])$')
	plt.title(r'$[\mathrm{H_{2}/H}]-n$ of cold gas for '+lmodel[indm]+' at $z=$'+str(int(ds['Redshift']*100)/100),size=14)
	plt.tight_layout()
	plt.savefig(rep+'XH2n_cold_'+lmodel[indm]+'_'+str(sn)+'.pdf')
	#plt.show()
	plt.figure()
	#plt.subplot(111)
	plt.hist2d(np.log10(data['n']),np.log10(data['HD']),bins=100,norm=LogNorm(),range=[rn,rxd])
	cb = plt.colorbar()
	cb.set_label(r'$\log(N)$')
	plt.ylabel(r'$\log([\mathrm{HD/D}])$')
	plt.xlabel(r'$\log(n\ [\mathrm{cm^{-3}}])$')
	plt.title(r'$[\mathrm{HD/D}]-n$ of cold gas for '+lmodel[indm]+' at $z=$'+str(int(ds['Redshift']*100)/100),size=14)
	plt.tight_layout()
	plt.savefig(rep+'XHDn_cold_'+lmodel[indm]+'_'+str(sn)+'.pdf')

if __name__ == "__main__":
	sn = int(sys.argv[1])
	ref0 = [0.5, 0.5, 0.5]#np.array([float(sys.argv[i]) for i in range(2,5)])
	print(ref0)
	rep0 = 'figure_'+str(sn).zfill(3)+'/'

	data = gas_data(sn)
	ad = data['ds'].all_data()
	sink_num = len(ad[('PartType3','Masses')])
	print('Number of sink particles: {}'.format(sink_num))
	posg = ad[('PartType0','Coordinates')].to('kpccm/h')
	redshift = int(data['ds']['Redshift']*100)/100
	peaks = density_peak(data)
	projected_dis(sn,posg,peaks['pos'],ax=[0,1],ref=ref0,rep='./')
	projected_dis(sn,posg,peaks['pos'],ax=[1,2],ref=ref0,rep='./')
	projected_dis(sn,posg,peaks['pos'],ax=[2,0],ref=ref0,rep='./')
	
	phase(data, sn, edge=[0,100], nbin=200,rep=rep0)
	if len(sys.argv)<3:
		mode = 0
	else:
		mode = int(sys.argv[2])
	if mode==0:
		peaks = density_peak(data)
		dc = clouds(data, peaks, rmax = 3.0, min_N = 1)
	else:
		dc = read_clouds(data['ds'], mmin = 1)
	
	lcolor = []
	count = 0
	count1 = 0
	for i in range(dc['Mc'].shape[0]):
		if dc['ifsink'][i]==1:
			lcolor.append('b')
			count += 1
		elif dc['ifsink'][i]==2:
			lcolor.append('purple')
			count += 1
			count1 += 1
		else:
			lcolor.append('g')
	print('Number of clouds with sink particles: {}, number of compact clouds: {}'.format(count,count1))	

	ls = np.log10(dc['Mc'])*6
	projected_dis(sn,posg,dc['pos'],ax=[0,1],ref=ref0,rep=rep0,ls=ls,lc=lcolor)
	projected_dis(sn,posg,dc['pos'],ax=[1,2],ref=ref0,rep=rep0,ls=ls,lc=lcolor)
	projected_dis(sn,posg,dc['pos'],ax=[2,0],ref=ref0,rep=rep0,ls=ls,lc=lcolor)
	
	nc = 6
	rbase = 10**np.linspace(np.log10(0.02875), 1.0, 50)
	pos = dc['pos'][nc]
	lsp = [data['ds'].sphere(pos, (x, 'kpccm/h')) for x in rbase]
	lratio = [cloud_mass(lsp[i]).to('g')/cloud_Jeans(pos, lsp[i], redshift) for i in range(len(rbase))]
	plt.figure()
	plt.plot(rbase, lratio, label='cloud '+str(nc))
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'comoving $r_{\mathrm{cloud}}\ [h^{-1}\mathrm{kpc}]$')
	plt.ylabel(r'$\log(M_{\mathrm{cloud}}/M_{J})$')
	plt.tight_layout()
	plt.savefig(rep0+'ratio_rad_'+str(nc)+'.pdf')
	#plt.show()
	
	#print(dc['ifsink'])
	plt.figure()
	plt.scatter(np.log10(dc['Mc']),np.log10(dc['rad']),c=lcolor)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')
	plt.ylabel(r'$\log(r_{\mathrm{cloud}}\ [\mathrm{kpc}])$')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_rad_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.scatter(np.log10(dc['Mc']),np.log10(dc['ratio']),c=lcolor)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')
	plt.ylabel(r'$\log(M_{\mathrm{cloud}}/M_{J})$')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_ratio_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.scatter(np.log10(dc['Mc']),np.log10(dc['r_ratio']),c=lcolor)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')
	plt.ylabel(r'$\log(r_{\mathrm{cloud}}/r_{\mathrm{smooth}})$')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_r_ratio_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.scatter(np.log10(dc['Mc']),np.log10(dc['Mc']/dc['ratio']),c=lcolor)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')
	plt.ylabel(r'$\log(M_{J}\ [M_{\odot}])$')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_MJ_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.scatter(np.log10(dc['Mc']),np.log10(dc['rho']),c=lcolor)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')
	plt.ylabel(r'$\log(\bar{\rho}\ [\mathrm{g\ cm^{-3}}])$')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_rho_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.scatter(np.log10(dc['Mc']),np.log10(dc['T']),c=lcolor)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')
	plt.ylabel(r'$\log(\bar{T}\ [\mathrm{K}])$')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_T_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.hist(np.log10(dc['Mc']),bins=20,density=True)
	plt.xlabel(r'$\log(M_{\mathrm{cloud}}\ [M_{\odot}])$')	
	plt.ylabel(r'probability density')
	plt.tight_layout()
	plt.savefig(rep0+'Mc_z'+str(redshift)+'.pdf')

	plt.figure()
	plt.hist(np.log10(dc['rad']),bins=20,density=True)
	plt.xlabel(r'$\log(r_{\mathrm{cloud}}\ [\mathrm{kpc}])$')
	plt.ylabel(r'probability density')
	plt.tight_layout()
	plt.savefig(rep0+'rad_z'+str(redshift)+'.pdf')

	#plt.show()

#dc['rad'], dc['ratio'], dc['Mc'], dc['rho'], dc['T']
#'rad [kpc]', 'ratio', 'Mc [Msun]', 'rho [g cm**-3]', 'T [K]'


