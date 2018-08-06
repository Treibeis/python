from process import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
import time
import multiprocessing as mp
from matplotlib.colors import LogNorm

# Cosmology
def H(a, Om = 0.315, h = 0.6774):
	H0 = h*100*UV/UL/1e3
	H = H0*(Om/a**3+(1-Om))**0.5
	return H

def DZ(z, Om = 0.315, h = 0.6774):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2)/H(a, Om, h)
	I = quad(integrand, 1/(1+z), 1, epsrel = 1e-8)
	return I[0]

def TZ(z, Om = 0.315, h = 0.6774):
	def integrand(a):
		return 1/a/H(a, Om, h)
	I = quad(integrand, 0, 1/(1+z), epsrel = 1e-8)
	return I[0]

# Bremsstrahlung
def gff(nu, T):
	lognu = np.log10(nu*HBAR*2*np.pi/T/BOL)
	llognu = np.linspace(-5, 4, 10)
	lgff = [5.8]+[4.5, 3.2, 2.2, 1.7, 1.4, 1.4, 1.5]+[1.45,1.4]
	gff_lognu = interp1d(llognu, lgff, kind='cubic')
	#if lognu<-4:
	#	return -1.3*lognu-0.7
	#elif lognu>2:
	#	a = (1-1/(1+np.exp(3)))
	#	return 1.5  -0.1*(1/(1+np.exp(5-lognu)) -1/(1+np.exp(3)) )/a
	#else:
	#	return gff_lognu(lognu)
	lognu_ = -4*(lognu<-4) + lognu*(-4<=lognu)*(lognu<=2) + 2*(lognu>2)
	a = (1-1/(1+np.exp(3)))
	return (-1.3*lognu-0.7)*(lognu<-4) + gff_lognu(lognu_)*(-4<=lognu)*(lognu<=2) \
			+ (1.5  -0.1*(1/(1+np.exp(5-lognu)) -1/(1+np.exp(3)) )/a)*(lognu>2)

def jnu(nu, T, ne, ni, Z = 1):
	enu = 2**5*np.pi*CHARGE**6/3/ELECTRON/SPEEDOFLIGHT**3 * (2*np.pi/3/BOL/ELECTRON)**0.5 / T**0.5 * ne * ni * np.exp(-HBAR*2*np.pi*nu/BOL/T) * gff(nu, T)
	return enu/4/np.pis

def neniT(ne, ni, T):
	return 2**5*np.pi*CHARGE**6/3/ELECTRON/SPEEDOFLIGHT**3 * (2*np.pi/3/BOL/ELECTRON)**0.5 / T**0.5 * ne * ni

def jnu_(nu, T, x):
	return x * np.exp(-HBAR*2*np.pi*nu/BOL/T) * gff(nu, T)/4/np.pi

def Bnu(nu, T):
	x = HBAR*2*np.pi*nu/BOL/T * (HBAR*2*np.pi*nu/BOL/T<100) + 100*(HBAR*2*np.pi*nu/BOL/T>=100)
	return 2*HBAR*2*np.pi*nu**3/SPEEDOFLIGHT**2/(np.exp(x)-1)

# Mesh
def kernel(r, h):
	q = r/h
	a = 8/np.pi/h**3
	b = ((0<=q)*(q<0.5))*(1+6*q**2*(q-1)) + ((0.5<=q)*(q<1))*2*(1-q)**3
	#if 0<=q<0.5:
	#	b = 1+6*q**2*(q-1)
	#elif 0.5<=q<1:
	#	b = 2*(1-q)**3
	#else:
	#	b = 0
	return a*b

def mesh_3d(x0 = 1750, x1 = 2250, xbin = 100, y0 = 1750, y1 = 2250, ybin = 100, z0 = 1750, z1 = 2250, zbin = 100):
	dx, dy, dz = (x1-x0)/(xbin), (y1-y0)/(ybin), (z1-z0)/(zbin)
	xref = np.linspace(x0+dx/2, x1-dx/2, xbin)
	yref = np.linspace(y0+dy/2, y1-dy/2, ybin)
	zref = np.linspace(z0+dz/2, z1-dz/2, zbin)
	X, Y, Z = np.meshgrid(xref, yref, zref)
	N = xbin*ybin*zbin
	lx = np.reshape(X, N)
	ly = np.reshape(Y, N)
	lz = np.reshape(Z, N)
	lpos = np.array([lx,ly,lz]).T
	d = {}
	d['pos'] = lpos
	d['bins'] = np.array([dx, dy, dz])
	d['bound'] = np.array([[x0,x1],[y0,y1],[z0,z1]])
	d['nb'] = np.array([xbin, ybin, zbin])
	d['mesh'] = [X, Y, Z]
	return d

def grid(mesh, sn, rep = './', box = [[1750]*3,[2250]*3], Tsh = 1e3, mode = 0, base = 'snapshot', ext = '.hdf5', Tmin = 10.0, ncore = 1):#, h = 0.6774):
	start = time.time()
	ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	#z = ds['Redshift']
	#posg = mesh['pos']
	dr = mesh['bins']
	ad = ds.box(box[0],box[1])#ds.box([(x, 'kpccm/h') for x in box[0]],[(x, 'kpccm/h') for x in box[1]])#ds.all_data()
	N = mesh['pos'].shape[0]
	shock = np.array(temp(ad[('PartType0','InternalEnergy')],ad[('PartType0','Primordial HII')]))>Tsh
	posp = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))[shock]
	nump = posp.shape[0]
	print('Number of gas particles (pre): {}'.format(nump))
	head = [[str(mesh['bound'][x][0]), str(mesh['bound'][x][1]), str(mesh['nb'][x])] for x in range(3)]
	head = head[0]+head[1]+head[2]
	#tcount = 1
	Vc = dr[0]*dr[1]*dr[2]
	low = np.array([mesh['bound'][x][0] for x in range(3)])
	up = np.array([mesh['bound'][x][1] for x in range(3)])
	np_core = int(nump/ncore)
	lpr = [[i*np_core, (i+1)*np_core] for i in range(ncore-1)] + [[(ncore-1)*np_core, nump]]
	print(lpr)
	manager = mp.Manager()
	output = manager.Queue()
	def sess(pr0, pr1):
		mni = np.zeros(mesh['nb'])
		mne = np.zeros(mesh['nb'])
		mT = np.zeros(mesh['nb'])
		meps = np.zeros(mesh['nb'])
		mtag = np.zeros(mesh['nb'])
		mpe = np.zeros(mesh['nb'])
		count = 0
		for i in range(pr0,pr1):
			label = [(posp[i][j]-low[j]-dr[j]/2)/dr[j] for j in range(3)]
			if label[0]<0 or label[1]<0 or label[2]<0:
				continue
			if label[0]>=mesh['nb'][0]-1 or label[1]>=mesh['nb'][1]-1 or label[2]>=mesh['nb'][2]-1:
				continue
			count += 1
			ind = [int(x) for x in label]
			n = np.array(ad[('PartType0','Density')][shock][i].to_equivalent("cm**-3", "number_density",mu=mmw(ad[('PartType0','Primordial HII')][shock][i])))
			m = ad[('PartType0','Masses')][shock][i]
			rho = ad[('PartType0','Density')][shock][i]
			T = np.array(temp(ad[('PartType0','InternalEnergy')][shock][i],ad[('PartType0','Primordial HII')][shock][i]))
			if T<=Tsh:
				print('?')
			h = np.array(ad[('PartType0','SmoothingLength')][shock][i].to('kpccm/h'))
			V = np.array((m/rho).to('(kpccm/h)**3'))
			ni = n*ad[('PartType0','Primordial HII')][shock][i]
			ne = n*ad[('PartType0','Primordial e-')][shock][i]
			for x in [ind[0], ind[0]+1]:
				for y in [ind[1], ind[1]+1]:
					for z in [ind[2], ind[2]+1]:
						posg_ = [x, y, z]
						posg = np.array([(posg_[j]+0.5)*dr[j]+low[j] for j in range(3)])
						dis = np.abs(posp[i] - posg)
						if mode==0:
							side = [dr[j]-dis[j] for j in range(3)]
							weight = side[0]*side[1]*side[2]/Vc
							mni[x][y][z] += ni * weight *V/Vc
							mne[x][y][z] += ne * weight *V/Vc
							mpe[x][y][z] += ne*BOL*T * weight *V/Vc
							mT[x][y][z] += T * weight
							mtag[x][y][z] += weight
							meps[x][y][z] += neniT(ne, ni, T) * weight * V/Vc
							#print(meps[x][y][z])
						else:
							r = np.sum(dis**2)**0.5
							weight = kernel(r,h)*V
							mni[x][y][z] += ni * weight
							mne[x][y][z] += ne * weight
							mpe[x][y][z] += ne*BOL*T * weight
							mT[x][y][z] += T * weight
							mtag[x][y][z] = 1.0
							meps[x][y][z] += neniT(ne, ni, T) * weight
		output.put([count, mni, mne, mT, meps, mpe, mtag])
	processes = [mp.Process(target=sess, args=(lpr[i][0], lpr[i][1])) for i in range(ncore)]
	for p in processes:
		p.start()
	for p in processes:
		p.join()
	mni = np.zeros(mesh['nb'])
	mne = np.zeros(mesh['nb'])
	mT = np.zeros(mesh['nb'])
	meps = np.zeros(mesh['nb'])
	mtag = np.zeros(mesh['nb'])
	mpe = np.zeros(mesh['nb'])
	count = 0
	for p in processes:
		buff = output.get()
		mni += buff[1]
		mne += buff[2]
		mT += buff[3]
		meps += buff[4]
		mpe += buff[5]
		mtag += buff[6]
		count += buff[0]
	lni = np.reshape(mni, N)
	lne = np.reshape(mne, N)
	lT = np.reshape(mT, N)
	leps = np.reshape(meps, N)
	lpe = np.reshape(mpe, N)
	ltag = np.reshape(mtag, N)
	for i in range(N):
		if ltag[i]>0:
			lT[i] = max(lT[i]/ltag[i],Tmin)
		else:
			lT[i] = Tmin
	d = {}
	d['ni'], d['ne'], d['T'] = lni, lne, lT
	d['tag'] = ltag
	d['eps'] = leps
	d['pe'] = lpe
	d['bins'] = mesh['bins']# np.array([dx, dy, dz])
	d['bound'] = mesh['bound']#np.array([[x0,x1],[y0,y1],[z0,z1]])
	d['nb'] = mesh['nb']#np.array([xbin, ybin, zbin])
	d['mesh'] = mesh['mesh']#[X, Y, Z]
	d['ds'] = ds
	nem = len(ltag[ltag==0])
	end = time.time()
	print('Number of gas particles (post): {}'.format(count))
	print('Fraction of empty grids: {}'.format(nem/N))
	print('Time: {} s'.format(end-start))
	totxt(rep+'mesh'+str(mode)+'_'+str(sn)+'_'+str(d['nb'][0])+'.txt',[d['ni'], d['ne'], d['T'], d['eps'], d['pe'], d['tag']],head,1,0)
	return d

def read_grid(sn, nb, rep = './', mode=0, base = 'snapshot', ext = '.hdf5'):
	ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	l = np.array(retxt(rep+'mesh'+str(mode)+'_'+str(sn)+'_'+str(nb)+'.txt',6,1,0))
	head = restr(rep+'mesh'+str(mode)+'_'+str(sn)+'_'+str(nb)+'.txt')
	x0 = float(head[0][0])
	x1 = float(head[0][1])
	xbin = int(head[0][2])
	y0 = float(head[0][3])
	y1 = float(head[0][4])
	ybin = int(head[0][5])
	z0 = float(head[0][6])
	z1 = float(head[0][7])
	zbin = int(head[0][8])
	dx, dy, dz = (x1-x0)/(xbin), (y1-y0)/(ybin), (z1-z0)/(zbin)
	xref = np.linspace(x0+dx/2, x1-dx/2, xbin)
	yref = np.linspace(y0+dy/2, y1-dy/2, ybin)
	zref = np.linspace(z0+dz/2, z1-dz/2, zbin)
	X, Y, Z = np.meshgrid(xref, yref, zref)
	d = {}
	d['ni'], d['ne'], d['T'] = l[0], l[1], l[2]
	d['eps'] = l[3]
	d['pe'] = l[4]
	d['tag'] = l[5]
	d['bins'] = np.array([dx, dy, dz])
	d['bound'] = np.array([[x0,x1],[y0,y1],[z0,z1]])
	d['nb'] = np.array([xbin, ybin, zbin])
	d['mesh'] = [X, Y, Z]
	d['ds'] = ds
	return d

def numax(T = 1e4):
	return 2.0*BOL*T/HBAR/2/np.pi

def numin(T = 1e4, n = 1e-6, lc = 1000, z = 10, h = 0.6774):
	l = lc*UL/h/(1+z)
	return (4*CHARGE**6/3/ELECTRON/SPEEDOFLIGHT/BOL * (2*np.pi/3/BOL/ELECTRON)**0.5/T**1.5 * n**2 *l)**0.5

def intensity(dg, nu, h = 0.6774):
	z = dg['ds']['Redshift']
	darea = dg['bins'][0]*dg['bins'][1]*(UL/h/(1+z))**2
	ldtau = jnu_(nu, dg['T'], dg['eps'])/Bnu(nu, dg['T']) * dg['bins'][2]*UL/h/(1+z)
	mdtau = np.reshape(ldtau, dg['nb'])
	mpe = np.reshape(dg['pe'], dg['nb'])
	mSnu = Bnu(nu, np.reshape(dg['T'], dg['nb']))
	dx, dy = dg['bins'][0], dg['bins'][1]
	x0, x1 = dg['bound'][0][0], dg['bound'][0][1]
	y0, y1 = dg['bound'][1][0], dg['bound'][1][1]
	xref = np.linspace(x0+dx/2, x1-dx/2, dg['nb'][0])
	yref = np.linspace(y0+dy/2, y1-dy/2, dg['nb'][1])
	X, Y = np.meshgrid(xref, yref)
	Xraymap = np.zeros(X.shape)
	ymap = np.zeros(X.shape)
	for i in range(dg['nb'][0]):
		for j in range(dg['nb'][1]):
			Xraymap[i][j] = np.sum([ np.exp(np.sum(mdtau[i][j][:k])-np.sum(mdtau[i][j])) * mSnu[i][j][k]*mdtau[i][j][k] for k in range(dg['nb'][2])])
			ymap[i][j] = np.sum(mpe[i][j]*SIGMATH/ELECTRON/SPEEDOFLIGHT**2 * dg['bins'][2]*UL/h/(1+z))
	d = {}
	d['X'], d['Y'], d['I'] = X, Y, Xraymap
	d['SZ'] = ymap
	d['L'] = np.sum(Xraymap)*darea*4*np.pi
	return d
			
def SZy(dg, h = 0.6774):
	z = dg['ds']['Redshift']
	mpe = np.reshape(dg['pe'], dg['nb'])
	dx, dy = dg['bins'][0], dg['bins'][1]
	x0, x1 = dg['bound'][0][0], dg['bound'][0][1]
	y0, y1 = dg['bound'][1][0], dg['bound'][1][1]
	xref = np.linspace(x0+dx/2, x1-dx/2, dg['nb'][0])
	yref = np.linspace(y0+dy/2, y1-dy/2, dg['nb'][1])
	X, Y = np.meshgrid(xref, yref)
	ymap = np.zeros(X.shape)
	for i in range(dg['nb'][0]):
		for j in range(dg['nb'][1]):
			ymap[i][j] = np.sum(mpe[i][j]*SIGMATH/ELECTRON/SPEEDOFLIGHT**2 * dg['bins'][2]*UL/h/(1+z))
	d = {}
	d['X'], d['Y'] = X, Y
	d['SZ'] = ymap
	return d

def luminosity(dg, rnu = [np.log10(numax())-2.5, np.log10(numax())+0.5], nb = 1000, ncore = 4, h = 0.6774):
	z = dg['ds']['Redshift']
	Vc = (dg['bins'][0] * dg['bins'][1] * dg['bins'][2])*(UL/h/(1+z))**3
	manager = mp.Manager()
	output = manager.Queue()
	lnu0 = np.linspace(10**rnu[0], 10**rnu[1], ncore+1)
	nb_p = int(nb/ncore)
	def sess(i):
		edge = np.linspace(lnu0[i],lnu0[i+1],nb_p+1)
		lnu = (edge[:-1]+edge[1:])/2.0
		dnu = edge[1:]-edge[:-1]
		out = np.sum([Vc*np.sum(jnu_(lnu[x], dg['T'], dg['eps'])*4*np.pi)*dnu[x] for x in range(nb_p)])
		output.put(out)
	processes = [mp.Process(target=sess, args=(i,)) for i in range(ncore)]
	for p in processes:
		p.start()
	for p in processes:
		p.join()
	out = [z, np.sum([output.get() for p in processes])]
	return out
	
if __name__ == "__main__":
	ncore = 4
	Tsh = 1e4
	indm = 1
	sn = int(sys.argv[1])
	mode = int(sys.argv[2])
	bins = int(sys.argv[3])
	#nn = int(sys.argv[3])
	if len(sys.argv)>5:
		low, up = int(sys.argv[4]), int(sys.argv[5])
	else:
		low, up = 1900, 2000
	if mode==0:
		mesh = mesh_3d(low, up, bins, low, up, bins, low, up, bins)
		d0 = grid(mesh,sn,box =[[low]*3,[up]*3],ncore=ncore,Tsh=Tsh)
		#d1 = grid(mesh,sn,box =[[low]*3,[up]*3],mode=1,ncore=ncore,Tsh=Tsh)
	else:
		d0 = read_grid(sn,nb = bins)
		#d1 = read_grid(sn,nb = bins, mode=1)

	redshift = int(d0['ds']['Redshift']*1000)/1000

	nump = 32
	lnu = 10**np.linspace(np.log10(numin())+0.5,np.log10(numax())+1.0,nump)
	np_core = int(nump/ncore)
	lpr = [[i*np_core, (i+1)*np_core] for i in range(ncore-1)] + [[(ncore-1)*np_core, nump]]
	manager = mp.Manager()
	output = manager.Queue()
	def sess(low, up, i):
		output.put([i]+[intensity(d0, lnu[x])['L'] for x in range(low, up)])
	processes = [mp.Process(target=sess, args=(lpr[i][0], lpr[i][1], i)) for i in range(ncore)]
	for p in processes:
		p.start()
	for p in processes:
		p.join()
	raw = [output.get() for p in processes]
	raw = sorted(raw, key=lambda x: x[0])
	lmap = np.hstack([x[1:] for x in raw])
	plt.plot(lnu,lmap)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\nu\ [\mathrm{Hz}]$')
	plt.ylabel(r'$L_{\nu}\ [\mathrm{erg\ s^{-1}\ Hz^{-1}}]$')
	plt.title(lmodel[indm]+r': $z='+str(redshift)+'$',size=14)
	plt.tight_layout()
	plt.savefig('luminosity_'+str(lmodel[indm])+'_'+str(sn)+'_'+str(bins)+'.pdf')
	#plt.show()

	nu = int(numax()/100/1e9)
	test = intensity(d0, numax()/100)
	test_ = SZy(d0)
	plt.figure()
	plt.contourf(test['Y'],test['X'],np.log10(test['I']),np.linspace(-36,-19,100),cmap=plt.cm.gist_ncar)#norm=LogNorm()
	cb = plt.colorbar()
	cb.set_label(r'$\log(I_{\nu}\ [\mathrm{erg\ s^{-1}\ cm^{-2}\ Hz^{-1}\ sr^{-1}}])$')
	plt.xlabel(r'$x\ [h^{-1}\mathrm{kpc}]$')
	plt.ylabel(r'$y\ [h^{-1}\mathrm{kpc}]$')
	plt.title(lmodel[indm]+r': $z='+str(redshift)+'$'+r', $\nu='+str(nu)+'\ \mathrm{GHz}$',size=14)
	plt.tight_layout()
	plt.savefig('intensity_'+str(lmodel[indm])+'_'+str(sn)+'_'+str(bins)+'.pdf')
	#plt.show()

	plt.figure()
	plt.contourf(test_['Y'],test_['X'],np.log10(test_['SZ']),np.linspace(-19.5,-9.5,100),cmap=plt.cm.gnuplot)#norm=LogNorm()
	cb = plt.colorbar()
	cb.set_label(r'$\log(y_{\mathrm{LoS}})$')
	plt.xlabel(r'$x\ [h^{-1}\mathrm{kpc}]$')
	plt.ylabel(r'$y\ [h^{-1}\mathrm{kpc}]$')
	plt.title(lmodel[indm]+r': $z='+str(redshift)+'$',size=14)
	plt.tight_layout()
	plt.savefig('SZ_'+str(lmodel[indm])+'_'+str(sn)+'_'+str(bins)+'.pdf')
	#plt.show()

	h = 0.6774
	z = d0['ds']['Redshift']
	dc = DZ(z)*(1+z)
	dz = d0['bins'][2]*UL/(1+z)/h
	angle = (dz/dc)**2
	Vc = (d0['bins'][0] * d0['bins'][1] * d0['bins'][2])*(UL/h/(1+z))**3
	print('Flux: {} [erg s^-1 cm^-2 Hz^-1]'.format((1+z)*angle*np.sum(test['I'])))
	#print('Flux1: {} [erg s^-1 cm^-2 Hz^-1]'.format(angle*dz*np.sum(d1['eps'])/4/np.pi ))
	print('Power: {} [erg s^-1]'.format(luminosity(d0)[1]))
	print('Average yLoS: {}'.format(np.average(test_['SZ'])))
	#print('Power1: {} [erg s^-1 Hz^-1]'.format(Vc*np.sum(d1['eps'])))
	
	rne = [-10, 0]
	rT = [int(np.log10(Tsh)), 5]
	reps = [-58, -42]
	histb = 200

	plt.figure()
	a = plt.hist(np.log10(d0['ne'][d0['ne']>1e-10]),bins=histb,alpha=0.5,label='CIC',range=rne)#,density=True)
	#b = plt.hist(np.log10(d1['ne'][d1['ne']>1e-10]),bins=histb,alpha=0.5,label='SPH grid',range=rne)#,density=True)
	plt.xlabel(r'$\log(n_{\mathrm{e}}\ [\mathrm{cm^{-3}}])$')
	#plt.ylabel(r'Probability density')
	dh = (rne[1]-rne[0])/histb
	plt.ylabel(r'$\frac{d N_{\mathrm{grid}}}{d\log n_{\mathrm{e}}}\ [\mathrm{('+str(dh)+'\ dex)^{-1}}]$')
	#plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig('logne_'+str(lmodel[indm])+'_'+str(sn)+'_'+str(bins)+'.pdf')
	#plt.show()
	
	plt.figure()
	a = plt.hist(np.log10(d0['T'][d0['T']>30]),bins=histb,alpha=0.5,label='CIC',range=rT)#,density=True)
	#b = plt.hist(np.log10(d1['T'][d1['T']>30]),bins=histb,alpha=0.5,label='SPH grid',range=rT)#,density=True)
	plt.xlabel(r'$\log(T\ [\mathrm{K}])$')
	#plt.ylabel(r'Probability density')
	dh = (rT[1]-rT[0])/histb
	plt.ylabel(r'$\frac{d N_{\mathrm{grid}}}{d\log T}\ [\mathrm{('+str(dh)+'\ dex)^{-1}}]$')
	#plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig('logT_'+str(lmodel[indm])+'_'+str(sn)+'_'+str(bins)+'.pdf')
	#plt.show()

	plt.figure()
	a = plt.hist(np.log10(d0['eps'][d0['eps']>1e-60]),bins=histb,alpha=0.5,label='CIC',range=reps)#,density=True)
	#b = plt.hist(np.log10(d1['eps'][d1['eps']>1e-60]),bins=histb,alpha=0.5,label='SPH grid',range=reps)#,density=True)
	plt.xlabel(r'$\log(\epsilon_{\nu}\ [\mathrm{erg\ s^{-1}\ cm^{-3}\ Hz^{-1}}])$')
	#plt.ylabel(r'Probability density')
	dh = (reps[1]-reps[0])/histb
	plt.ylabel(r'$\frac{d N_{\mathrm{grid}}}{d\log \epsilon_{\nu}}\ [\mathrm{('+str(dh)+'\ dex)^{-1}}]$')
	#plt.yscale('log')
	#plt.xlim(-50,-40)
	plt.legend()
	plt.tight_layout()
	plt.savefig('logeps_'+str(lmodel[indm])+'_'+str(sn)+'_'+str(bins)+'.pdf')
	#plt.show()



"""
def grid_(mesh, sn, rep = './', box = [[1750]*3,[2250]*3], nn = 129, mode = 0, base = 'snapshot', ext = '.hdf5', wall = 3600, Tmin = 10.0, fac = 1.0):#, h = 0.6774):
	start = time.time()
	ds = yt.load(rep+base+'_'+str(sn).zfill(3)+ext)
	#z = ds['Redshift']
	posg = mesh['pos']
	dr = mesh['bins']
	ad = ds.box(box[0],box[1])#ds.box([(x, 'kpccm/h') for x in box[0]],[(x, 'kpccm/h') for x in box[1]])#ds.all_data()
	posp = np.array(ad[('PartType0','Coordinates')].to('kpccm/h'))
	nump = posp.shape[0]
	print('Number of gas particles: {}'.format(nump))
	N = posg.shape[0]
	pos = np.vstack([posg, posp])
	lni = np.zeros(N)
	lne = np.zeros(N)
	lT = np.zeros(N)#np.ones(N)*10.0
	ltag = np.zeros(N)
	lneniT = np.zeros(N)
	print('Neighbor finding: Begin')
	nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(pos)
	distances, indices = nbrs.kneighbors(pos)
	print('Neighbor finding: Done: {} s'.format(time.time()-start))
	start = time.time()
	head = [[str(mesh['bound'][x][0]), str(mesh['bound'][x][1]), str(mesh['nb'][x])] for x in range(3)]
	head = head[0]+head[1]+head[2]
	tcount = 1
	Vc = dr[0]*dr[1]*dr[2]
	low = np.array([mesh['bound'][x][0] for x in range(3)])
	up = np.array([mesh['bound'][x][1] for x in range(3)])
	if mode==0:
		for i in range(N):
			if (time.time()-start)/wall>tcount:
				totxt('mesh'+str(mode)+'_'+str(sn)+'_'+str(tcount)+'.txt',[lni, lne, lT],head,1,0)
				tcount += 1
			if indices[i][1]<N:
				continue
			for j in range(1,nn):
				if distances[i][j]>np.sum(dr**2)**0.5:
					break
				if indices[i][j]<N:
					continue
				#dis = np.abs(pos[indices[i][j]] - pos[i])
				#if dis[0]>=dr[0] or dis[1]>=dr[1] or dis[2]>=dr[2]:
				#	continue
				r = distances[i][j]
				n = np.array(ad[('PartType0','Density')][indices[i][j]-N].to_equivalent("cm**-3", "number_density",mu=mmw(ad[('PartType0','Primordial HII')][indices[i][j]-N])))
				m = ad[('PartType0','Masses')][indices[i][j]-N]
				rho = ad[('PartType0','Density')][indices[i][j]-N]
				T = np.array(temp(ad[('PartType0','InternalEnergy')][indices[i][j]-N],ad[('PartType0','Primordial HII')][indices[i][j]-N]))
				h = np.array(ad[('PartType0','SmoothingLength')][indices[i][j]-N].to('kpccm/h'))
				V = np.array((m/rho).to('(kpccm/h)**3'))
				#weight = kernel(r,max(dr)/fac)*V
				weight = kernel(r,h)*V
				ni = n*ad[('PartType0','Primordial HII')][indices[i][j]-N]
				ne = n*ad[('PartType0','Primordial e-')][indices[i][j]-N]
				lni[i] += ni * weight
				lne[i] += ne * weight
				lT[i] += T * weight/V
				ltag[i] += weight/V
				lneniT[i] += neniT(ne, ni, T) * weight
			#lT[i] = lT[i]/ltag[i]
	else:
		for i in range(nump):
			if (time.time()-start)/wall>tcount:
				totxt('mesh'+str(mode)+'_'+str(sn)+'_'+str(tcount)+'.txt',[lni, lne, lT],head,1,0)
				tcount += 1
			count = 0
			label = 0
			lindex = np.ones(8)*(-1)
			if min(min(abs((posp[i]-low)/dr)),min(abs((posp[i]-up)/dr)))<1:
				label = 1
			for j in range(1,nn):
				if indices[i+N][j]>=N:
					continue
				dis = np.abs(posp[i] - pos[indices[i+N][j]])
				if dis[0]>dr[0] or dis[1]>dr[1] or dis[2]>dr[2]:
					continue
				if count==8 or distances[i+N][j]>np.sum(dr**2)**0.5:
					break
				lindex[count] = indices[i+N][j]
				count += 1
				dis = np.abs(posp[i] - pos[indices[i+N][j]])
				side = np.zeros(3)
				for x in range(3):
					 side[x] = (dr[x]-dis[x])
				weight = side[0]*side[1]*side[2]/Vc
				n = np.array(ad[('PartType0','Density')][i].to_equivalent("cm**-3", "number_density",mu=mmw(ad[('PartType0','Primordial HII')][i])))
				m = ad[('PartType0','Masses')][i]
				rho = ad[('PartType0','Density')][i]
				T = np.array(temp(ad[('PartType0','InternalEnergy')][i],ad[('PartType0','Primordial HII')][i]))
				#h = np.array(ad[('PartType0','SmoothingLength')][indices[i][j]-N].to('kpccm/h'))
				V = np.array((m/rho).to('(kpccm/h)**3'))
				ni = n*ad[('PartType0','Primordial HII')][i]
				ne = n*ad[('PartType0','Primordial e-')][i]
				lni[indices[i+N][j]] += ni * weight *V/Vc
				lne[indices[i+N][j]] += ne * weight *V/Vc
				lT[indices[i+N][j]] += T * weight
				ltag[indices[i+N][j]] += weight
				lneniT[indices[i+N][j]] += neniT(ne, ni, T) * weight * V/Vc
			if count==8 or label>0:
				continue
			print('Not enough neighbors ({}) for gas particle {} at {}'.format(count,i,(posp[i]-low)/dr))
			#continue
			r0 = posp[i]
			pos1 = np.vstack([[r0],posg])
			nbrs1 = NearestNeighbors(n_neighbors=28, algorithm='ball_tree').fit(pos1)
			distances1, indices1 = nbrs1.kneighbors(pos1)
			for j in range(1,28):
				if count==8:
					break
				dis = np.abs(r0 - pos1[indices1[0][j]])
				if dis[0]>dr[0] or dis[1]>dr[1] or dis[2]>dr[2]:
					continue
				if np.max([indices1[0][j]-1 == x for x in lindex])>0:
					continue
				count += 1
				side = np.zeros(3)
				for x in range(3):
					 side[x] = (dr[x]-dis[x])
				weight = side[0]*side[1]*side[2]/Vc
				n = np.array(ad[('PartType0','Density')][i].to_equivalent("cm**-3", "number_density",mu=mmw(ad[('PartType0','Primordial HII')][i])))
				m = ad[('PartType0','Masses')][i]
				rho = ad[('PartType0','Density')][i]
				T = np.array(temp(ad[('PartType0','InternalEnergy')][i],ad[('PartType0','Primordial HII')][i]))
				#h = np.array(ad[('PartType0','SmoothingLength')][indices[i][j]-N].to('kpccm/h'))
				V = np.array((m/rho).to('(kpccm/h)**3'))
				ni = n*ad[('PartType0','Primordial HII')][i]
				ne = n*ad[('PartType0','Primordial e-')][i]
				lni[indices1[0][j]-1] += ni * weight *V/Vc
				lne[indices1[0][j]-1] += ne * weight *V/Vc
				lT[indices1[0][j]-1] += T * weight
				ltag[indices1[0][j]-1] += weight
				lneniT[indices1[0][j]-1] += neniT(ne, ni, T) * weight *V/Vc
			if count<8:
				print('Weird behavior ({})!'.format(count))
	for i in range(N):
		if ltag[i]>0:
			lT[i] = max(lT[i]/ltag[i],Tmin)
		else:
			lT[i] = Tmin
	d = {}
	d['ni'], d['ne'], d['T'] = lni, lne, lT
	d['tag'] = ltag
	d['eps'] = lneniT
	nem = len(ltag[ltag==0])
	end = time.time()
	print('Fraction of empty grids: {}'.format(nem/N))
	print('Time: {} s'.format(end-start))
	totxt('mesh'+str(mode)+'_'+str(sn)+'.txt',[d['ni'], d['ne'], d['T'], d['eps']],head,1,0)
	return d
"""

