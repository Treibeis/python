from tophat import *

# gaussian distribution of streaming motion velocity
def vdis(v, sigma = 30.):
	return v**2 * np.exp(-v**2*1.5/sigma**2)

# cosmic average number density of halos above the mass threshold by integrating 
# the number density over the distribution of streaming motion velocity
def Nhalo(z, lm0, lv0, mode = 0, Mdm = 0.3, h = 0.6774, sigma = 30., vmax = 5.):
	"""
		lm0: array of mass threshold (corresponding to lv0)
		lv0: array of streaming motion velocity
		mode=0: CDM, mode=1: BDMS
		vmax: boundary of the integration (over the gaussian distribution)
		sigma: rms of the streaming motion velocity at z=1100
	"""
	sel = lv0 < vmax*sigma
	lm = lm0[sel]
	lv = lv0[sel]
	lw = vdis(lv, sigma)
	Mmin = min(np.log10(np.min(lm*h))-1, np.log10(Mup(z))-1)
	if mode == 0:
		hmf_ = hmf.MassFunction()
		hmf_.update(n=0.966, sigma_8=0.829,cosmo_params={'Om0':0.315,'H0':67.74},Mmin=Mmin,Mmax=np.log10(Mup(z))+2,z=z)
	else:
		hmf_ = hmf.wdm.MassFunctionWDM(wdm_mass=Mdm*1e6)
		hmf_.update(n=0.966, sigma_8=0.829,cosmo_params={'Om0':0.315,'H0':67.74},Mmin=Mmin,Mmax=np.log10(Mup(z))+2,z=z)
	intm = np.log10(hmf_.m/h)
	intn = np.log10(hmf_.ngtm)
	nm = interp1d(intm, intn)
	lnpopIII = (10**nm(np.log10(lm))- 10**nm(np.log10(Mup(z))))
	lnpopIII = lnpopIII * (lnpopIII>0)
	out = lnpopIII * lw
	return np.trapz(out, lv)/np.trapz(lw, lv)
		
# wrapper of Nhalo that sets up a grid of streaming motion velocity 
# for integration under certain model parameters
def nh_z(z = 20, m1 = 1e2, m2 = 1e10, mode = 0, Mdm = 0.3, sigma = 8e-20, rat = 1.0, dmax = 2e4, Om = 0.3089, h = 0.6774, fac = 1e-3, vmin = 0.0, beta = .7, sk = False, v1 = 0, v2 = 150, nv = 16, ncore = 8):
	lv = np.linspace(v1, v2, nv) #np.geomspace(v0, v1, nv)
	np_core = int(nv/ncore)
	lpr = [[i*np_core, (i+1)*np_core] for i in range(ncore-1)] + [[(ncore-1)*np_core, nv]]
	print(lpr)
	manager = mp.Manager()
	output = manager.Queue()
	def sess(pr0, pr1):
		lm = []
		for i in range(pr0, pr1):
			init = initial(v0 = lv[i], mode = mode, Mdm = Mdm, sigma = sigma)
			d = Mth_z(z, z, 1, mode = mode, v0 = lv[i], rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init, Mdm = Mdm, sigma = sigma)
			lm.append(d[0][0])
		output.put([pr0, lm])
	pros = [mp.Process(target=sess, args=(lpr[k][0], lpr[k][1])) for k in range(ncore)]
	for p in pros:
		p.start()
	for p in pros:
		p.join()
	out = [output.get() for p in pros]
	out.sort()
	lm = np.hstack([x[1] for x in out])
	nh = Nhalo(z, np.array(lm), lv, mode, Mdm)
	return nh

# scan the parameter space of BDMS to calculate the cosmic average number 
# density of halos above the mass threshold
def nh_para(m1 = -4, m2 = 2, s1 = -1, s2 = 6, z = 20, dmax = 2e4, nbin = 2, fac = 1e-3, rat = 1.0, beta = .7, sk = False, ncore = 8):
	lm = np.logspace(m1, m2, nbin)
	ls = np.logspace(s1, s2, nbin)
	X, Y = np.meshgrid(lm, ls, indexing = 'ij')
	#lMh = np.zeros(X.shape)
	lnh = np.array([[nh_z(z, Mdm = m, sigma = s*1e-20, mode=1, rat=rat, fac=fac, dmax=dmax, beta=beta, sk=sk, ncore=ncore) for m in lm] for s in ls]).T
	return X, Y*1e-20, lnh

if __name__ == '__main__':
	rep = 'Nhrat_test/'
	ncore = 4
	nbin = 32
	z = 20
	dmax = delta0 * 100
	rat = 1.
	fac = 1e-3
	beta = 0.7
	sk = False
	if not os.path.exists(rep):
		os.makedirs(rep)
	d0 = nh_z(z=z, dmax=dmax, fac=fac, rat=rat, beta=beta, sk=sk, ncore=ncore)
	totxt(rep+'nh_ref_z'+str(z)+'.txt',[[d0]],0,0,0)
	X, Y, Mh = nh_para(z=z, dmax=dmax, nbin=nbin, fac=fac, rat=rat, beta=beta, sk=sk, ncore=ncore)
	totxt(rep+'nh_z'+str(z)+'.txt',Mh,0,0,0)
	totxt(rep+'X_z'+str(z)+'.txt',X,0,0,0)
	totxt(rep+'Y_z'+str(z)+'.txt',Y,0,0,0)
	lowb = 1e-8
	fh = Mh/d0
	fh = fh + lowb*(fh<lowb)
	plt.contourf(X, Y, np.log10(fh), nbin*2)
	cb = plt.colorbar()
	cb.set_label(r'$f_{h}$',size=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'fhMap.pdf')
	plt.close()
	



