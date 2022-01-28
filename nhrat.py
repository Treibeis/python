from main import *

def nh_z(z = 20, m1 = 1e2, m2 = 1e10, mode = 0, Mdm = 0.3, sigma = 8e-20, rat = 1.0, dmax = 2e4, Om = 0.315, h = 0.6774, fac = 1e-3, vmin = 0.0, alpha = .7, sk = False, v1 = 0, v2 = 150, nv = 16, ncore = 8):
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
			d = Mth_z(z, z, 1, mode = mode, v0 = lv[i], rat = rat, dmax = dmax, fac = fac, alpha = alpha, sk = sk, init = init, Mdm = Mdm, sigma = sigma)
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

def nh_para(m1 = -4, m2 = 2, s1 = -1, s2 = 6, z = 20, dmax = 2e4, nbin = 2, fac = 1e-3, rat = 1.0, alpha = .7, sk = False, ncore = 8):
	lm = np.logspace(m1, m2, nbin)
	ls = np.logspace(s1, s2, nbin)
	X, Y = np.meshgrid(lm, ls, indexing = 'ij')
	#lMh = np.zeros(X.shape)
	lMh = np.array([[nh_z(z, Mdm = m, sigma = s*1e-20, mode=1, rat=rat, fac=fac, dmax=dmax, alpha=alpha, sk=sk, ncore=ncore) for m in lm] for s in ls]).T
	return X, Y*1e-20, lMh

if __name__ == '__main__':
	rep = 'Nhrat_test/'
	ncore = 8
	nbin = 32
	z = 20
	dmax = delta0 * 100
	rat = 1.
	fac = 1e-3
	alpha = 0.7
	sk = False
	if not os.path.exists(rep):
		os.makedirs(rep)
	d0 = nh_z(z=z, dmax=dmax, fac=fac, rat=rat, alpha=alpha, sk=sk, ncore=ncore)
	totxt(rep+'nh_ref_z'+str(z)+'.txt',[[d0]],0,0,0)
	X, Y, Mh = nh_para(z=z, dmax=dmax, nbin=nbin, fac=fac, rat=rat, alpha=alpha, sk=sk, ncore=ncore)
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
	plt.show()
	



